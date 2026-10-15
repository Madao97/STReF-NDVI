# -*- coding: utf-8 -*-
r"""
Reconstruct NDVI time series using a trained STReF-NDVI model.

Required inputs:
    S1.npy            (T, 5, H, W): normalized Sentinel-1 features
    S2.npy            (T, H, W, 3): NDVI, cloud mask, water mask
    target_mask.npy   (T, H, W): optical supervision validity
    S1_valid.npy      (T, H, W): Sentinel-1 input validity
    good_pixel.npy    (H, W): pixels selected for reconstruction
    metadata.json
    Trained model checkpoint (.pth)

The model architecture must match the training implementation.
Feature channels are inferred from the checkpoint filename unless
explicitly supplied with --channels.

Predictions are generated for good_pixel locations at all time steps.
Other locations remain NaN. The inference attention mask uses S1_valid.

Outputs:
    ndvi_pred.npy
    ndvi_true.npy
    NDVI_Predict_<DATE>.tif, when reference georeferencing is available
    compare/compare_<DATE>.png
    per_frame_metrics.csv
    reconstruction_summary.json

Metrics use locations where target_mask is True and both observations
and predictions are finite. They are reconstruction diagnostics and
do not automatically represent independent held-out test performance.

"""

import os
import json
import argparse
from contextlib import nullcontext
from datetime import datetime
from math import pi

import numpy as np
import torch
import torch.nn as nn

from torch_geometric.nn import GCNConv
from tqdm import tqdm
import matplotlib.pyplot as plt


# =============================================================
# DEFAULT PATHS
# Override these paths using command-line arguments.
# =============================================================
DATA_DIR = os.path.join("data", "Dataset")
MODEL = os.path.join(DATA_DIR, "models", "vh__vv_diff.pth")
OUT_DIR = os.path.join("outputs", "Reconstruction")
S2_TIF = os.path.join("data", "reference.tif")

ALL_CHANNELS = ["vh", "rvi", "vv_div", "vv_diff", "log_ratio"]
EPS = 1e-6


# ------------------ AMP compatibility ------------------ #
def make_autocast(enabled, device_type):
    if not enabled:
        return nullcontext()
    try:
        from torch.amp import autocast as ac2
        return ac2(device_type=device_type)
    except Exception:
        from torch.cuda.amp import autocast as ac1
        return ac1(enabled=True)


# ------------------------------ CLI ------------------------------ #

def parse_args():
    p = argparse.ArgumentParser(description="Reconstruct NDVI from a trained STReF-NDVI model.")
    p.add_argument("--data-dir", default=DATA_DIR)
    p.add_argument("--model",    default=MODEL)
    p.add_argument("--out-dir",  default=OUT_DIR)
    p.add_argument("--s2-tif",   default=S2_TIF,
                   help="Optional reference GeoTIFF sharing the exact spatial "
                        "grid of the input arrays; used for CRS and transform.")
    p.add_argument("--channels", nargs="+", default=None,
                   help="Override combo, e.g. --channels vh vv_diff")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--patch",      type=int, default=3)
    p.add_argument("--tex-dim",    type=int, default=32)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--gpu", type=int, default=None)
    p.add_argument("--no-tif", action="store_true",
                   help="skip GeoTIFF output (only npy + PNG)")
    return p.parse_args()


# ------------------------------- Model ------------------------------- #

def build_batch_edge_index(B, T, device):
    src = torch.arange(0, T - 1, dtype=torch.long, device=device)
    tgt = src + 1
    base = torch.stack([torch.cat([src, tgt]), torch.cat([tgt, src])], dim=0)
    offsets = torch.arange(B, dtype=torch.long, device=device)\
                   .repeat_interleave(2 * (T - 1)) * T
    return base.repeat(1, B) + offsets.unsqueeze(0)


class HybridNDVIModel(nn.Module):
    """Architecture matching the STReF-NDVI training implementation."""

    def __init__(self, seq_ch, tex_ch, seq_len, tex_dim, patch):
        super().__init__()
        self.tex = nn.Sequential(
            nn.Conv2d(tex_ch, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, tex_dim, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool2d(1)
        )
        feat_dim = seq_ch * 2 + tex_dim
        self.conv1d = nn.Conv1d(feat_dim, 64, kernel_size=3, padding=1)
        self.gcn1 = GCNConv(64, 64)
        self.gcn2 = GCNConv(64, 64)
        self.lstm = nn.LSTM(64, 64, num_layers=3,
                            bidirectional=True, batch_first=True)
        enc = nn.TransformerEncoderLayer(
            d_model=128, nhead=4, dim_feedforward=256,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(enc, num_layers=4)
        self.out = nn.Linear(128, 1)
        self._edge_cache = {}

    def _get_edge_index(self, B, T, device):
        key = (B, T, str(device))
        if key not in self._edge_cache:
            self._edge_cache[key] = build_batch_edge_index(B, T, device)
        return self._edge_cache[key]

    def forward(self, seq, mean, tex, mask):
        B, T, Ctex, P, _ = tex.shape
        tf = self.tex(tex.reshape(B * T, Ctex, P, P)).view(B, T, -1)
        x = torch.cat([seq, mean, tf], dim=-1)
        x = self.conv1d(x.permute(0, 2, 1)).permute(0, 2, 1)

        edge_index = self._get_edge_index(B, T, x.device)
        flat = x.reshape(-1, 64)
        g1 = torch.relu(self.gcn1(flat, edge_index))
        g2 = self.gcn2(g1, edge_index)
        x = g2.reshape(B, T, 64)

        x, _ = self.lstm(x)
        x = self.transformer(x, src_key_padding_mask=~mask)
        return self.out(x).squeeze(-1)


def parse_channels_from_model_path(model_path):
    name = os.path.splitext(os.path.basename(model_path))[0]
    tokens = name.split("__")
    return [t for t in tokens if t in ALL_CHANNELS]


# ------------------------------ Main ------------------------------ #

def main():
    args = parse_args()

    if args.gpu is not None and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device = {device}")

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "compare"), exist_ok=True)

    # -------- load arrays --------
    S1 = np.load(os.path.join(args.data_dir, "S1.npy"))
    S2 = np.load(os.path.join(args.data_dir, "S2.npy"))
    target_mask = np.load(os.path.join(args.data_dir, "target_mask.npy")).astype(bool)
    S1_valid = np.load(os.path.join(args.data_dir, "S1_valid.npy")).astype(bool)
    good_pixel = np.load(os.path.join(args.data_dir, "good_pixel.npy")).astype(bool)

    with open(os.path.join(args.data_dir, "metadata.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)

    dates = meta["dates_s1"]
    ndvi_min = float(meta.get("ndvi_min", -0.2))
    ndvi_max = float(meta.get("ndvi_max", 1.0))
    ndvi_range = ndvi_max - ndvi_min

    T, C, H, W = S1.shape
    print(f"[INFO] Loaded: T={T}, C={C}, H={H}, W={W}")
    print(f"[INFO] NDVI range: [{ndvi_min}, {ndvi_max}]")

    # -------- channels --------
    if args.channels:
        for c in args.channels:
            if c not in ALL_CHANNELS:
                raise ValueError(f"unknown channel: {c}")
        channels = list(args.channels)
    else:
        channels = parse_channels_from_model_path(args.model)
        if not channels:
            raise ValueError(
                "cannot infer channels from model path; use --channels"
            )
    ch_ids = [ALL_CHANNELS.index(c) for c in channels]
    seq_ch = len(ch_ids) + 2
    tex_ch = len(ch_ids)
    print(f"[INFO] channels = {channels}  ch_ids = {ch_ids}")

    # -------- model --------
    model = HybridNDVIModel(seq_ch, tex_ch, T, args.tex_dim, args.patch).to(device)
    sd = torch.load(args.model, map_location=device)
    model.load_state_dict(sd)
    model.eval()
    print(f"[INFO] Model loaded: {os.path.basename(args.model)}")

    # -------- pad S1 --------
    pad = max(0, args.patch // 2)
    S1_pad = np.pad(S1, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="reflect")

    # -------- DOY --------
    doys = np.array(
        [datetime.strptime(str(d), "%Y%m%d").timetuple().tm_yday for d in dates],
        dtype=np.int32,
    )
    sin_doy = np.sin(2 * pi * doys / 365.0).astype(np.float32)
    cos_doy = np.cos(2 * pi * doys / 365.0).astype(np.float32)
    extras = np.stack([sin_doy, cos_doy], axis=1).astype(np.float32)

    # -------- pixels to predict --------
    coords = np.argwhere(good_pixel)
    print(f"[INFO] Predicting {len(coords):,} good pixels ...")

    pred_n = np.full((T, H, W), np.nan, dtype=np.float32)

    # -------- batched inference --------
    with torch.no_grad():
        for i in tqdm(range(0, len(coords), args.batch_size), desc="Predict"):
            batch = coords[i:i + args.batch_size]

            S_batch, M_batch, P_batch, MK_batch = [], [], [], []
            for r, c in batch:
                seq = S1[:, ch_ids, r, c].astype(np.float32)
                seq = np.concatenate([seq, extras], axis=1)

                pat = S1_pad[:, ch_ids, r:r + args.patch, c:c + args.patch].astype(np.float32)
                mean = pat.mean(axis=(2, 3)).astype(np.float32)
                mean = np.concatenate([mean, extras], axis=1)

                mask = S1_valid[:, r, c]

                S_batch.append(seq)
                M_batch.append(mean)
                P_batch.append(pat)
                MK_batch.append(mask)

            S_t = torch.from_numpy(np.stack(S_batch)).to(device, non_blocking=True)
            M_t = torch.from_numpy(np.stack(M_batch)).to(device, non_blocking=True)
            P_t = torch.from_numpy(np.stack(P_batch)).to(device, non_blocking=True)
            MK_t = torch.from_numpy(np.stack(MK_batch)).to(device, non_blocking=True)

            with make_autocast(args.amp, device.type):
                PR = model(S_t, M_t, P_t, MK_t).cpu().numpy()

            for j, (r, c) in enumerate(batch):
                pred_n[:, r, c] = PR[j]

    # -------- de-normalize --------
    pred_raw = pred_n * ndvi_range + ndvi_min

    # -------- true NDVI --------
    ndvi_raw = S2[..., 0].astype(np.float32)
    true_raw = np.where(target_mask, ndvi_raw, np.nan).astype(np.float32)

    # -------- save npy --------
    np.save(os.path.join(args.out_dir, "ndvi_pred.npy"), pred_raw.astype(np.float32))
    np.save(os.path.join(args.out_dir, "ndvi_true.npy"), true_raw.astype(np.float32))
    print(f"[INFO] Saved ndvi_pred.npy / ndvi_true.npy")

    # -------- global metrics --------
    from sklearn.metrics import r2_score
    mask_all = target_mask & np.isfinite(true_raw) & np.isfinite(pred_raw)
    if mask_all.any():
        yt = true_raw[mask_all]
        yp = pred_raw[mask_all]
        r2 = float(r2_score(yt, yp))
        rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
        mae = float(np.mean(np.abs(yt - yp)))
    else:
        r2, rmse, mae = -1.0, float("inf"), float("inf")

    print(f"[INFO] Reconstruction metrics (target_mask only):")
    print(f"        R2   = {r2:.4f}")
    print(f"        RMSE = {rmse:.4f}")
    print(f"        MAE  = {mae:.4f}")

    # -------- geo info --------
    geo_ok = False
    crs = None
    transform = None
    if args.s2_tif and os.path.exists(args.s2_tif) and not args.no_tif:
        try:
            import rasterio
            with rasterio.open(args.s2_tif) as src:
                crs = src.crs
                transform = src.transform
                height, width = src.height, src.width
            if (height, width) != (H, W):
                print(f"[WARN] s2-tif size ({height},{width}) != npy ({H},{W}); TIF output skipped")
            else:
                geo_ok = True
                print(f"[INFO] Geo info from {os.path.basename(args.s2_tif)}: CRS={crs}")
        except Exception as e:
            print(f"[WARN] cannot read s2-tif: {e}")

    # -------- per-date outputs --------
    print("\n[INFO] Writing per-date outputs ...")
    if np.isfinite(true_raw).any():
        vmin_plot = float(np.nanmin(true_raw))
        vmax_plot = float(np.nanmax(true_raw))
    else:
        vmin_plot, vmax_plot = ndvi_min, ndvi_max

    for t, d in enumerate(dates):
        pred_t = pred_raw[t]
        true_t = true_raw[t]

        if geo_ok:
            import rasterio
            out_tif = os.path.join(args.out_dir, f"NDVI_Predict_{d}.tif")
            meta_tif = {
                "driver": "GTiff",
                "height": H,
                "width": W,
                "count": 1,
                "dtype": "float32",
                "crs": crs,
                "transform": transform,
                "nodata": np.nan,
            }
            with rasterio.open(out_tif, "w", **meta_tif) as dst:
                dst.write(pred_t.astype(np.float32), 1)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

        axes[0].imshow(true_t, cmap="YlGn", vmin=vmin_plot, vmax=vmax_plot)
        n_true = int(np.isfinite(true_t).sum())
        axes[0].set_title(f"Observed NDVI ({n_true:,} px)")
        axes[0].axis("off")

        axes[1].imshow(pred_t, cmap="YlGn", vmin=vmin_plot, vmax=vmax_plot)
        axes[1].set_title("Predicted NDVI")
        axes[1].axis("off")

        cbar = fig.colorbar(axes[1].images[0], ax=axes, fraction=0.030, pad=0.02)
        cbar.set_label("NDVI")

        fig.suptitle(f"{d}", fontsize=13)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, "compare", f"compare_{d}.png"), dpi=150)
        plt.close(fig)

    # -------- per-frame metrics --------
    frame_metrics = []
    for t, d in enumerate(dates):
        m = target_mask[t] & np.isfinite(true_raw[t]) & np.isfinite(pred_raw[t])
        if m.any() and m.sum() > 1:
            yt = true_raw[t][m]
            yp = pred_raw[t][m]
            frame_metrics.append({
                "date": d,
                "n_pixels": int(m.sum()),
                "r2": float(r2_score(yt, yp)),
                "rmse": float(np.sqrt(np.mean((yt - yp) ** 2))),
                "mae": float(np.mean(np.abs(yt - yp))),
                "true_mean": float(yt.mean()),
                "pred_mean": float(yp.mean()),
            })
        else:
            frame_metrics.append({
                "date": d,
                "n_pixels": int(m.sum()),
                "r2": None, "rmse": None, "mae": None,
                "true_mean": None, "pred_mean": None,
            })

    import pandas as pd
    pd.DataFrame(frame_metrics).to_csv(
        os.path.join(args.out_dir, "per_frame_metrics.csv"),
        index=False, encoding="utf-8-sig",
    )

    summary = {
        "model": os.path.basename(args.model),
        "channels": channels,
        "T": int(T),
        "H": int(H),
        "W": int(W),
        "ndvi_min": ndvi_min,
        "ndvi_max": ndvi_max,
        "metrics_all": {"r2": r2, "rmse": rmse, "mae": mae,
                        "n_pixels": int(mask_all.sum())},
        "per_frame_csv": "per_frame_metrics.csv",
    }
    with open(os.path.join(args.out_dir, "reconstruction_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n[INFO] Done. Output: {args.out_dir}")


if __name__ == "__main__":
    main()
