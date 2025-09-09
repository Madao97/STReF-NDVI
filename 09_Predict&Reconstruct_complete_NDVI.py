#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
09_Predict&Reconstruct_complete_NDVI.py

Reconstruct (predict) full NDVI time series from trained ReNDVIval-RaST models.

Inputs (produced by 07_Dataset_Construction.py):
  fullyear_S1.npy : (T, 5, H, W)  channels = ["vh","rvi","vv_div","vv_diff","log_ratio"]
  fullyear_S2.npy : (T, H, W, 3)  [NDVI_norm, CloudMask, WaterMask]
  normalization_params.pkl : {"dates":[YYYYMMDD,...], "S2":{"ndvi":{"min":..,"max":..}}}
  geo_info.pkl : {"transform": tuple(...), "crs": "EPSG:XXXX" or WKT}

Outputs (in --out-dir):
  NDVI_Predict_<DATE>.tif
  compare/compare_<DATE>.png

Example:
  python 09_Predict&Reconstruct_complete_NDVI.py \
    --data-dir ./fullyear_2024_HS \
    --model ./fullyear_2024_HS/models/vh__vv_div__vv_diff.pth \
    --out-dir ./fullyear_2024_HS/predict_vh__vv_div__vv_diff \
    --aoi ./AOI/Hulu_Selangor.shp \
    --batch-size 512 --patch 3 --tex-dim 32 --amp
"""

import os
import re
import argparse
import pickle
import numpy as np
import torch
import torch.nn as nn

# AMP (PT2.x: torch.amp; PT<2.0: torch.cuda.amp)
try:
    from torch.amp import autocast
    _AMP_DEVICE_ARG = "device_type"
except Exception:
    from torch.cuda.amp import autocast
    _AMP_DEVICE_ARG = "enabled"  # unused; we guard with args.amp

from torch_geometric.nn import GCNConv
from tqdm import tqdm
import rasterio
from rasterio.crs import CRS
from affine import Affine
import geopandas as gpd
import matplotlib.pyplot as plt
from rasterio.features import geometry_mask


ALL_CHANNELS = ["vh","rvi","vv_div","vv_diff","log_ratio"]
EPS = 1e-6


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Predict full NDVI time series from a trained model.")
    p.add_argument("--data-dir", required=True, help="Directory containing fullyear_* and aux pickles.")
    p.add_argument("--model", required=True, help="Path to model .pth (e.g., vh__vv_div__vv_diff.pth).")
    p.add_argument("--out-dir", required=True, help="Output directory.")
    p.add_argument("--aoi", default=None, help="Optional AOI vector (SHP/GeoJSON/GPKG).")
    p.add_argument("--channels", nargs="+", default=None,
                   help="Override channel combo (subset of vh rvi vv_div vv_diff log_ratio).")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--patch", type=int, default=3)
    p.add_argument("--tex-dim", type=int, default=32)
    p.add_argument("--amp", action="store_true", help="Enable mixed precision for inference.")
    p.add_argument("--gpu", type=int, default=None, help="CUDA device index (default: auto).")
    return p.parse_args()


class HybridNDVIModel(nn.Module):
    """Tex-CNN → GCN (temporal chain) → Bi-LSTM → Transformer → Linear head."""
    def __init__(self, seq_ch: int, tex_ch: int, seq_len: int, tex_dim: int):
        super().__init__()
        self.tex = nn.Sequential(
            nn.Conv2d(tex_ch, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, tex_dim, 3, padding=1), nn.AdaptiveAvgPool2d(1)
        )
        feat_dim = seq_ch * 2 + tex_dim  # concat(seq, mean, tex)
        self.conv1d = nn.Conv1d(feat_dim, 64, 3, padding=1)
        self.gcn1 = GCNConv(64, 64)
        self.gcn2 = GCNConv(64, 64)
        self.lstm = nn.LSTM(64, 64, num_layers=3, bidirectional=True, batch_first=True)
        enc = nn.TransformerEncoderLayer(d_model=128, nhead=4, dim_feedforward=256,
                                         dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc, num_layers=4)
        self.out = nn.Linear(128, 1)

        # bidirectional chain edges for temporal graph (shared across batch)
        src = torch.arange(seq_len - 1, dtype=torch.long)
        tgt = src + 1
        edge = torch.cat([torch.stack([src, tgt]), torch.stack([tgt, src])], dim=1)
        self.register_buffer("edge_index", edge)

    def forward(self, seq, mean, tex, mask):
        # seq: (B,T,seq_ch); mean: (B,T,seq_ch); tex: (B,T,tex_ch,P,P); mask: (B,T)
        B, T, Ctex, P, _ = tex.shape
        tf = self.tex(tex.reshape(B * T, Ctex, P, P)).view(B, T, -1)   # (B,T,tex_dim)
        x = torch.cat([seq, mean, tf], dim=-1)                         # (B,T,feat_dim)
        x = self.conv1d(x.permute(0, 2, 1)).permute(0, 2, 1)           # (B,T,64)

        flat = x.reshape(-1, 64)
        g1 = torch.relu(self.gcn1(flat, self.edge_index))
        g2 = self.gcn2(g1, self.edge_index)
        x = g2.view(B, T, 64)

        x, _ = self.lstm(x)                                            # (B,T,128)
        x = self.transformer(x, src_key_padding_mask=~mask)            # pad True → ignored
        return self.out(x).squeeze(-1)                                  # (B,T)


def parse_channels_from_model_path(model_path: str) -> list[str]:
    name = os.path.splitext(os.path.basename(model_path))[0]
    # accept tokens separated by "__"
    tokens = name.split("__")
    # filter to known channel names
    chans = [t for t in tokens if t in ALL_CHANNELS]
    return chans


def main():
    args = parse_args()

    # Device
    if args.gpu is not None and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "compare"), exist_ok=True)

    # Load arrays and metadata
    S1 = np.load(os.path.join(args.data_dir, "fullyear_S1.npy"))   # (T,5,H,W)
    S2 = np.load(os.path.join(args.data_dir, "fullyear_S2.npy"))   # (T,H,W,3)
    with open(os.path.join(args.data_dir, "normalization_params.pkl"), "rb") as f:
        norm = pickle.load(f)
    with open(os.path.join(args.data_dir, "geo_info.pkl"), "rb") as f:
        geo = pickle.load(f)

    dates = norm["dates"]
    T, C, H, W = S1.shape
    ndvi_n = S2[..., 0].astype(np.float32)
    cloud  = S2[..., 1].astype(bool)
    water  = S2[..., 2].astype(bool)
    valid_all = (~cloud) & (~water)

    nd_min = float(norm["S2"]["ndvi"]["min"])
    nd_max = float(norm["S2"]["ndvi"]["max"])

    # Rebuild CRS/transform for rasterio
    crs = CRS.from_string(geo["crs"]) if not isinstance(geo["crs"], CRS) else geo["crs"]
    transform = Affine(*geo["transform"]) if not isinstance(geo["transform"], Affine) else geo["transform"]

    # DOY features
    doys = np.array([int(d[4:]) for d in dates], dtype=np.int32)
    sin_doy = np.sin(2 * np.pi * doys / 365).astype(np.float32)
    cos_doy = np.cos(2 * np.pi * doys / 365).astype(np.float32)

    # Channel selection
    if args.channels:
        for c in args.channels:
            if c not in ALL_CHANNELS:
                raise ValueError(f"Unknown channel: {c}")
        channels = list(args.channels)
    else:
        parsed = parse_channels_from_model_path(args.model)
        if not parsed:
            raise ValueError("Cannot infer channels from model name. Use --channels to specify.")
        channels = parsed
    ch_ids = [ALL_CHANNELS.index(c) for c in channels]
    seq_ch = len(ch_ids) + 2
    tex_ch = len(ch_ids)

    # Model
    model = HybridNDVIModel(seq_ch, tex_ch, T, args.tex_dim).to(device)
    sd = torch.load(args.model, map_location=device)
    model.load_state_dict(sd)
    model.eval()

    # Pad S1 once for texture patches
    pad = max(0, args.patch // 2)
    S1_pad = np.pad(S1, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="reflect")

    # Optional AOI mask in raster CRS
    roi_mask = np.ones((H, W), dtype=bool)  # True=inside/keep
    if args.aoi:
        gdf = gpd.read_file(args.aoi)
        if gdf.empty:
            raise ValueError("AOI has no features.")
        try:
            gdf = gdf.to_crs(crs)
        except Exception:
            # if geo["crs"] is string EPSG
            gdf = gdf.set_crs(gdf.crs or crs).to_crs(crs)
        geoms = [geom.__geo_interface__ for geom in gdf.geometry]
        roi_mask = ~geometry_mask(geoms, invert=True, transform=transform, out_shape=(H, W))  # True inside

        # We want True inside AOI; geometry_mask with invert=True returns mask for features,
        # but we inverted again above to ensure roi_mask True=inside.
        roi_mask = ~roi_mask

    # Build list of pixel coords that have at least one valid time and are inside AOI
    coords = [(r, c) for r in range(H) for c in range(W) if roi_mask[r, c] and valid_all[:, r, c].any()]
    pred_ndvi = np.full((T, H, W), np.nan, dtype=np.float32)

    # Batched inference
    with torch.no_grad():
        for i in tqdm(range(0, len(coords), args.batch_size), desc="Predict"):
            batch = coords[i:i + args.batch_size]
            S_batch, M_batch, P_batch, MK_batch = [], [], [], []

            extras = np.stack([sin_doy, cos_doy], axis=1)  # (T,2)
            for r, c in batch:
                seq = S1[:, ch_ids, r, c].astype(np.float32)                  # (T, Csel)
                seq = np.concatenate([seq, extras], axis=1)                   # (T, Csel+2)

                pat = S1_pad[:, ch_ids, r:r + args.patch, c:c + args.patch].astype(np.float32)  # (T,Csel,P,P)
                mean = pat.mean(axis=(2, 3)).astype(np.float32)               # (T, Csel)
                mean = np.concatenate([mean, extras], axis=1)                 # (T, Csel+2)

                mask = valid_all[:, r, c]                                     # (T,)

                S_batch.append(seq)
                M_batch.append(mean)
                P_batch.append(pat)
                MK_batch.append(mask)

            S_t  = torch.from_numpy(np.stack(S_batch)).to(device)
            M_t  = torch.from_numpy(np.stack(M_batch)).to(device)
            P_t  = torch.from_numpy(np.stack(P_batch)).to(device)
            MK_t = torch.from_numpy(np.stack(MK_batch)).to(device)

            autocast_kwargs = {_AMP_DEVICE_ARG: device.type} if args.amp else {}
            with autocast(**autocast_kwargs) if args.amp else torch.no_grad():
                PR = model(S_t, M_t, P_t, MK_t).cpu().numpy()  # (B,T)

            for j, (r, c) in enumerate(batch):
                pred_ndvi[:, r, c] = PR[j]

    # Write per-date GeoTIFF + comparison PNG
    for t, d in enumerate(dates):
        pred = pred_ndvi[t] * (nd_max - nd_min) + nd_min
        true = ndvi_n[t]   * (nd_max - nd_min) + nd_min

        # outside AOI set to NaN
        pred_vis = pred.copy()
        pred_vis[~roi_mask] = np.nan

        # true-visible only where clear & inside AOI
        true_vis = true.copy()
        true_vis[~(roi_mask & ~cloud[t] & ~water[t])] = np.nan

        # GeoTIFF
        out_tif = os.path.join(args.out_dir, f"NDVI_Predict_{d}.tif")
        meta = {
            "driver": "GTiff",
            "height": H,
            "width": W,
            "count": 1,
            "dtype": "float32",
            "crs": crs,
            "transform": transform,
            "nodata": np.nan,
        }
        with rasterio.open(out_tif, "w", **meta) as dst:
            dst.write(pred_vis.astype(np.float32), 1)

        # Compare PNG
        fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
        im0 = axes[0].imshow(true_vis, cmap="YlGn", vmin=nd_min, vmax=nd_max)
        axes[0].set_title("True NDVI"); axes[0].axis("off")
        im1 = axes[1].imshow(pred_vis, cmap="YlGn", vmin=nd_min, vmax=nd_max)
        axes[1].set_title("Pred NDVI"); axes[1].axis("off")

        cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        fig.colorbar(im1, cax=cax, label="NDVI")

        fig.suptitle(d, fontsize=14)
        fig.subplots_adjust(left=0.05, right=0.90, top=0.88, bottom=0.05)
        fig.savefig(os.path.join(args.out_dir, "compare", f"compare_{d}.png"), dpi=300)
        plt.close(fig)

    print(f"[INFO] NDVI reconstruction complete. Saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
