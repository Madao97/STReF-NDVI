# -*- coding: utf-8 -*-
"""
Train STReF-NDVI using Sentinel-1 feature sequences and Sentinel-2 NDVI labels.

Model components:
    Spatial texture CNN, temporal convolution, temporal-chain GCN,
    bidirectional LSTM, Transformer encoder, and linear prediction head.

Training:
    Weighted combination of MSE and MAE:
        loss = 0.7 * MSE + 0.3 * MAE

    With the default cloud weight of zero, only valid optical labels
    contribute to the loss.

Validation:
    Training and validation use disjoint center-pixel sets.
    Each center pixel's complete time series belongs to one set.
    This split does not enforce spatial separation between image patches.

Feature selection:
    By default, all non-empty subsets of the five SAR features are trained.
    Completed combinations are skipped based on checkpoints or summary.csv.
    Skipping completed combinations does not resume an interrupted optimizer.

Required inputs:
    S1.npy
    S2.npy
    target_mask.npy
    good_pixel.npy
    metadata.json
"""

import os
import csv
import json
import argparse
import random
from contextlib import nullcontext
from datetime import datetime
from itertools import combinations, chain
from math import pi

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import r2_score
from tqdm import tqdm

try:
    from torch_geometric.nn import GCNConv
except ImportError as e:
    raise ImportError(
        "torch_geometric is required. Install with:\n"
        "    pip install torch_geometric\n"
    ) from e


# =============================================================
# EDIT THESE IF YOU WANT TO DOUBLE-CLICK AND RUN
# =============================================================
# Default paths; override with --data-dir and --out-dir.
DATA_DIR = os.path.join("data", "Dataset")
OUT_DIR = os.path.join("data", "Dataset")


# ------------------ AMP compatibility ------------------ #
def make_scaler(enabled, device_type):
    try:
        from torch.amp import GradScaler as GS2
        return GS2(device_type=device_type, enabled=enabled)
    except Exception:
        from torch.cuda.amp import GradScaler as GS1
        return GS1(enabled=enabled)


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

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train STReF-NDVI (S1 -> NDVI).")
    p.add_argument("--data-dir", default=DATA_DIR)
    p.add_argument("--out-dir",  default=OUT_DIR)
    p.add_argument("--channels", nargs="+",
                   default=["vh", "rvi", "vv_div", "vv_diff", "log_ratio"])

    # combo selection
    p.add_argument("--single-combo", action="store_true",
                   help="only train the FULL --channels list (no subset search)")
    p.add_argument("--only-combos", nargs="+", default=None,
                   help="explicit combos, e.g. vh,rvi vv_div,vv_diff,log_ratio")
    p.add_argument("--no-skip-trained", action="store_true",
                   help="do NOT skip combos whose .pth already exists")

    # training
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--early-stop", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--clear-wt", type=float, default=1.0)
    p.add_argument("--cloud-wt", type=float, default=0.0)
    p.add_argument("--samples-per-bin", type=int, default=250_000)
    p.add_argument("--val-pixel-ratio", type=float, default=0.20)

    # model
    p.add_argument("--patch", type=int, default=3)
    p.add_argument("--tex-dim", type=int, default=32)

    # misc
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--memmap", action="store_true")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--gpu", type=int, default=None)
    return p.parse_args()


# ------------------------ Constants ------------------------ #

ALL_CHANNELS = ["vh", "rvi", "vv_div", "vv_diff", "log_ratio"]
BIN_EDGES = np.linspace(0.0, 1.0, 11)   # 10 bins in normalized NDVI space
EPS = 1e-6


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ------------------------------- Dataset ------------------------------- #

class PixelDataset(Dataset):
    """One sample = one full time series for a single pixel."""

    def __init__(self, triplets, ch_ids, S1, S1_pad, ndvi_n, target_mask,
                 sin_doy, cos_doy, patch, clear_wt=1.0, cloud_wt=0.05):
        self.triplets = triplets
        self.ch_ids = ch_ids
        self.S1 = S1
        self.S1_pad = S1_pad
        self.ndvi_n = ndvi_n
        self.target_mask = target_mask
        self.sin_doy = sin_doy
        self.cos_doy = cos_doy
        self.patch = patch
        self.clear_wt = float(clear_wt)
        self.cloud_wt = float(cloud_wt)

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        t, r, c = self.triplets[idx]

        seq = self.S1[:, self.ch_ids, r, c].astype(np.float32)
        seq = np.concatenate(
            [seq, self.sin_doy[:, None], self.cos_doy[:, None]], axis=1
        )

        P = self.patch
        pat = self.S1_pad[:, self.ch_ids, r:r + P, c:c + P].astype(np.float32)
        mean = pat.mean(axis=(2, 3)).astype(np.float32)
        mean = np.concatenate(
            [mean, self.sin_doy[:, None], self.cos_doy[:, None]], axis=1
        )

        y = self.ndvi_n[:, r, c].astype(np.float32)
        m = self.target_mask[:, r, c].astype(bool)

        wts = np.where(m, self.clear_wt, self.cloud_wt).astype(np.float32)
        nan_idx = np.isnan(y)
        if nan_idx.any():
            wts[nan_idx] = 0.0
            y[nan_idx] = 0.0

        return seq, mean, pat, y, m, wts


# ------------------------------- Model ------------------------------- #

def build_batch_edge_index(B, T, device):
    src = torch.arange(0, T - 1, dtype=torch.long, device=device)
    tgt = src + 1
    base = torch.stack([torch.cat([src, tgt]), torch.cat([tgt, src])], dim=0)
    offsets = torch.arange(B, dtype=torch.long, device=device)\
                   .repeat_interleave(2 * (T - 1)) * T
    return base.repeat(1, B) + offsets.unsqueeze(0)


class HybridNDVIModel(nn.Module):
    """Tex-CNN -> GCN (temporal chain) -> Bi-LSTM -> Transformer -> linear."""

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

        enc_layer = nn.TransformerEncoderLayer(
            d_model=128, nhead=4, dim_feedforward=256,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=4)
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


# --------------------------- Sampling --------------------------- #

def reservoir_bin_sampling(ndvi_n, target_mask, pixel_mask,
                           samples_per_bin, bin_edges=BIN_EDGES):
    """
    Balanced sampling over NDVI bins, restricted to a given pixel_mask.
    Returns list of (t, r, c).
    """
    T, H, W = ndvi_n.shape
    n_bins = len(bin_edges) - 1

    ts_all, rs_all, cs_all, bin_all = [], [], [], []

    for t in range(T):
        valid = target_mask[t] & pixel_mask
        if not valid.any():
            continue
        rr, cc = np.where(valid)
        vals = ndvi_n[t][rr, cc]
        ok = np.isfinite(vals)
        rr = rr[ok]; cc = cc[ok]; vals = vals[ok]
        if rr.size == 0:
            continue
        bid = np.clip(np.digitize(vals, bin_edges) - 1, 0, n_bins - 1)

        ts_all.append(np.full(rr.size, t, dtype=np.int16))
        rs_all.append(rr.astype(np.int32))
        cs_all.append(cc.astype(np.int32))
        bin_all.append(bid.astype(np.int8))

    if len(ts_all) == 0:
        return []

    ts_all = np.concatenate(ts_all)
    rs_all = np.concatenate(rs_all)
    cs_all = np.concatenate(cs_all)
    bin_all = np.concatenate(bin_all)

    coords = []
    for b in range(n_bins):
        idx = np.where(bin_all == b)[0]
        n = idx.size
        if n == 0:
            continue
        if n <= samples_per_bin:
            chosen = idx
        else:
            chosen = np.random.choice(idx, samples_per_bin, replace=False)
        for i in chosen:
            coords.append((int(ts_all[i]), int(rs_all[i]), int(cs_all[i])))

    random.shuffle(coords)
    return coords


def split_pixels(good_pixel, val_pixel_ratio, seed):
    """
    Pixel-wise split of the good_pixel mask.
    Returns (train_pixel_mask, val_pixel_mask) as (H, W) bool arrays.
    """
    H, W = good_pixel.shape
    good_flat = np.where(good_pixel.flatten())[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(good_flat))

    n_val = int(val_pixel_ratio * len(good_flat))
    val_idx = good_flat[perm[:n_val]]
    train_idx = good_flat[perm[n_val:]]

    train_mask = np.zeros(H * W, dtype=bool)
    val_mask = np.zeros(H * W, dtype=bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True

    return train_mask.reshape(H, W), val_mask.reshape(H, W)


# ---------------------------- Training ---------------------------- #

def train_one_combo(combo, env, args):
    name = "__".join(combo)
    ch_ids = [env["channel_index"][c] for c in combo]
    seq_ch = len(ch_ids) + 2
    tex_ch = len(ch_ids)

    ds_tr = PixelDataset(env["train_coords"], ch_ids,
                         clear_wt=args.clear_wt, cloud_wt=args.cloud_wt,
                         **env["ds_kwargs"])
    ds_va = PixelDataset(env["val_coords"], ch_ids,
                         clear_wt=args.clear_wt, cloud_wt=args.cloud_wt,
                         **env["ds_kwargs"])

    pin = (env["device"].type == "cuda")
    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True,
                       num_workers=args.num_workers, pin_memory=pin)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False,
                       num_workers=args.num_workers, pin_memory=pin)

    model = HybridNDVIModel(seq_ch, tex_ch, env["T"],
                            args.tex_dim, args.patch).to(env["device"])
    optimizer = optim.AdamW(model.parameters(),
                            lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=3, T_mult=2
    )
    mse = nn.MSELoss(reduction="none")
    mae = nn.L1Loss(reduction="none")
    scaler = make_scaler(args.amp, env["device"].type)

    best_r2 = -1.0
    bad_epochs = 0
    best_path = None
    hist_fp = os.path.join(env["out_dir"], f"{name}_history.csv")
    ndvi_range = env["ndvi_max"] - env["ndvi_min"]

    with open(hist_fp, "w", newline="", encoding="utf-8") as hf:
        writer = csv.writer(hf)
        writer.writerow(["epoch", "r2", "rmse", "train_loss"])

        for ep in range(1, args.epochs + 1):

            # ---------------- Train ----------------
            model.train()
            total_loss, n_batches = 0.0, 0
            pbar = tqdm(dl_tr, desc=f"{name} Ep{ep}", leave=False)
            for seq, mean, tex, y, mask, wts in pbar:
                seq = seq.to(env["device"], non_blocking=True)
                mean = mean.to(env["device"], non_blocking=True)
                tex = tex.to(env["device"], non_blocking=True)
                y = y.to(env["device"], non_blocking=True)
                mask = mask.to(env["device"], non_blocking=True)
                wts = wts.to(env["device"], non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                with make_autocast(args.amp, env["device"].type):
                    pr = model(seq, mean, tex, mask)
                    loss_mse = (mse(pr, y) * wts).sum() / torch.clamp(wts.sum(), min=1e-6)
                    loss_mae = (mae(pr, y) * wts).sum() / torch.clamp(wts.sum(), min=1e-6)
                    loss = 0.7 * loss_mse + 0.3 * loss_mae

                if args.amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                total_loss += loss.item()
                n_batches += 1
                pbar.set_postfix(loss=f"{loss.item():.4f}")

            scheduler.step()
            avg_loss = total_loss / max(n_batches, 1)

            # ---------------- Validate ----------------
            model.eval()
            all_p, all_t = [], []
            with torch.no_grad():
                for seq, mean, tex, y, mask, _ in dl_va:
                    seq = seq.to(env["device"], non_blocking=True)
                    mean = mean.to(env["device"], non_blocking=True)
                    tex = tex.to(env["device"], non_blocking=True)
                    y = y.to(env["device"], non_blocking=True)
                    mask = mask.to(env["device"], non_blocking=True)

                    pr = model(seq, mean, tex, mask).cpu().numpy()
                    yt = y.cpu().numpy()
                    mk = mask.cpu().numpy()

                    for b in range(pr.shape[0]):
                        if mk[b].any():
                            all_p.append(pr[b, mk[b]])
                            all_t.append(yt[b, mk[b]])

            if not all_p:
                r2, rmse = -1.0, float("inf")
            else:
                yp_n = np.concatenate(all_p)
                yt_n = np.concatenate(all_t)
                yp = yp_n * ndvi_range + env["ndvi_min"]
                yt = yt_n * ndvi_range + env["ndvi_min"]
                r2 = float(r2_score(yt, yp))
                rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))

            writer.writerow([ep, f"{r2:.4f}", f"{rmse:.4f}", f"{avg_loss:.6f}"])
            hf.flush()
            print(f"[{name}] Ep{ep:02d}: R2={r2:.4f}  RMSE={rmse:.4f}  loss={avg_loss:.4f}")

            if r2 > best_r2:
                best_r2 = r2
                bad_epochs = 0
                best_path = os.path.join(env["out_dir"], f"{name}.pth")
                torch.save(model.state_dict(), best_path)
            else:
                bad_epochs += 1
                if bad_epochs >= args.early_stop:
                    print(f"[{name}] early stop at epoch {ep}")
                    break

    return best_path, best_r2, name


# ------------------------------ Main ------------------------------ #

def main():
    args = parse_args()
    set_seed(args.seed)

    if args.gpu is not None and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device = {device}")

    os.makedirs(args.out_dir, exist_ok=True)
    models_dir = os.path.join(args.out_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    # -------- load arrays --------
    mmap_mode = "r" if args.memmap else None
    S1 = np.load(os.path.join(args.data_dir, "S1.npy"), mmap_mode=mmap_mode)
    S2 = np.load(os.path.join(args.data_dir, "S2.npy"), mmap_mode=mmap_mode)
    target_mask = np.load(os.path.join(args.data_dir, "target_mask.npy")).astype(bool)
    good_pixel = np.load(os.path.join(args.data_dir, "good_pixel.npy")).astype(bool)

    with open(os.path.join(args.data_dir, "metadata.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)

    dates = meta["dates_s1"]
    ndvi_min = float(meta.get("ndvi_min", -0.2))
    ndvi_max = float(meta.get("ndvi_max", 1.0))

    T, C, H, W = S1.shape
    print(f"[INFO] Loaded: T={T}, C={C}, H={H}, W={W}")

    # -------- NDVI normalization --------
    ndvi_raw = np.array(S2[..., 0], dtype=np.float32)
    ndvi_n = (ndvi_raw - ndvi_min) / (ndvi_max - ndvi_min + EPS)
    ndvi_n = np.where(np.isfinite(ndvi_raw), ndvi_n, np.nan).astype(np.float32)

    print(f"[INFO] NDVI range (raw)   : [{ndvi_min:.3f}, {ndvi_max:.3f}]")
    print(f"[INFO] target_mask pixels : {int(target_mask.sum()):,}")
    print(f"[INFO] good_pixel         : {int(good_pixel.sum()):,} / {H*W:,}")

    # -------- PIXEL-WISE SPLIT --------
    train_pixel_mask, val_pixel_mask = split_pixels(
        good_pixel, args.val_pixel_ratio, args.seed
    )
    print(f"[INFO] Pixel-wise split  : train={int(train_pixel_mask.sum()):,}  "
          f"val={int(val_pixel_mask.sum()):,}")

    # -------- DOY --------
    doys = np.array(
        [datetime.strptime(str(d), "%Y%m%d").timetuple().tm_yday for d in dates],
        dtype=np.int32,
    )
    sin_doy = np.sin(2 * pi * doys / 365.0).astype(np.float32)
    cos_doy = np.cos(2 * pi * doys / 365.0).astype(np.float32)

    # -------- pad S1 --------
    pad = max(0, args.patch // 2)
    S1_pad = np.pad(S1, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="reflect")

    # -------- sample triplets on each side --------
    print(f"[INFO] Sampling (bin-balanced, samples_per_bin={args.samples_per_bin:,}) ...")
    train_coords = reservoir_bin_sampling(
        ndvi_n, target_mask, train_pixel_mask, args.samples_per_bin
    )
    val_coords = reservoir_bin_sampling(
        ndvi_n, target_mask, val_pixel_mask, args.samples_per_bin
    )
    print(f"[INFO] train triplets: {len(train_coords):,}")
    print(f"[INFO] val triplets  : {len(val_coords):,}")
    if len(train_coords) < 10 or len(val_coords) < 10:
        raise RuntimeError("Too few samples; check target_mask / good_pixel.")

    # -------- combos --------
    channel_index = {c: ALL_CHANNELS.index(c) for c in ALL_CHANNELS}
    if args.only_combos:
        combos = [tok.split(",") for tok in args.only_combos]
    elif args.single_combo:
        combos = [list(args.channels)]
    else:
        base = list(args.channels)
        combos = [list(c) for c in chain.from_iterable(
            combinations(base, r) for r in range(1, len(base) + 1)
        )]
        # sort by subset size, then alphabetically — small combos first
        combos.sort(key=lambda x: (len(x), x))

    print(f"[INFO] Number of channel combos: {len(combos)}")
    for i, c in enumerate(combos, 1):
        print(f"       {i:2d}. {'__'.join(c)}")

    env = dict(
        device=device,
        out_dir=models_dir,
        T=T,
        channel_index=channel_index,
        train_coords=train_coords,
        val_coords=val_coords,
        ndvi_min=ndvi_min,
        ndvi_max=ndvi_max,
        ds_kwargs=dict(
            S1=S1, S1_pad=S1_pad,
            ndvi_n=ndvi_n, target_mask=target_mask,
            sin_doy=sin_doy, cos_doy=cos_doy,
            patch=args.patch,
        ),
    )

    # -------- resume support --------
    summary_fp = os.path.join(models_dir, "summary.csv")
    summary_exists = os.path.exists(summary_fp)
    done_combos = set()
    if summary_exists:
        try:
            with open(summary_fp, encoding="utf-8") as f:
                reader = csv.reader(f)
                next(reader, None)   # skip header
                for row in reader:
                    if len(row) >= 3:
                        done_combos.add(row[2])
        except Exception:
            pass
        print(f"[INFO] Resume: {len(done_combos)} combos already in summary.csv")

    skip_trained = not args.no_skip_trained

    mode = "a" if summary_exists else "w"
    with open(summary_fp, mode, newline="", encoding="utf-8") as sf:
        writer = csv.writer(sf)
        if not summary_exists:
            writer.writerow(["model_path", "best_r2", "combo"])

        for i, combo in enumerate(combos, 1):
            name = "__".join(combo)
            ckpt = os.path.join(models_dir, f"{name}.pth")

            if skip_trained and (name in done_combos or os.path.exists(ckpt)):
                print(f"[SKIP {i}/{len(combos)}] {name} (already trained)")
                continue

            print(f"\n==== [{i}/{len(combos)}] Training combo: {combo} ====")
            best_path, r2, name = train_one_combo(combo, env, args)
            writer.writerow([best_path or "", f"{r2:.4f}", name])
            sf.flush()

            if env["device"].type == "cuda":
                torch.cuda.empty_cache()

    print(f"\n[INFO] Training complete. Models under: {models_dir}")


if __name__ == "__main__":
    main()
