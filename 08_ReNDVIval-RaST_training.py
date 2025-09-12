#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
08_ST-ClearNDVI_training.py

Train the ReNDVIval-RaST model to reconstruct NDVI from Sentinel-1 feature stacks.
Architecture: Tex-CNN (local texture) + GCN (temporal edges) + Bi-LSTM + Transformer.
Mask-aware loss: clear pixels weighted higher than cloud/water pixels.

Inputs (produced by 07_Dataset_Construction.py):
  fullyear_S1.npy : (T, 5, H, W)  channels = ["vh","rvi","vv_div","vv_diff","log_ratio"] in linear domain
  fullyear_S2.npy : (T, H, W, 3)  [NDVI_norm, CloudMask, WaterMask]
  normalization_params.pkl (dates list)
  good_pixel.npy  : (H, W) boolean mask of spatially “good” pixels

Outputs:
  models/<combo>.pth          Best checkpoint per channel combo
  models/<combo>_history.csv  Per-epoch R2/RMSE
  models/summary.csv          Overview of best scores per combo
"""

import os
import re
import csv
import math
import argparse
import random
import pickle
from itertools import combinations, chain
from math import pi

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# AMP (PyTorch 2.x: torch.amp, older: torch.cuda.amp)
try:
    from torch.amp import autocast, GradScaler
    _AMP_DEVICE_ARG = "device_type"
except Exception:  # fallback for PT<2.0
    from torch.cuda.amp import autocast, GradScaler
    _AMP_DEVICE_ARG = "enabled"  # will be ignored below

from torch.utils.data import Dataset, DataLoader
from torch_geometric.nn import GCNConv
from sklearn.metrics import r2_score
from tqdm import tqdm


# ------------------------------ CLI ------------------------------ #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train ReNDVIval-RaST (S1→NDVI) with mask-aware loss.")
    p.add_argument("--data-dir", required=True, help="Directory with fullyear_* and aux files.")
    p.add_argument("--out-dir", required=True, help="Directory to save models and logs.")
    p.add_argument("--channels", nargs="+", default=["vh","rvi","vv_div","vv_diff","log_ratio"],
                   help="Channel names subset to consider. Default: all 5.")
    p.add_argument("--search-all-combos", action="store_true",
                   help="Train all non-empty combos (1..len(channels)).")
    p.add_argument("--only-combos", nargs="+", default=None,
                   help="Explicit combos like vh,rvi vv_div,vv_diff,log_ratio (space-separated, comma-joined).")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--early-stop", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--clear-wt", type=float, default=1.0)
    p.add_argument("--cloud-wt", type=float, default=0.05)
    p.add_argument("--samples-per-bin", type=int, default=250_000,
                   help="Max samples per NDVI bin for reservoir sampling.")
    p.add_argument("--patch", type=int, default=3, help="Square patch size for texture branch.")
    p.add_argument("--tex-dim", type=int, default=32, help="Output channels of texture CNN.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--memmap", action="store_true",
                   help="Load .npy with mmap_mode='r' to save RAM.")
    p.add_argument("--amp", action="store_true", help="Enable mixed precision.")
    p.add_argument("--gpu", type=int, default=None, help="CUDA device index (default: auto).")
    return p.parse_args()


# ------------------------ Utilities / Setup ---------------------- #

ALL_CHANNELS = ["vh","rvi","vv_div","vv_diff","log_ratio"]
BIN_EDGES = np.linspace(0.0, 1.0, 11)  # 10 bins
EPS = 1e-6

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ------------------------------- Data ---------------------------- #

class PixelDataset(Dataset):
    """Return time sequences for a single spatial pixel with DOY features, texture patch, and weights."""
    def __init__(self, triplets, ch_ids, S1, S1_pad, ndvi_n, cloud, water, sin_doy, cos_doy, patch):
        self.triplets = triplets
        self.ch_ids   = ch_ids
        self.S1       = S1
        self.S1_pad   = S1_pad
        self.ndvi_n   = ndvi_n
        self.cloud    = cloud
        self.water    = water
        self.sin_doy  = sin_doy
        self.cos_doy  = cos_doy
        self.patch    = patch
        self.pad      = patch // 2

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        t, r, c = self.triplets[idx]
        # Temporal S1 at the pixel (T, C)
        seq = self.S1[:, self.ch_ids, r, c].astype(np.float32)
        seq = np.concatenate([seq, self.sin_doy[:, None], self.cos_doy[:, None]], axis=1)  # (T, C+2)

        # Texture patch for each time step: (T, C, P, P) + mean branch (T, C+2)
        P = self.patch
        pat = self.S1_pad[:, self.ch_ids, r:r+P, c:c+P].astype(np.float32)     # (T,C,P,P)
        mean = pat.mean(axis=(2, 3)).astype(np.float32)                         # (T,C)
        mean = np.concatenate([mean, self.sin_doy[:, None], self.cos_doy[:, None]], axis=1)

        # Targets and weights
        y = self.ndvi_n[:, r, c].astype(np.float32)                             # (T,)
        m = (~self.cloud[:, r, c]) & (~self.water[:, r, c])                     # (T,)
        wts = np.where(m, 1.0, 0.05).astype(np.float32)                         # default weights; scaled later

        nan_idx = np.isnan(y)
        if nan_idx.any():
            wts[nan_idx] = 0.0
            y[nan_idx] = 0.0
        return seq, mean, pat, y, m.astype(np.bool_), wts


# ----------------------------- Model ----------------------------- #

class HybridNDVIModel(nn.Module):
    """
    Tex-CNN → GCN (temporal chain graph) → Bi-LSTM → Transformer → linear head
    """
    def __init__(self, seq_ch: int, tex_ch: int, seq_len: int, tex_dim: int, patch: int):
        super().__init__()
        self.tex = nn.Sequential(
            nn.Conv2d(tex_ch, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, tex_dim, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool2d(1)
        )
        feat_dim = seq_ch * 2 + tex_dim  # concat(seq, mean, tex)
        self.conv1d = nn.Conv1d(feat_dim, 64, kernel_size=3, padding=1)

        self.gcn1 = GCNConv(64, 64)
        self.gcn2 = GCNConv(64, 64)

        self.lstm = nn.LSTM(64, 64, num_layers=3, bidirectional=True, batch_first=True)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=128, nhead=4, dim_feedforward=256, dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=4)
        self.out = nn.Linear(128, 1)

        # chain edges for temporal graph (shared across batch)
        src = torch.arange(seq_len - 1, dtype=torch.long)
        tgt = src + 1
        edge = torch.cat([torch.stack([src, tgt]), torch.stack([tgt, src])], dim=1)  # bidirectional chain
        self.register_buffer("edge_index", edge)

    def forward(self, seq, mean, tex, mask):
        # seq:  (B,T,seq_ch) ; mean: (B,T,seq_ch) ; tex: (B,T,tex_ch,P,P)
        B, T, Ctex, P, _ = tex.shape
        tf = self.tex(tex.reshape(B * T, Ctex, P, P)).view(B, T, -1)  # (B,T,tex_dim)
        x = torch.cat([seq, mean, tf], dim=-1)                        # (B,T,feat_dim)
        x = self.conv1d(x.permute(0, 2, 1)).permute(0, 2, 1)          # (B,T,64)

        flat = x.reshape(-1, 64)
        g1 = torch.relu(self.gcn1(flat, self.edge_index))
        g2 = self.gcn2(g1, self.edge_index)
        x = g2.reshape(B, T, 64)

        x, _ = self.lstm(x)                                           # (B,T,128)
        x = self.transformer(x, src_key_padding_mask=~mask)           # pad positions True → ignored
        return self.out(x).squeeze(-1)                                 # (B,T)


# ---------------------------- Training --------------------------- #

def reservoir_bin_sampling(S1, S2, good_pixel, samples_per_bin: int, bin_edges=BIN_EDGES):
    """Balanced pixel sampling over NDVI bins with reservoir sampling per bin."""
    T, _, H, W = S1.shape
    ndvi_n = S2[..., 0]
    cloud  = S2[..., 1].astype(bool)
    water  = S2[..., 2].astype(bool)

    bins = {i: [] for i in range(len(bin_edges) - 1)}
    counts = {i: 0 for i in range(len(bin_edges) - 1)}

    for t in range(T):
        valid = (~cloud[t]) & (~water[t]) & good_pixel
        if not valid.any():
            continue
        rows, cols = np.where(valid)
        seg = np.digitize(ndvi_n[t][valid], bin_edges) - 1
        for k, b in enumerate(seg):
            counts[b] += 1
            sample = (t, int(rows[k]), int(cols[k]))
            if len(bins[b]) < samples_per_bin:
                bins[b].append(sample)
            else:
                j = random.randint(0, counts[b] - 1)
                if j < samples_per_bin:
                    bins[b][j] = sample

    coords = []
    for b in range(len(bin_edges) - 1):
        coords.extend(bins[b])
    random.shuffle(coords)
    return coords


def train_one_combo(combo, env, args):
    name = "__".join(combo)
    ch_ids = [env["channel_index"][c] for c in combo]
    seq_ch = len(ch_ids) + 2
    tex_ch = len(ch_ids)

    n_val = int(0.20 * len(env["coords"]))
    ds_tr = PixelDataset(env["coords"][n_val:], ch_ids, **env["ds_kwargs"])
    ds_va = PixelDataset(env["coords"][:n_val],  ch_ids, **env["ds_kwargs"])

    pin = (env["device"].type == "cuda")
    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True,
                       num_workers=args.num_workers, pin_memory=pin, persistent_workers=False)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False,
                       num_workers=args.num_workers, pin_memory=pin, persistent_workers=False)

    model = HybridNDVIModel(seq_ch, tex_ch, env["T"], args.tex_dim, args.patch).to(env["device"])
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=3, T_mult=2)
    mse = nn.MSELoss(reduction="none")
    mae = nn.L1Loss(reduction="none")
    scaler = GradScaler(enabled=args.amp)

    best_r2, bad_epochs, best_path = -1.0, 0, None
    hist_fp = os.path.join(env["out_dir"], f"{name}_history.csv")
    with open(hist_fp, "w", newline="") as hf:
        writer = csv.writer(hf)
        writer.writerow(["epoch", "r2", "rmse"])

        for ep in range(1, args.epochs + 1):
            # ---- Train ----
            model.train()
            pbar = tqdm(dl_tr, desc=f"{name} Ep{ep}")
            for seq, mean, tex, y, mask, wts in pbar:
                seq  = seq.to(env["device"], non_blocking=True)
                mean = mean.to(env["device"], non_blocking=True)
                tex  = tex.to(env["device"], non_blocking=True)
                y    = y.to(env["device"], non_blocking=True)
                mask = mask.to(env["device"], non_blocking=True)
                wts  = (wts * args.clear_wt).where(mask, wts * (args.cloud_wt / max(args.clear_wt, EPS)))

                autocast_kwargs = {_AMP_DEVICE_ARG: env["amp_device_arg_val"]} if args.amp else {}
                with autocast(**autocast_kwargs) if args.amp else torch.no_grad() if False else torch.enable_grad():
                    pr = model(seq, mean, tex, mask)
                    loss_mse = (mse(pr, y) * wts).sum() / torch.clamp(wts.sum(), min=1e-6)
                    loss_mae = (mae(pr, y) * wts).sum() / torch.clamp(wts.sum(), min=1e-6)
                    loss = 0.7 * loss_mse + 0.3 * loss_mae

                optimizer.zero_grad(set_to_none=True)
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

            scheduler.step()

            # ---- Validate ----
            model.eval()
            all_p, all_t = [], []
            with torch.no_grad():
                for seq, mean, tex, y, mask, _ in dl_va:
                    seq  = seq.to(env["device"], non_blocking=True)
                    mean = mean.to(env["device"], non_blocking=True)
                    tex  = tex.to(env["device"], non_blocking=True)
                    y    = y.to(env["device"], non_blocking=True)
                    mask = mask.to(env["device"], non_blocking=True)

                    pr = model(seq, mean, tex, mask).cpu().numpy()
                    yt = y.cpu().numpy()
                    mk = mask.cpu().numpy()
                    for b in range(pr.shape[0]):
                        all_p.append(pr[b, mk[b]])
                        all_t.append(yt[b, mk[b]])

            if not all_p:
                r2, rmse = -1.0, float("inf")
            else:
                yp = np.concatenate(all_p)
                yt = np.concatenate(all_t)
                r2 = float(r2_score(yt, yp))
                rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))

            writer.writerow([ep, f"{r2:.4f}", f"{rmse:.4f}"])
            hf.flush()
            print(f"[{name}] Ep{ep}: R²={r2:.4f}  RMSE={rmse:.4f}")

            if r2 > best_r2:
                best_r2, bad_epochs = r2, 0
                best_path = os.path.join(env["out_dir"], f"{name}.pth")
                torch.save(model.state_dict(), best_path)
            else:
                bad_epochs += 1
                if bad_epochs >= args.early_stop:
                    break

    return best_path, best_r2, name


# ------------------------------ Main ----------------------------- #

def main():
    args = parse_args()
    set_seed(args.seed)

    # Device
    if args.gpu is not None and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_device_arg_val = device.type  # "cuda" or "cpu" for torch.amp

    # I/O
    os.makedirs(args.out_dir, exist_ok=True)
    models_dir = os.path.join(args.out_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    # Load arrays
    mmap_mode = "r" if args.memmap else None
    S1 = np.load(os.path.join(args.data_dir, "fullyear_S1.npy"), mmap_mode=mmap_mode)  # (T,5,H,W)
    S2 = np.load(os.path.join(args.data_dir, "fullyear_S2.npy"), mmap_mode=mmap_mode)  # (T,H,W,3)
    with open(os.path.join(args.data_dir, "normalization_params.pkl"), "rb") as f:
        norm = pickle.load(f)
    good_pixel = np.load(os.path.join(args.data_dir, "good_pixel.npy"))  # (H,W) bool

    dates = norm["dates"]
    T, C, H, W = S1.shape
    ndvi_n = np.array(S2[..., 0], dtype=np.float32)
    cloud  = np.array(S2[..., 1] > 0, dtype=bool)
    water  = np.array(S2[..., 2] > 0, dtype=bool)

    doys = np.array([int(d[4:]) for d in dates], dtype=np.int32)
    sin_doy = np.sin(2 * pi * doys / 365).astype(np.float32)
    cos_doy = np.cos(2 * pi * doys / 365).astype(np.float32)

    # Pad S1 for texture patches
    pad = max(0, args.patch // 2)
    S1_pad = np.pad(S1, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="reflect")

    # Sampling
    coords = reservoir_bin_sampling(S1, S2, good_pixel.astype(bool), args.samples_per_bin)
    print(f"[INFO] Total training samples (after balancing): {len(coords):,}")

    # Channel combinations
    channel_index = {c: ALL_CHANNELS.index(c) for c in ALL_CHANNELS}
    combos = []

    if args.only_combos:
        # e.g., --only-combos vh,rvi vv_div,vv_diff,log_ratio
        for token in args.only_combos:
            parts = token.split(",")
            combos.append(parts)
    elif args.search_all_combos:
        base = [c for c in args.channels]
        combos = list(chain.from_iterable(combinations(base, r) for r in range(1, len(base) + 1)))
        combos = [list(c) for c in combos]
    else:
        combos = [args.channels]  # single run with the provided list

    print(f"[INFO] Number of combos to train: {len(combos)}")

    # Shared env
    env = dict(
        device=device,
        amp_device_arg_val=amp_device_arg_val,
        out_dir=models_dir,
        T=T,
        channel_index=channel_index,
        coords=coords,
        ds_kwargs=dict(
            S1=S1, S1_pad=S1_pad, ndvi_n=ndvi_n, cloud=cloud, water=water,
            sin_doy=sin_doy, cos_doy=cos_doy, patch=args.patch
        )
    )

    summary_fp = os.path.join(models_dir, "summary.csv")
    with open(summary_fp, "w", newline="") as sf:
        writer = csv.writer(sf)
        writer.writerow(["model_path", "best_r2", "combo"])

        for combo in combos:
            print("==== Training combo:", combo)
            best_path, r2, name = train_one_combo(combo, env, args)
            writer.writerow([best_path or "", f"{r2:.4f}", name])
            sf.flush()

    print(f"[INFO] Training complete. Models saved under: {models_dir}")


if __name__ == "__main__":
    main()

