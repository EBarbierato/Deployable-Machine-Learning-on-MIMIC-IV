#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python3
"""
08_train_mtl_variants.py

Purpose
-------
Train advanced multitask learning (MTL) variants for comorbidity prediction:
  - Cross-Stitch networks (two task-specific towers with learnable cross-stitch units)
  - MMoE (Mixture-of-Experts with per-task soft gating)

This script mirrors I/O and training semantics of 07_train_mtl_shared_bottom.py to enable
apples-to-apples comparisons.

Inputs
------
- X_train.parquet, X_val.parquet, X_test.parquet (from 05_impute_and_scale.py)
- labels.parquet (from 02_define_labels.py)
- Two boolean label columns (comma-separated)

Outputs
-------
{outdir}/
  config.json
  feature_names.json
  train_log.csv
  best.ckpt
  metrics.json
  VAL_preds.parquet
  TEST_preds.parquet

Examples
--------
# Cross-Stitch
python scripts/08_train_mtl_variants.py \
  --variant cross_stitch \
  --x-train artifacts/imputed/X_train.parquet \
  --x-val   artifacts/imputed/X_val.parquet \
  --x-test  artifacts/imputed/X_test.parquet \
  --labels  artifacts/labels/labels.parquet \
  --task-cols diab_incident_365d,cvd_incident_365d \
  --tower-sizes 256,128 \
  --dropout 0.2 \
  --epochs 200 --patience 20 \
  --lr 1e-3 --weight-decay 1e-4 \
  --select-metric auprc \
  --seed 42 \
  --outdir artifacts/mtl_cross_stitch

# MMoE
python scripts/08_train_mtl_variants.py \
  --variant mmoe \
  --x-train artifacts/imputed/X_train.parquet \
  --x-val   artifacts/imputed/X_val.parquet \
  --x-test  artifacts/imputed/X_test.parquet \
  --labels  artifacts/labels/labels.parquet \
  --task-cols diab_incident_365d,cvd_incident_365d \
  --experts 4 --expert-size 128 \
  --tower-sizes 64 \
  --dropout 0.2 \
  --epochs 200 --patience 20 \
  --lr 1e-3 --weight-decay 1e-4 \
  --select-metric auprc \
  --seed 42 \
  --outdir artifacts/mtl_mmoe
"""
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss
from torch.utils.data import Dataset, DataLoader


# ----------------------------- CLI ------------------------------------------ #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train Cross-Stitch or MMoE MTL models (PyTorch).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--variant", type=str, choices=["cross_stitch", "mmoe"], required=True)

    p.add_argument("--x-train", type=str, required=True)
    p.add_argument("--x-val",   type=str, required=True)
    p.add_argument("--x-test",  type=str, required=True)
    p.add_argument("--labels",  type=str, required=True)
    p.add_argument("--task-cols", type=str, required=True,
                   help="Comma-separated two label columns, e.g., diab_incident_365d,cvd_incident_365d")

    # Shared training hyperparams
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--loss-weights", type=str, default="1.0,1.0",
                   help="Weights for the two task losses (lambda1,lambda2).")
    p.add_argument("--pos-weight-mode", type=str, choices=["inverse_prevalence", "none"], default="inverse_prevalence")
    p.add_argument("--select-metric", type=str, choices=["auprc", "auroc"], default="auprc")
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--amp", type=str2bool, default=True, help="Use mixed precision if CUDA is available.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--outdir", type=str, default="artifacts/mtl_variant")

    # Cross-Stitch-specific
    p.add_argument("--tower-sizes", type=str, default="256,128",
                   help="Comma-separated hidden sizes for each task tower (same layout for both tasks).")
    p.add_argument("--stitch-init-alpha", type=float, default=0.9,
                   help="Initial diagonal value for cross-stitch matrices; off-diagonals start at 1-alpha.")

    # MMoE-specific
    p.add_argument("--experts", type=int, default=4, help="Number of experts.")
    p.add_argument("--expert-size", type=int, default=128, help="Hidden size of each expert output.")
    p.add_argument("--gate-hidden", type=int, default=64, help="Hidden size for gating nets (single hidden layer).")

    return p.parse_args()

def str2bool(v: str) -> bool:
    return str(v).lower() in {"1", "true", "t", "yes", "y"}


# ----------------------------- Data utils ----------------------------------- #

def _load_matrix(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "subject_id" not in df.columns:
        raise KeyError(f"{path} must contain 'subject_id'.")
    return df

def _load_labels(labels_path: Path, tasks: List[str]) -> pd.DataFrame:
    lab = pd.read_parquet(labels_path)
    for t in tasks:
        if t not in lab.columns:
            raise KeyError(f"Label column '{t}' not found in {labels_path}.")
    keep = ["subject_id"] + tasks
    lab = lab[keep].drop_duplicates("subject_id")
    for t in tasks:
        lab[t] = lab[t].astype(bool).astype(int)
    return lab

@dataclass
class XY:
    X: np.ndarray
    y: np.ndarray
    subject_id: np.ndarray
    feat_names: List[str]

def _join_Xy(X_df: pd.DataFrame, lab_df: pd.DataFrame, tasks: List[str]) -> XY:
    df = X_df.merge(lab_df, on="subject_id", how="inner")
    if df.empty:
        raise ValueError("Join of features and labels is empty. Check subject_id alignment.")
    feat_cols = [c for c in df.columns if c not in (["subject_id"] + tasks)]
    X = df[feat_cols].to_numpy(dtype=np.float32, copy=True)
    y = df[tasks].to_numpy(dtype=np.int64, copy=True)
    sid = df["subject_id"].to_numpy(dtype=np.int64, copy=True)
    return XY(X=X, y=y, subject_id=sid, feat_names=feat_cols)

class XYDataset(Dataset):
    def __init__(self, xy: XY):
        self.X = xy.X
        self.y = xy.y
        self.sid = xy.subject_id
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i: int):
        return self.X[i], self.y[i]


# ----------------------------- Models --------------------------------------- #

class CrossStitchUnit(nn.Module):
    """
    Learns a 2x2 mixing matrix A applied to feature maps x1, x2 with identical shape.
    y1 = a11*x1 + a12*x2
    y2 = a21*x1 + a22*x2
    """
    def __init__(self, init_alpha: float = 0.9):
        super().__init__()
        # Initialize near identity to encourage specialization: [[a, 1-a], [1-a, a]]
        a = float(init_alpha)
        init = torch.tensor([[a, 1 - a],
                             [1 - a, a]], dtype=torch.float32)
        self.A = nn.Parameter(init)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        A = self.A
        y1 = A[0, 0] * x1 + A[0, 1] * x2
        y2 = A[1, 0] * x1 + A[1, 1] * x2
        return y1, y2


class CrossStitchNet(nn.Module):
    """
    Two task-specific towers with cross-stitch units after each layer.
    Both towers share the same layer widths (tower_sizes).
    """
    def __init__(self, d_in: int, tower_sizes: List[int], dropout: float, stitch_alpha: float):
        super().__init__()
        sizes = [d_in] + tower_sizes
        self.layers1 = nn.ModuleList()
        self.layers2 = nn.ModuleList()
        self.stitches = nn.ModuleList()
        for i in range(1, len(sizes)):
            self.layers1.append(nn.Sequential(nn.Linear(sizes[i-1], sizes[i]), nn.ReLU(), nn.Dropout(dropout)))
            self.layers2.append(nn.Sequential(nn.Linear(sizes[i-1], sizes[i]), nn.ReLU(), nn.Dropout(dropout)))
            self.stitches.append(CrossStitchUnit(init_alpha=stitch_alpha))
        last = sizes[-1]
        self.head1 = nn.Linear(last, 1)
        self.head2 = nn.Linear(last, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z1 = x
        z2 = x
        for l1, l2, cs in zip(self.layers1, self.layers2, self.stitches):
            z1 = l1(z1)
            z2 = l2(z2)
            z1, z2 = cs(z1, z2)
        return self.head1(z1).squeeze(-1), self.head2(z2).squeeze(-1)


class ExpertMLP(nn.Module):
    def __init__(self, d_in: int, d_out: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_out),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_out, d_out),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Gate(nn.Module):
    """Single-hidden-layer gate producing softmax weights over K experts."""
    def __init__(self, d_in: int, k: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, k)
        )
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.softmax(self.net(x))


class MMoE(nn.Module):
    """
    Mixture-of-Experts with K experts shared across tasks.
    Each task has a gate; task-specific tower(s) after mixture and then a logistic head.
    """
    def __init__(self, d_in: int, k: int, expert_size: int, tower_sizes: List[int], gate_hidden: int, dropout: float):
        super().__init__()
        self.k = k
        self.experts = nn.ModuleList([ExpertMLP(d_in, expert_size, dropout) for _ in range(k)])
        self.gate1 = Gate(d_in, k, gate_hidden)
        self.gate2 = Gate(d_in, k, gate_hidden)
        def make_tower(d0: int, sizes: List[int]) -> nn.Sequential:
            layers = []
            last = d0
            for h in sizes:
                layers += [nn.Linear(last, h), nn.ReLU(), nn.Dropout(dropout)]
                last = h
            return nn.Sequential(*layers) if layers else nn.Identity()
        self.tower1 = make_tower(expert_size, tower_sizes)
        self.tower2 = make_tower(expert_size, tower_sizes)
        self.head1 = nn.Linear(tower_sizes[-1] if tower_sizes else expert_size, 1)
        self.head2 = nn.Linear(tower_sizes[-1] if tower_sizes else expert_size, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Compute expert outputs [B, k, d]
        expert_outs = torch.stack([e(x) for e in self.experts], dim=1)  # (B, k, d)
        # Gates
        g1 = self.gate1(x).unsqueeze(-1)  # (B, k, 1)
        g2 = self.gate2(x).unsqueeze(-1)  # (B, k, 1)
        # Weighted sums
        m1 = torch.sum(g1 * expert_outs, dim=1)  # (B, d)
        m2 = torch.sum(g2 * expert_outs, dim=1)  # (B, d)
        # Task towers + heads
        z1 = self.tower1(m1)
        z2 = self.tower2(m2)
        return self.head1(z1).squeeze(-1), self.head2(z2).squeeze(-1)


# ----------------------------- Metrics -------------------------------------- #

def _compute_metrics(y_true: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    eps = 1e-12
    y = y_true.astype(int)
    p = np.clip(p, eps, 1 - eps)
    out = {}
    try:
        out["auroc"] = float(roc_auc_score(y, p))
    except ValueError:
        out["auroc"] = float("nan")
    try:
        out["auprc"] = float(average_precision_score(y, p))
    except ValueError:
        out["auprc"] = float("nan")
    out["brier"] = float(brier_score_loss(y, p))
    try:
        out["logloss"] = float(log_loss(y, p))
    except ValueError:
        out["logloss"] = float("nan")
    out["pos_rate"] = float(np.mean(y))
    return out

def _early_stop_score(m1: Dict[str, float], m2: Dict[str, float], key: str) -> float:
    a = m1.get(key, float("nan"))
    b = m2.get(key, float("nan"))
    if math.isnan(a) or math.isnan(b):
        return -float("inf")
    return 0.5 * (a + b)


# ----------------------------- Determinism ---------------------------------- #

def set_seeds(seed: int) -> None:
    import random
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
    try:
        import torch.backends.cudnn as cudnn
        cudnn.deterministic = True
        cudnn.benchmark = False
    except Exception:
        pass
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


# ----------------------------- Training ------------------------------------- #

def train_loop(
    model: nn.Module,
    loaders: Dict[str, DataLoader],
    device: torch.device,
    loss_fns: Tuple[nn.Module, nn.Module],
    loss_weights: Tuple[float, float],
    epochs: int,
    patience: int,
    optimizer: torch.optim.Optimizer,
    grad_clip: float,
    use_amp: bool,
    select_metric: str,
    outdir: Path,
) -> Dict[str, float]:
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    best = {"score": -float("inf"), "epoch": -1}
    history_rows: List[str] = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        n_batches = 0
        for xb, yb in loaders["train"]:
            xb = xb.to(device, non_blocking=True).float()
            y1 = yb[:, 0].to(device, non_blocking=True).float()
            y2 = yb[:, 1].to(device, non_blocking=True).float()

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                l1, l2 = model(xb)
                loss1 = loss_fns[0](l1, y1)
                loss2 = loss_fns[1](l2, y2)
                loss = loss_weights[0] * loss1 + loss_weights[1] * loss2
            scaler.scale(loss).backward()
            if grad_clip and grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            scaler.step(optimizer)
            scaler.update()

            train_loss += float(loss.item())
            n_batches += 1

        # Validation
        model.eval()
        with torch.no_grad():
            p1_val, p2_val, y1_val, y2_val = [], [], [], []
            for xb, yb in loaders["val"]:
                xb = xb.to(device).float()
                yb = yb.to(device).float()
                l1, l2 = model(xb)
                p1 = torch.sigmoid(l1).cpu().numpy()
                p2 = torch.sigmoid(l2).cpu().numpy()
                p1_val.append(p1); p2_val.append(p2)
                y1_val.append(yb[:, 0].cpu().numpy()); y2_val.append(yb[:, 1].cpu().numpy())
            p1 = np.concatenate(p1_val); p2 = np.concatenate(p2_val)
            y1 = np.concatenate(y1_val); y2 = np.concatenate(y2_val)
            m1 = _compute_metrics(y1, p1)
            m2 = _compute_metrics(y2, p2)
            score = _early_stop_score(m1, m2, select_metric)

        row = f"{epoch},{train_loss/max(1,n_batches):.6f},{m1['auroc']:.5f},{m1['auprc']:.5f},{m2['auroc']:.5f},{m2['auprc']:.5f},{score:.5f}"
        history_rows.append(row)
        (outdir / "train_log.csv").write_text("epoch,train_loss,head1_auroc,head1_auprc,head2_auroc,head2_auprc,early_score\n" + "\n".join(history_rows))

        # Early stopping
        improved = score > best["score"]
        if improved:
            best.update({"score": score, "epoch": epoch})
            torch.save({"model_state": model.state_dict(),
                        "epoch": epoch, "score": score}, outdir / "best.ckpt")
        elif epoch - best["epoch"] >= patience:
            break

    return {"best_epoch": best["epoch"], "best_score": best["score"]}


# ----------------------------- Orchestration -------------------------------- #

def main() -> int:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    set_seeds(args.seed)

    device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
    use_amp = bool(args.amp and device.type == "cuda")

    task_cols = [s.strip() for s in args.task_cols.split(",") if s.strip()]
    if len(task_cols) != 2:
        raise ValueError("--task-cols must specify exactly two label columns.")

    # Load data
    Xtr_df = _load_matrix(Path(args.x_train))
    Xva_df = _load_matrix(Path(args.x_val))
    Xte_df = _load_matrix(Path(args.x_test))
    lab_df = _load_labels(Path(args.labels), task_cols)

    tr = _join_Xy(Xtr_df, lab_df, task_cols)
    va = _join_Xy(Xva_df, lab_df, task_cols)
    te = _join_Xy(Xte_df, lab_df, task_cols)

    # Datasets/Loaders
    ds_tr = XYDataset(tr)
    ds_va = XYDataset(va)
    ds_te = XYDataset(te)
    loader_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=0, pin_memory=(device.type=="cuda"))
    loader_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=0, pin_memory=(device.type=="cuda"))
    loader_te = DataLoader(ds_te, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=0, pin_memory=(device.type=="cuda"))
    loaders = {"train": loader_tr, "val": loader_va, "test": loader_te}

    # Build model
    if args.variant == "cross_stitch":
        tower_sizes = [int(x) for x in args.tower_sizes.split(",") if x.strip()]
        model = CrossStitchNet(d_in=tr.X.shape[1], tower_sizes=tower_sizes,
                               dropout=args.dropout, stitch_alpha=args.stitch_init_alpha).to(device)
    else:  # mmoe
        tower_sizes = [int(x) for x in args.tower_sizes.split(",") if x.strip()]
        model = MMoE(d_in=tr.X.shape[1], k=int(args.experts), expert_size=int(args.expert_size),
                     tower_sizes=tower_sizes, gate_hidden=int(args.gate_hidden),
                     dropout=args.dropout).to(device)

    # Loss functions with pos_weight from TRAIN
    def pos_weight_from_train(y_col: np.ndarray) -> torch.Tensor:
        pos = float(y_col.sum())
        neg = float(len(y_col) - pos)
        w = (neg / max(1.0, pos)) if pos > 0 else 1.0
        return torch.tensor(w, dtype=torch.float32, device=device)

    if args.pos_weight_mode == "inverse_prevalence":
        w1 = pos_weight_from_train(tr.y[:, 0])
        w2 = pos_weight_from_train(tr.y[:, 1])
        loss1 = nn.BCEWithLogitsLoss(pos_weight=w1)
        loss2 = nn.BCEWithLogitsLoss(pos_weight=w2)
    else:
        loss1 = nn.BCEWithLogitsLoss()
        loss2 = nn.BCEWithLogitsLoss()

    lw = [float(x) for x in args.loss_weights.split(",")]
    if len(lw) != 2:
        raise ValueError("--loss-weights must provide two floats.")
    loss_weights = (lw[0], lw[1])

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Train
    train_stats = train_loop(
      model=model, loaders=loaders, device=device,
      loss_fns=(loss1, loss2), loss_weights=loss_weights,
      epochs=args.epochs, patience=args.patience,
      optimizer=optimizer, grad_clip=args.grad_clip,
      use_amp=use_amp, select_metric=args.select_metric, outdir=outdir,
    )

    # Load best and evaluate on VAL/TEST
    ckpt = torch.load(outdir / "best.ckpt", map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    def eval_split(loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        ps1, ps2, ys1, ys2 = [], [], [], []
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(device).float()
                yb = yb.to(device).float()
                l1, l2 = model(xb)
                ps1.append(torch.sigmoid(l1).cpu().numpy())
                ps2.append(torch.sigmoid(l2).cpu().numpy())
                ys1.append(yb[:, 0].cpu().numpy())
                ys2.append(yb[:, 1].cpu().numpy())
        return (np.concatenate(ys1), np.concatenate(ps1)), (np.concatenate(ys2), np.concatenate(ps2))

    (y1_v, p1_v), (y2_v, p2_v) = eval_split(loaders["val"])
    (y1_t, p1_t), (y2_t, p2_t) = eval_split(loaders["test"])

    m1_val = _compute_metrics(y1_v, p1_v); m2_val = _compute_metrics(y2_v, p2_v)
    m1_test = _compute_metrics(y1_t, p1_t); m2_test = _compute_metrics(y2_t, p2_t)

    # Save predictions
    def write_preds(split_name: str, xy: XY, p1: np.ndarray, p2: np.ndarray, y1: np.ndarray, y2: np.ndarray) -> None:
        df = pd.DataFrame({
            "subject_id": xy.subject_id,
            f"{task_cols[0]}_y_true": y1.astype(int),
            f"{task_cols[0]}_y_score": p1.astype(np.float32),
            f"{task_cols[1]}_y_true": y2.astype(int),
            f"{task_cols[1]}_y_score": p2.astype(np.float32),
        })
        df.to_parquet(outdir / f"{split_name}_preds.parquet", index=False)

    write_preds("VAL", va, p1_v, p2_v, y1_v, y2_v)
    write_preds("TEST", te, p1_t, p2_t, y1_t, y2_t)

    # Metrics JSON
    metrics_payload = {
        "variant": args.variant,
        "task_cols": task_cols,
        "select_metric": args.select_metric,
        "best_epoch": int(train_stats["best_epoch"]),
        "best_val_score": float(train_stats["best_score"]),
        "val": {
            task_cols[0]: m1_val,
            task_cols[1]: m2_val,
            "early_stop_score": _early_stop_score(m1_val, m2_val, args.select_metric),
        },
        "test": {
            task_cols[0]: m1_test,
            task_cols[1]: m2_test,
        },
        "generated_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    (outdir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2))

    # Save config and feature names
    cfg = {
        "variant": args.variant,
        "dropout": args.dropout,
        "epochs": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "loss_weights": loss_weights,
        "pos_weight_mode": args.pos_weight_mode,
        "select_metric": args.select_metric,
        "grad_clip": args.grad_clip,
        "amp": args.amp,
        "seed": args.seed,
        "device": str(device),
        "files": {
            "x_train": str(Path(args.x_train)),
            "x_val": str(Path(args.x_val)),
            "x_test": str(Path(args.x_test)),
            "labels": str(Path(args.labels)),
        }
    }
    if args.variant == "cross_stitch":
        cfg.update({"tower_sizes": [int(x) for x in args.tower_sizes.split(',') if x.strip()],
                    "stitch_init_alpha": args.stitch_init_alpha})
    else:
        cfg.update({"experts": int(args.experts),
                    "expert_size": int(args.expert_size),
                    "gate_hidden": int(args.gate_hidden),
                    "tower_sizes": [int(x) for x in args.tower_sizes.split(',') if x.strip()]})
    (outdir / "config.json").write_text(json.dumps(cfg, indent=2))
    (outdir / "feature_names.json").write_text(json.dumps(tr.feat_names, indent=2))

    # Console summary
    print(f"Variant: {args.variant} | Best epoch: {train_stats['best_epoch']} | "
          f"best val {args.select_metric} mean: {train_stats['best_score']:.4f}")
    print(f"VAL {task_cols[0]} AUC={m1_val['auroc']:.4f} AUPRC={m1_val['auprc']:.4f} | "
          f"{task_cols[1]} AUC={m2_val['auroc']:.4f} AUPRC={m2_val['auprc']:.4f}")
    print(f"TEST {task_cols[0]} AUC={m1_test['auroc']:.4f} AUPRC={m1_test['auprc']:.4f} | "
          f"{task_cols[1]} AUC={m2_test['auroc']:.4f} AUPRC={m2_test['auprc']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

