#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python3
"""
14_external_validation.py

Purpose
-------
Evaluate trained models on an external cohort (features + labels). Optionally:
  - Apply calibrators learned on the internal validation set (from 09).
  - Evaluate thresholded metrics on the external cohort using thresholds chosen on VAL (from 09).

Supported model families:
  A) Baselines (06_train_baselines.py): LogisticRegression, RandomForest, XGBoost
  B) MTL models (07/08): Shared-Bottom, Cross-Stitch, or MMoE (auto-detected from config.json)

Inputs
------
Required (external cohort):
  --x-ext           Parquet with external features; must include 'subject_id' and the same feature columns used in training
  --labels-ext      Parquet with external labels; must include 'subject_id' and the target column(s)

Choose ONE of:
  --baseline-model-path  Path to model.joblib from 06 (use with --label-col)
  --mtl-run-dir          Directory from 07/08 (must contain config.json, feature_names.json, best.ckpt)

If baseline:
  --label-col       Name of incident label to evaluate (e.g., diab_incident_365d)

If MTL:
  --task-cols       Comma-separated list of two label columns (e.g., diab_incident_365d,cvd_incident_365d)

Optional:
  --calibration-params   Path to calibration_params.json from 09 (to apply Platt/Isotonic fitted on VAL)
  --thresholds-json      Path to thresholds.json from 09 (to compute thresholded metrics on EXTERNAL)
  --ece-bins             Number of bins for ECE (default 15)
  --outdir               Output directory (default artifacts/external_validation)

Outputs
-------
For baselines:
  {outdir}/external_preds.parquet           # [subject_id, y_true, y_score_raw, y_score_platt?, y_score_isotonic?]
  {outdir}/metrics.json
  {outdir}/METRICS.md
  {outdir}/threshold_tables.json            # optional, if --thresholds-json is provided

For MTL:
  {outdir}/{task}/external_preds.parquet    # one file per task with raw+calibrated scores
  {outdir}/metrics.json                     # includes both tasks
  {outdir}/METRICS.md
  {outdir}/{task}/threshold_tables.json     # per task, if thresholds provided

Examples
--------
# Baseline model on an external cohort, with calibration + thresholds from 09
python scripts/14_external_validation.py \
  --x-ext data_ext/X_external.parquet \
  --labels-ext data_ext/labels_external.parquet \
  --baseline-model-path artifacts/baselines/diab_incident_365d/logistic/model.joblib \
  --label-col diab_incident_365d \
  --calibration-params artifacts/calibration/diab_logistic/calibration_params.json \
  --thresholds-json artifacts/calibration/diab_logistic/thresholds.json \
  --outdir artifacts/external_validation/diab_logistic

# MTL shared-bottom run on an external cohort (two tasks)
python scripts/14_external_validation.py \
  --x-ext data_ext/X_external.parquet \
  --labels-ext data_ext/labels_external.parquet \
  --mtl-run-dir artifacts/mtl_shared_bottom \
  --task-cols diab_incident_365d,cvd_incident_365d \
  --calibration-params artifacts/calibration/mtl_shared_bottom_diab/calibration_params.json \
  --thresholds-json artifacts/calibration/mtl_shared_bottom_diab/thresholds.json \
  --outdir artifacts/external_validation/mtl_shared_bottom
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss, log_loss
)

# Optional torch/xgboost for MTL and baseline-xgb
try:
    import torch
    import torch.nn as nn
    _HAVE_TORCH = True
except Exception:
    torch = None  # type: ignore
    nn = None     # type: ignore
    _HAVE_TORCH = False

try:
    import xgboost as xgb  # type: ignore
    _HAVE_XGB = True
except Exception:
    xgb = None  # type: ignore
    _HAVE_XGB = False


# ----------------------------- CLI ------------------------------------------ #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="External validation for baseline or MTL models (with optional calibration & thresholds).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--x-ext", type=str, required=True)
    p.add_argument("--labels-ext", type=str, required=True)

    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--baseline-model-path", type=str, help="Path to 06 model.joblib")
    grp.add_argument("--mtl-run-dir", type=str, help="Directory from 07/08 with config.json + best.ckpt")

    p.add_argument("--label-col", type=str, help="(Baseline) label column to evaluate")
    p.add_argument("--task-cols", type=str, help="(MTL) two label columns, comma-separated")

    p.add_argument("--calibration-params", type=str, default=None,
                   help="calibration_params.json from 09 (to apply Platt/Isotonic)")
    p.add_argument("--thresholds-json", type=str, default=None,
                   help="thresholds.json from 09 (to compute thresholded metrics on EXTERNAL)")
    p.add_argument("--ece-bins", type=int, default=15)
    p.add_argument("--outdir", type=str, default="artifacts/external_validation")
    return p.parse_args()


# ----------------------------- IO helpers ----------------------------------- #

def _load_parquet(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "subject_id" not in df.columns:
        raise KeyError(f"{path} must contain 'subject_id'.")
    return df

def _align_features(X_ext: pd.DataFrame, feature_names: List[str]) -> np.ndarray:
    missing = [c for c in feature_names if c not in X_ext.columns]
    if missing:
        raise KeyError(f"External features missing {len(missing)} columns; first few: {missing[:10]}")
    return X_ext[feature_names].to_numpy(dtype=np.float32, copy=True)

def _extract_label_map(labels_df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col not in labels_df.columns:
        raise KeyError(f"labels file missing '{col}'.")
    out = labels_df[["subject_id", col]].drop_duplicates("subject_id").copy()
    out[col] = out[col].astype(bool).astype(int)
    return out

def _extract_label_map_multi(labels_df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c not in labels_df.columns:
            raise KeyError(f"labels file missing '{c}'.")
    out = labels_df[["subject_id"] + cols].drop_duplicates("subject_id").copy()
    for c in cols:
        out[c] = out[c].astype(bool).astype(int)
    return out


# ----------------------------- Metrics -------------------------------------- #

def _metrics(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    eps = 1e-12
    y = y.astype(int)
    p = np.clip(p.astype(float), eps, 1 - eps)
    out = {}
    try: out["auroc"] = float(roc_auc_score(y, p))
    except ValueError: out["auroc"] = float("nan")
    try: out["auprc"] = float(average_precision_score(y, p))
    except ValueError: out["auprc"] = float("nan")
    out["brier"] = float(brier_score_loss(y, p))
    try: out["logloss"] = float(log_loss(y, p))
    except ValueError: out["logloss"] = float("nan")
    out["pos_rate"] = float(np.mean(y))
    out["n"] = int(len(y))
    return out

def _ece(y: np.ndarray, p: np.ndarray, n_bins: int = 15) -> float:
    y = y.astype(int); p = np.clip(p.astype(float), 1e-6, 1-1e-6)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    inds = np.digitize(p, bins) - 1
    total = len(p)
    ece = 0.0
    for b in range(n_bins):
        m = inds == b
        kb = np.sum(m)
        if kb == 0: 
            continue
        conf = np.mean(p[m])
        acc  = np.mean(y[m])
        ece += (kb / total) * abs(acc - conf)
    return float(ece)

def _confusion_at_thresh(y: np.ndarray, p: np.ndarray, t: float) -> Dict[str, float]:
    y = y.astype(int); p = p.astype(float)
    pred = (p >= t).astype(int)
    tp = int(np.sum((pred == 1) & (y == 1)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    tn = int(np.sum((pred == 0) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    sens = tp / max(1, tp + fn)
    spec = tn / max(1, tn + fp)
    ppv  = tp / max(1, tp + fp)
    npv  = tn / max(1, tn + fn)
    acc  = (tp + tn) / max(1, tp + tn + fp + fn)
    f1   = (2 * tp) / max(1, 2 * tp + fp + fn)
    return {"threshold": t, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "sensitivity": sens, "specificity": spec, "ppv": ppv, "npv": npv, "accuracy": acc, "f1": f1}


# ----------------------------- Calibration utils ---------------------------- #

@dataclass
class PlattParams:
    A: float
    B: float
    eps: float = 1e-6

def _apply_platt(p: np.ndarray, prm: PlattParams) -> np.ndarray:
    p = np.clip(p, prm.eps, 1 - prm.eps)
    z = prm.B + prm.A * np.log(p / (1 - p))
    return 1.0 / (1.0 + np.exp(-z))

@dataclass
class IsoParams:
    x: List[float]
    y: List[float]

def _apply_iso(p: np.ndarray, prm: IsoParams) -> np.ndarray:
    x = np.array(prm.x, dtype=float)
    y = np.array(prm.y, dtype=float)
    idx = np.searchsorted(x, p, side="right") - 1
    idx = np.clip(idx, 0, len(y) - 1)
    return y[idx]

def load_calibration_params(path: Optional[Path]) -> Dict[str, object]:
    if not path:
        return {}
    meta = json.loads(path.read_text())
    out = {}
    prm = meta.get("params", {})
    if "platt" in prm:
        A = float(prm["platt"]["A"]); B = float(prm["platt"]["B"]); eps = float(prm["platt"].get("eps", 1e-6))
        out["platt"] = PlattParams(A=A, B=B, eps=eps)
    if "isotonic" in prm:
        out["isotonic"] = IsoParams(x=list(prm["isotonic"]["x"]), y=list(prm["isotonic"]["y"]))
    return out


# ----------------------------- Baseline path -------------------------------- #

def eval_baseline(model_path: Path, x_ext_df: pd.DataFrame, labels_df: pd.DataFrame,
                  label_col: str, cal_params: Dict[str, object], ece_bins: int,
                  thresholds: Optional[dict], outdir: Path) -> Dict[str, object]:
    # Load model
    model = joblib.load(model_path)
    # Determine feature order from X columns minus subject_id
    feat_names = [c for c in x_ext_df.columns if c != "subject_id"]
    X = x_ext_df[feat_names].to_numpy(dtype=np.float32, copy=True)

    ymap = _extract_label_map(labels_df, label_col)
    df = x_ext_df[["subject_id"]].merge(ymap, on="subject_id", how="inner")
    if df.empty:
        raise ValueError("Join of external features and labels is empty.")
    y = df[label_col].astype(int).values

    # Predict
    p_raw = model.predict_proba(X)[:, 1]
    scores = {"raw": p_raw}
    if "platt" in cal_params:
        scores["platt"] = _apply_platt(p_raw, cal_params["platt"])  # type: ignore
    if "isotonic" in cal_params:
        scores["isotonic"] = _apply_iso(p_raw, cal_params["isotonic"])  # type: ignore

    # Metrics per variant
    metrics = {}
    for name, p in scores.items():
        m = _metrics(y, p)
        m["ece"] = _ece(y, p, n_bins=ece_bins)
        metrics[name] = m

    # Threshold tables if provided
    thresh_tables = {}
    if thresholds and isinstance(thresholds, dict):
        thmap = thresholds.get("thresholds", {})
        # Keys like 'raw','platt','isotonic' expected
        for variant, rules in thmap.items():
            if variant not in scores:
                continue
            p = scores[variant]
            tbl = {rule: _confusion_at_thresh(y, p, float(t)) for rule, t in rules.items()}
            thresh_tables[variant] = tbl
        (outdir / "threshold_tables.json").write_text(json.dumps(thresh_tables, indent=2))

    # Persist predictions
    out = pd.DataFrame({"subject_id": df["subject_id"].values, "y_true": y, "y_score_raw": scores["raw"].astype(np.float32)})
    if "platt" in scores:
        out["y_score_platt"] = scores["platt"].astype(np.float32)
    if "isotonic" in scores:
        out["y_score_isotonic"] = scores["isotonic"].astype(np.float32)
    out.to_parquet(outdir / "external_preds.parquet", index=False)

    # Save metrics
    payload = {
        "family": "baseline",
        "label": label_col,
        "metrics": metrics,
        "n": int(len(y)),
        "generated_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    (outdir / "metrics.json").write_text(json.dumps(payload, indent=2))

    # Markdown summary
    def fmt(m: Dict[str, float]) -> str:
        return (f"AUROC {m.get('auroc', float('nan')):.4f} | "
                f"AUPRC {m.get('auprc', float('nan')):.4f} | "
                f"Brier {m.get('brier', float('nan')):.4f} | "
                f"LogLoss {m.get('logloss', float('nan')):.4f} | "
                f"ECE {m.get('ece', float('nan')):.4f} | n={m.get('n', 0)}")
    lines = []
    lines.append(f"# External Validation (Baseline)\n\nGenerated: {datetime.utcnow().isoformat(timespec='seconds')}Z\n\n")
    lines.append(f"**Label:** {label_col}\n\n")
    for name in scores.keys():
        lines.append(f"- **{name}**: {fmt(metrics[name])}\n")
    if thresh_tables:
        lines.append("\n## Threshold tables (external)\n")
        for variant, tbl in thresh_tables.items():
            lines.append(f"### Variant: {variant}\n")
            for rule, d in tbl.items():
                lines.append(f"- {rule}: thr={d['threshold']:.4f} | sens={d['sensitivity']:.3f} spec={d['specificity']:.3f} "
                             f"ppv={d['ppv']:.3f} npv={d['npv']:.3f} f1={d['f1']:.3f} "
                             f"tp={d['tp']} fp={d['fp']} tn={d['tn']} fn={d['fn']}\n")
    (outdir / "METRICS.md").write_text("".join(lines))
    return payload


# ----------------------------- MTL path ------------------------------------- #
# Recreate the minimal model classes used in 07/08

class SharedBottom(nn.Module):  # type: ignore[misc]
    def __init__(self, d_in: int, hidden: List[int], dropout: float):
        super().__init__()
        layers = []
        last = d_in
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU(), nn.Dropout(dropout)]
            last = h
        self.shared = nn.Sequential(*layers) if layers else nn.Identity()
        self.head1 = nn.Linear(last, 1)
        self.head2 = nn.Linear(last, 1)
    def forward(self, x: torch.Tensor):
        z = self.shared(x)
        return self.head1(z).squeeze(-1), self.head2(z).squeeze(-1)

class CrossStitchUnit(nn.Module):  # type: ignore[misc]
    def __init__(self, init_alpha: float = 0.9):
        super().__init__()
        a = float(init_alpha)
        init = torch.tensor([[a, 1 - a],[1 - a, a]], dtype=torch.float32)
        self.A = nn.Parameter(init)
    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        A = self.A
        y1 = A[0,0]*x1 + A[0,1]*x2
        y2 = A[1,0]*x1 + A[1,1]*x2
        return y1, y2

class CrossStitchNet(nn.Module):  # type: ignore[misc]
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
    def forward(self, x: torch.Tensor):
        z1 = x; z2 = x
        for l1, l2, cs in zip(self.layers1, self.layers2, self.stitches):
            z1 = l1(z1); z2 = l2(z2); z1, z2 = cs(z1, z2)
        return self.head1(z1).squeeze(-1), self.head2(z2).squeeze(-1)

class ExpertMLP(nn.Module):  # type: ignore[misc]
    def __init__(self, d_in: int, d_out: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_out), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_out, d_out), nn.ReLU(), nn.Dropout(dropout),
        )
    def forward(self, x: torch.Tensor):
        return self.net(x)

class Gate(nn.Module):  # type: ignore[misc]
    def __init__(self, d_in: int, k: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Linear(hidden, k))
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x: torch.Tensor):
        return self.softmax(self.net(x))

class MMoE(nn.Module):  # type: ignore[misc]
    def __init__(self, d_in: int, k: int, expert_size: int, tower_sizes: List[int], gate_hidden: int, dropout: float):
        super().__init__()
        self.k = k
        self.experts = nn.ModuleList([ExpertMLP(d_in, expert_size, dropout) for _ in range(k)])
        self.gate1 = Gate(d_in, k, gate_hidden)
        self.gate2 = Gate(d_in, k, gate_hidden)
        def make_tower(d0: int, sizes: List[int]) -> nn.Sequential:
            layers = []; last = d0
            for h in sizes: layers += [nn.Linear(last, h), nn.ReLU(), nn.Dropout(dropout)]; last = h
            return nn.Sequential(*layers) if layers else nn.Identity()
        self.tower1 = make_tower(expert_size, tower_sizes)
        self.tower2 = make_tower(expert_size, tower_sizes)
        self.head1 = nn.Linear(tower_sizes[-1] if tower_sizes else expert_size, 1)
        self.head2 = nn.Linear(tower_sizes[-1] if tower_sizes else expert_size, 1)
    def forward(self, x: torch.Tensor):
        expert_outs = torch.stack([e(x) for e in self.experts], dim=1)  # (B,k,d)
        g1 = self.gate1(x).unsqueeze(-1); g2 = self.gate2(x).unsqueeze(-1)
        m1 = torch.sum(g1 * expert_outs, dim=1); m2 = torch.sum(g2 * expert_outs, dim=1)
        z1 = self.tower1(m1); z2 = self.tower2(m2)
        return self.head1(z1).squeeze(-1), self.head2(z2).squeeze(-1)

def _build_mtl_from_config(cfg: dict, d_in: int) -> nn.Module:
    variant = cfg.get("variant", "shared_bottom")
    dropout = float(cfg.get("dropout", 0.2))
    if variant == "shared_bottom":
        hidden = [int(x) for x in cfg.get("hidden_sizes", [256, 128])]
        return SharedBottom(d_in=d_in, hidden=hidden, dropout=dropout)
    if variant == "cross_stitch":
        tower = [int(x) for x in cfg.get("tower_sizes", [256, 128])]
        alpha = float(cfg.get("stitch_init_alpha", 0.9))
        return CrossStitchNet(d_in=d_in, tower_sizes=tower, dropout=dropout, stitch_alpha=alpha)
    if variant == "mmoe":
        k = int(cfg.get("experts", 4))
        es = int(cfg.get("expert_size", 128))
        gh = int(cfg.get("gate_hidden", 64))
        tower = [int(x) for x in cfg.get("tower_sizes", [64])]
        return MMoE(d_in=d_in, k=k, expert_size=es, tower_sizes=tower, gate_hidden=gh, dropout=dropout)
    # Treat unknown as shared-bottom
    hidden = [int(x) for x in cfg.get("hidden_sizes", [256, 128])]
    return SharedBottom(d_in=d_in, hidden=hidden, dropout=dropout)

def eval_mtl(run_dir: Path, x_ext_df: pd.DataFrame, labels_df: pd.DataFrame,
             task_cols: List[str], cal_params: Dict[str, object], ece_bins: int,
             thresholds: Optional[dict], outdir: Path) -> Dict[str, object]:
    if not _HAVE_TORCH:
        raise RuntimeError("PyTorch is required for MTL external validation.")
    cfg = json.loads((run_dir / "config.json").read_text())
    feat_names = json.loads((run_dir / "feature_names.json").read_text())
    d_in = len(feat_names)
    model = _build_mtl_from_config(cfg, d_in=d_in)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # Load checkpoint
    ckpt = torch.load(run_dir / "best.ckpt", map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Align features and labels
    X_mat = _align_features(x_ext_df, feat_names)
    lab = _extract_label_map_multi(labels_df, task_cols)
    df = x_ext_df[["subject_id"]].merge(lab, on="subject_id", how="inner")
    if df.empty:
        raise ValueError("Join of external features and labels is empty.")
    Y = df[task_cols].astype(int).to_numpy(copy=True)
    with torch.no_grad():
        xb = torch.from_numpy(X_mat).to(device).float()
        l1, l2 = model(xb)
        p1 = torch.sigmoid(l1).cpu().numpy()
        p2 = torch.sigmoid(l2).cpu().numpy()
    raw_map = {task_cols[0]: p1, task_cols[1]: p2}

    # For each task, assemble scores (raw + calibrated) and write predictions
    all_metrics = {}
    for idx, task in enumerate(task_cols):
        y = Y[:, idx]
        out_task_dir = outdir / task
        out_task_dir.mkdir(parents=True, exist_ok=True)

        p_raw = raw_map[task]
        scores = {"raw": p_raw}
        if "platt" in cal_params:
            scores["platt"] = _apply_platt(p_raw, cal_params["platt"])  # type: ignore
        if "isotonic" in cal_params:
            scores["isotonic"] = _apply_iso(p_raw, cal_params["isotonic"])  # type: ignore

        # Metrics
        m_task = {}
        for name, p in scores.items():
            mm = _metrics(y, p); mm["ece"] = _ece(y, p, n_bins=ece_bins)
            m_task[name] = mm
        all_metrics[task] = m_task

        # Threshold tables (if thresholds provided)
        if thresholds and isinstance(thresholds, dict):
            thmap = thresholds.get("thresholds", {})
            tbl_all = {}
            for variant, rules in thmap.items():
                if variant not in scores:
                    continue
                p = scores[variant]
                tbl = {rule: _confusion_at_thresh(y, p, float(t)) for rule, t in rules.items()}
                tbl_all[variant] = tbl
            (out_task_dir / "threshold_tables.json").write_text(json.dumps(tbl_all, indent=2))

        # Persist predictions for this task
        outp = pd.DataFrame({"subject_id": df["subject_id"].values,
                             f"{task}_y_true": y.astype(int),
                             f"{task}_y_score_raw": scores["raw"].astype(np.float32)})
        if "platt" in scores:
            outp[f"{task}_y_score_platt"] = scores["platt"].astype(np.float32)
        if "isotonic" in scores:
            outp[f"{task}_y_score_isotonic"] = scores["isotonic"].astype(np.float32)
        outp.to_parquet(out_task_dir / "external_preds.parquet", index=False)

    # Save metrics
    payload = {
        "family": "mtl",
        "variant": cfg.get("variant", "shared_bottom"),
        "tasks": task_cols,
        "metrics": all_metrics,
        "n": int(len(df)),
        "generated_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    (outdir / "metrics.json").write_text(json.dumps(payload, indent=2))

    # Markdown summary
    def fmt(m: Dict[str, float]) -> str:
        return (f"AUROC {m.get('auroc', float('nan')):.4f} | "
                f"AUPRC {m.get('auprc', float('nan')):.4f} | "
                f"Brier {m.get('brier', float('nan')):.4f} | "
                f"LogLoss {m.get('logloss', float('nan')):.4f} | "
                f"ECE {m.get('ece', float('nan')):.4f} | n={m.get('n', 0)}")
    lines = []
    lines.append(f"# External Validation (MTL: {payload['variant']})\n\nGenerated: {datetime.utcnow().isoformat(timespec='seconds')}Z\n\n")
    for task in task_cols:
        lines.append(f"## Task: {task}\n")
        for name, mm in all_metrics[task].items():
            lines.append(f"- **{name}**: {fmt(mm)}\n")
        lines.append("\n")
    (outdir / "METRICS.md").write_text("".join(lines))
    return payload


# ----------------------------- MAIN ----------------------------------------- #

def main() -> int:
    args = parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    x_ext_df = _load_parquet(Path(args.x_ext))
    labels_df = _load_parquet(Path(args.labels_ext))

    # Optional calibration + thresholds
    cal_params = load_calibration_params(Path(args.calibration_params)) if args.calibration_params else {}
    thresholds = json.loads(Path(args.thresholds_json).read_text()) if args.thresholds_json else None

    if args.baseline_model_path:
        if not args.label_col:
            raise ValueError("--label-col is required for baseline mode.")
        payload = eval_baseline(
            model_path=Path(args.baseline_model_path),
            x_ext_df=x_ext_df,
            labels_df=labels_df,
            label_col=args.label_col,
            cal_params=cal_params,
            ece_bins=int(args.ece_bins),
            thresholds=thresholds,
            outdir=outdir
        )
    else:
        if not args.task_cols:
            raise ValueError("--task-cols is required for MTL mode.")
        task_cols = [s.strip() for s in args.task_cols.split(",") if s.strip()]
        payload = eval_mtl(
            run_dir=Path(args.mtl_run_dir),
            x_ext_df=x_ext_df,
            labels_df=labels_df,
            task_cols=task_cols,
            cal_params=cal_params,
            ece_bins=int(args.ece_bins),
            thresholds=thresholds,
            outdir=outdir
        )

    # Console summary
    print(json.dumps(payload, indent=2))
    print(f"Wrote external validation outputs to: {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# In[ ]:




