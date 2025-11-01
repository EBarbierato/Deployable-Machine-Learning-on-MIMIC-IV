#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python3
"""
15_robustness_checks.py

Purpose
-------
Run robustness checks for trained models on the TEST set:
  - Feature ablation (zero/mean) by explicit list or by top-k from explain_global.csv (script 10)
  - Noise sensitivity (Gaussian noise scaled by feature std)
  - Temporal drift (evaluate metrics in time slices by year/quarter/month from a meta date column)
  - Alternative labels (evaluate metrics against alternative label columns)

Supports:
  - Baselines (06): Logistic / RF / XGBoost via model.joblib
  - MTL (07/08): Shared-Bottom / Cross-Stitch / MMoE via run dir with best.ckpt

Optionally applies calibration from 09 (Platt/Isotonic) and evaluates operating thresholds from 09.

Inputs
------
Required TEST features & labels:
  --x-test           Parquet with TEST features (must include 'subject_id')
  --labels           Parquet with labels (must include 'subject_id' and target col(s))

Choose ONE of:
  --baseline-model-path  Path to 06 model.joblib   (use with --label-col)
  --mtl-run-dir          Dir from 07/08 with config.json, feature_names.json, best.ckpt (use with --task-cols)

If baseline:
  --label-col            Name of primary label column (e.g., diab_incident_365d)

If MTL:
  --task-cols            Comma-separated list of two label columns

Optional calibration & thresholds:
  --calibration-params   Path to calibration_params.json from 09 (to apply Platt/Isotonic)
  --thresholds-json      Path to thresholds.json from 09 (evaluate operating points)

Robustness options (any subset):
  --ablate-features      Comma-separated feature names to ablate
  --ablate-from-explain  Path to explain_global.csv (script 10) to pull top-k features
  --ablate-topk          Integer top-k features to ablate from explain_global.csv (default 0=off)
  --ablate-strategy      zero | mean  (default: mean)

  --noise-levels         Comma floats for σ multipliers vs feature std (e.g., 0.1,0.2)
  --noise-seed           RNG seed for noise (default 123)

  --meta-file            Parquet with meta columns for slicing (must include 'subject_id' and --time-col if slicing)
  --time-col             Name of datetime column in meta-file (e.g., index_time)
  --slice-by             year | quarter | month (default: none)

  --alt-label-col        (Baseline) additional label column to evaluate alongside primary
  --alt-task-cols        (MTL) comma list of two alt task columns (same order as --task-cols)

General:
  --ece-bins             Bins for ECE (default 15)
  --outdir               Output directory (default artifacts/robustness)

Examples
--------
# Baseline with feature ablation and noise
python scripts/15_robustness_checks.py \
  --x-test artifacts/imputed/X_test.parquet \
  --labels artifacts/labels/labels.parquet \
  --baseline-model-path artifacts/baselines/diab_incident_365d/logistic/model.joblib \
  --label-col diab_incident_365d \
  --ablate-from-explain artifacts/explain/diab_logistic/explain_global.csv \
  --ablate-topk 20 --ablate-strategy mean \
  --noise-levels 0.1,0.2 \
  --calibration-params artifacts/calibration/diab_logistic/calibration_params.json \
  --thresholds-json artifacts/calibration/diab_logistic/thresholds.json \
  --outdir artifacts/robustness/diab_logistic

# MTL with time slices by year and alt labels
python scripts/15_robustness_checks.py \
  --x-test artifacts/imputed/X_test.parquet \
  --labels artifacts/labels/labels.parquet \
  --mtl-run-dir artifacts/mtl_shared_bottom \
  --task-cols diab_incident_365d,cvd_incident_365d \
  --meta-file artifacts/cohort/cohort.parquet \
  --time-col index_time --slice-by year \
  --outdir artifacts/robustness/mtl_shared_bottom
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Optional dependencies (baseline/MTL)
import joblib
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
        description="Robustness checks: feature ablation, noise sensitivity, temporal drift, alt labels.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--x-test", type=str, required=True)
    p.add_argument("--labels", type=str, required=True)

    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--baseline-model-path", type=str)
    grp.add_argument("--mtl-run-dir", type=str)

    p.add_argument("--label-col", type=str, help="(Baseline) primary label column")
    p.add_argument("--task-cols", type=str, help="(MTL) two task label columns, comma-separated")

    p.add_argument("--calibration-params", type=str, default=None)
    p.add_argument("--thresholds-json", type=str, default=None)
    p.add_argument("--ece-bins", type=int, default=15)

    # Ablation
    p.add_argument("--ablate-features", type=str, default=None)
    p.add_argument("--ablate-from-explain", type=str, default=None)
    p.add_argument("--ablate-topk", type=int, default=0)
    p.add_argument("--ablate-strategy", type=str, choices=["zero", "mean"], default="mean")

    # Noise
    p.add_argument("--noise-levels", type=str, default=None, help="Comma floats (σ multipliers vs std)")
    p.add_argument("--noise-seed", type=int, default=123)

    # Time slices
    p.add_argument("--meta-file", type=str, default=None)
    p.add_argument("--time-col", type=str, default=None)
    p.add_argument("--slice-by", type=str, choices=["year", "quarter", "month"], default=None)

    # Alt labels
    p.add_argument("--alt-label-col", type=str, default=None, help="(Baseline) evaluate also this label")
    p.add_argument("--alt-task-cols", type=str, default=None, help="(MTL) two alt task cols for horizon sensitivity")

    p.add_argument("--outdir", type=str, default="artifacts/robustness")
    return p.parse_args()


# ----------------------------- IO helpers ----------------------------------- #

def _load_parquet(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "subject_id" not in df.columns:
        raise KeyError(f"{path} must include 'subject_id'.")
    return df

def _extract_label(labels: pd.DataFrame, col: str) -> pd.DataFrame:
    if col not in labels.columns:
        raise KeyError(f"labels missing '{col}'.")
    out = labels[["subject_id", col]].drop_duplicates("subject_id").copy()
    out[col] = out[col].astype(bool).astype(int)
    return out

def _extract_labels_multi(labels: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c not in labels.columns:
            raise KeyError(f"labels missing '{c}'.")
    out = labels[["subject_id"] + cols].drop_duplicates("subject_id").copy()
    for c in cols:
        out[c] = out[c].astype(bool).astype(int)
    return out

def _align_features(X_df: pd.DataFrame, feature_names: List[str]) -> np.ndarray:
    missing = [c for c in feature_names if c not in X_df.columns]
    if missing:
        raise KeyError(f"Features missing {len(missing)} required columns (e.g., {missing[:8]}).")
    return X_df[feature_names].to_numpy(dtype=np.float32, copy=True)


# ----------------------------- Metrics & eval -------------------------------- #

def _metrics(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss
    eps = 1e-12
    y = y.astype(int); p = np.clip(p.astype(float), eps, 1-eps)
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
    total = len(p); ece = 0.0
    for b in range(n_bins):
        m = inds == b
        if np.sum(m) == 0: 
            continue
        conf = float(np.mean(p[m])); acc = float(np.mean(y[m]))
        ece += (np.sum(m) / total) * abs(acc - conf)
    return float(ece)

def _confusion(y: np.ndarray, p: np.ndarray, t: float) -> Dict[str, float]:
    y = y.astype(int); p = p.astype(float)
    pred = (p >= t).astype(int)
    tp = int(np.sum((pred==1) & (y==1)))
    fp = int(np.sum((pred==1) & (y==0)))
    tn = int(np.sum((pred==0) & (y==0)))
    fn = int(np.sum((pred==0) & (y==1)))
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
    x = np.array(prm.x, dtype=float); y = np.array(prm.y, dtype=float)
    idx = np.searchsorted(x, p, side="right") - 1
    idx = np.clip(idx, 0, len(y) - 1)
    return y[idx]

def _load_cal_params(path: Optional[Path]) -> Dict[str, object]:
    if not path:
        return {}
    meta = json.loads(path.read_text())
    out = {}
    prm = meta.get("params", {})
    if "platt" in prm:
        out["platt"] = PlattParams(A=float(prm["platt"]["A"]), B=float(prm["platt"]["B"]), eps=float(prm["platt"].get("eps", 1e-6)))
    if "isotonic" in prm:
        out["isotonic"] = IsoParams(x=list(prm["isotonic"]["x"]), y=list(prm["isotonic"]["y"]))
    return out


# ----------------------------- Models (MTL) ---------------------------------- #

class SharedBottom(nn.Module):  # type: ignore[misc]
    def __init__(self, d_in: int, hidden: List[int], dropout: float):
        super().__init__()
        layers = []; last = d_in
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU(), nn.Dropout(dropout)]
            last = h
        self.shared = nn.Sequential(*layers) if layers else nn.Identity()
        self.head1 = nn.Linear(last, 1); self.head2 = nn.Linear(last, 1)
    def forward(self, x: torch.Tensor):
        z = self.shared(x); return self.head1(z).squeeze(-1), self.head2(z).squeeze(-1)

class CrossStitchUnit(nn.Module):  # type: ignore[misc]
    def __init__(self, a: float=0.9):
        super().__init__(); A = torch.tensor([[a, 1-a],[1-a, a]], dtype=torch.float32); self.A = nn.Parameter(A)
    def forward(self, x1, x2): A=self.A; return A[0,0]*x1 + A[0,1]*x2, A[1,0]*x1 + A[1,1]*x2

class CrossStitchNet(nn.Module):  # type: ignore[misc]
    def __init__(self, d_in:int, tower_sizes:List[int], dropout:float, stitch_alpha:float):
        super().__init__(); sizes=[d_in]+tower_sizes
        self.l1=nn.ModuleList(); self.l2=nn.ModuleList(); self.cs=nn.ModuleList()
        for i in range(1,len(sizes)):
            self.l1.append(nn.Sequential(nn.Linear(sizes[i-1],sizes[i]),nn.ReLU(),nn.Dropout(dropout)))
            self.l2.append(nn.Sequential(nn.Linear(sizes[i-1],sizes[i]),nn.ReLU(),nn.Dropout(dropout)))
            self.cs.append(CrossStitchUnit(a=stitch_alpha))
        last=sizes[-1]; self.h1=nn.Linear(last,1); self.h2=nn.Linear(last,1)
    def forward(self,x):
        z1=x; z2=x
        for a,b,c in zip(self.l1,self.l2,self.cs): z1=a(z1); z2=b(z2); z1,z2=c(z1,z2)
        return self.h1(z1).squeeze(-1), self.h2(z2).squeeze(-1)

class ExpertMLP(nn.Module):  # type: ignore[misc]
    def __init__(self,d_in:int,d_out:int,dropout:float):
        super().__init__(); self.net=nn.Sequential(nn.Linear(d_in,d_out),nn.ReLU(),nn.Dropout(dropout),
                                                  nn.Linear(d_out,d_out),nn.ReLU(),nn.Dropout(dropout))
    def forward(self,x): return self.net(x)

class Gate(nn.Module):  # type: ignore[misc]
    def __init__(self,d_in:int,k:int,hidden:int):
        super().__init__(); self.net=nn.Sequential(nn.Linear(d_in,hidden),nn.ReLU(),nn.Linear(hidden,k)); self.sm=nn.Softmax(dim=-1)
    def forward(self,x): return self.sm(self.net(x))

class MMoE(nn.Module):  # type: ignore[misc]
    def __init__(self,d_in:int,k:int,expert_size:int,tower:List[int],gate_hidden:int,dropout:float):
        super().__init__(); self.experts=nn.ModuleList([ExpertMLP(d_in,expert_size,dropout) for _ in range(k)])
        self.g1=Gate(d_in,k,gate_hidden); self.g2=Gate(d_in,k,gate_hidden)
        def tw(d0:int,s:List[int]):
            layers=[]; last=d0
            for h in s: layers+=[nn.Linear(last,h),nn.ReLU(),nn.Dropout(dropout)]; last=h
            return nn.Sequential(*layers) if layers else nn.Identity()
        self.t1=tw(expert_size,tower); self.t2=tw(expert_size,tower)
        self.h1=nn.Linear(tower[-1] if tower else expert_size,1); self.h2=nn.Linear(tower[-1] if tower else expert_size,1)
    def forward(self,x):
        E=torch.stack([e(x) for e in self.experts],dim=1); m1=torch.sum(self.g1(x).unsqueeze(-1)*E,dim=1); m2=torch.sum(self.g2(x).unsqueeze(-1)*E,dim=1)
        z1=self.t1(m1); z2=self.t2(m2); return self.h1(z1).squeeze(-1), self.h2(z2).squeeze(-1)

def _build_mtl(cfg: dict, d_in: int) -> nn.Module:
    v = cfg.get("variant", "shared_bottom"); drop = float(cfg.get("dropout", 0.2))
    if v == "shared_bottom":
        hidden = [int(x) for x in cfg.get("hidden_sizes", [256,128])]
        return SharedBottom(d_in, hidden, drop)
    if v == "cross_stitch":
        tower = [int(x) for x in cfg.get("tower_sizes", [256,128])]
        return CrossStitchNet(d_in, tower, drop, float(cfg.get("stitch_init_alpha", 0.9)))
    if v == "mmoe":
        return MMoE(d_in, int(cfg.get("experts",4)), int(cfg.get("expert_size",128)),
                    [int(x) for x in cfg.get("tower_sizes",[64])], int(cfg.get("gate_hidden",64)), drop)
    # default
    hidden = [int(x) for x in cfg.get("hidden_sizes", [256,128])]
    return SharedBottom(d_in, hidden, drop)


# ----------------------------- Scenario builders ----------------------------- #

def pick_topk_from_explain(path: Path, k: int) -> List[str]:
    df = None
    if path.suffix.lower() == ".parquet":
        try: df = pd.read_parquet(path)
        except Exception: df = None
    if df is None:
        df = pd.read_csv(path)
    # Ranking preference: shap_mean_abs > perm_importance_mean > |coef/value|
    for col in ["shap_mean_abs", "perm_importance_mean", "value_abs", "value"]:
        if col in df.columns:
            rank_col = col
            break
    else:
        raise ValueError("explain_global missing ranking columns (shap_mean_abs / perm_importance_mean / value[_abs]).")
    if "value_abs" not in df.columns and "value" in df.columns:
        df["value_abs"] = df["value"].abs()
    sub = df[["feature", rank_col]].dropna().sort_values(rank_col, ascending=False).head(k)
    return sub["feature"].astype(str).tolist()

def build_scenarios(
    X_test: pd.DataFrame,
    meta: Optional[pd.DataFrame],
    args: argparse.Namespace
) -> Dict[str, Dict[str, object]]:
    """
    Returns a dict: scenario_name -> spec dict
    Spec fields:
      - type: base | ablate | noise | time_slice | alt_label
      - features: list[str] (for ablate)
      - strategy: zero|mean (for ablate)
      - sigma: float (for noise)
      - time_mask: pd.Index (subject_id index to keep) (for time_slice)
      - label_override: str or List[str] (for alt label(s))
    """
    scenarios: Dict[str, Dict[str, object]] = {}

    # base
    scenarios["base"] = {"type": "base"}

    # ablation
    feats_to_ablate: List[str] = []
    if args.ablate_features:
        feats_to_ablate += [s.strip() for s in args.ablate_features.split(",") if s.strip()]
    if args.ablate_from_explain and args.ablate_topk and args.ablate_topk > 0:
        feats_to_ablate += pick_topk_from_explain(Path(args.ablate_from_explain), k=int(args.ablate_topk))
    feats_to_ablate = [f for f in feats_to_ablate if f in X_test.columns and f != "subject_id"]
    if feats_to_ablate:
        scenarios[f"ablate_{args.ablate_strategy}_{len(feats_to_ablate)}"] = {
            "type": "ablate", "features": feats_to_ablate, "strategy": args.ablate_strategy
        }

    # noise
    if args.noise_levels:
        levels = [float(s) for s in [x.strip() for x in args.noise_levels.split(",") if x.strip()]]
        for sig in levels:
            scenarios[f"noise_{sig:g}"] = {"type": "noise", "sigma": float(sig), "seed": int(args.noise_seed)}

    # time slices
    if args.meta_file and args.time_col and args.slice_by:
        if meta is None:
            raise ValueError("meta-file not loaded.")
        if args.time_col not in meta.columns:
            raise KeyError(f"time column '{args.time_col}' not found in meta-file.")
        dt = pd.to_datetime(meta[args.time_col], errors="coerce")
        lab = None
        if args.slice_by == "year":
            lab = dt.dt.year
        elif args.slice_by == "quarter":
            lab = dt.dt.to_period("Q").astype(str)
        elif args.slice_by == "month":
            lab = dt.dt.to_period("M").astype(str)
        if lab is not None:
            mdf = pd.DataFrame({"subject_id": meta["subject_id"].values, "slice": lab})
            for val, sub in mdf.groupby("slice", dropna=True):
                keep = sub["subject_id"].values
                if keep.size == 0:
                    continue
                scenarios[f"time_{args.slice_by}_{val}"] = {"type": "time_slice", "keep_ids": keep.tolist(),
                                                            "slice_value": str(val)}

    # alt labels
    if args.alt_label_col:
        scenarios[f"alt_label_{args.alt_label_col}"] = {"type": "alt_label", "label_override": args.alt_label_col}
    if args.alt_task_cols:
        alt = [s.strip() for s in args.alt_task_cols.split(",") if s.strip()]
        if len(alt) == 2:
            scenarios[f"alt_tasks_{alt[0]}_{alt[1]}"] = {"type": "alt_label", "label_override": alt}

    return scenarios


# ----------------------------- Eval runners ---------------------------------- #

def apply_scenario_matrix(X: pd.DataFrame, spec: Dict[str, object]) -> pd.DataFrame:
    """Return a modified copy of X according to the scenario."""
    Xmod = X.copy()
    if spec["type"] == "ablate":
        feats: List[str] = spec["features"]  # type: ignore
        if spec.get("strategy", "mean") == "zero":
            for f in feats:
                if f in Xmod.columns:
                    Xmod[f] = 0.0
        else:
            # mean
            means = Xmod[feats].mean(numeric_only=True)
            for f in feats:
                if f in Xmod.columns:
                    Xmod[f] = float(means.get(f, 0.0))
    elif spec["type"] == "noise":
        sigma = float(spec.get("sigma", 0.0))
        if sigma > 0:
            rng = np.random.default_rng(int(spec.get("seed", 123)))
            num_cols = [c for c in Xmod.columns if c != "subject_id" and pd.api.types.is_numeric_dtype(Xmod[c])]
            std = Xmod[num_cols].std(ddof=0).replace(0.0, 1.0)
            noise = rng.normal(loc=0.0, scale=1.0, size=Xmod[num_cols].shape).astype(np.float32)
            Xmod[num_cols] = Xmod[num_cols].values + (sigma * std.values).astype(np.float32) * noise
    elif spec["type"] == "time_slice":
        keep = set(spec["keep_ids"])  # type: ignore
        Xmod = Xmod[Xmod["subject_id"].isin(keep)].copy()
    # base and alt_label do not modify X
    return Xmod

def variants_from_cal(scores_raw: np.ndarray, cal_params: Dict[str, object]) -> Dict[str, np.ndarray]:
    out = {"raw": scores_raw}
    if "platt" in cal_params:
        out["platt"] = _apply_platt(scores_raw, cal_params["platt"])  # type: ignore
    if "isotonic" in cal_params:
        out["isotonic"] = _apply_iso(scores_raw, cal_params["isotonic"])  # type: ignore
    return out

def eval_baseline_scenarios(
    model_path: Path,
    X_test: pd.DataFrame,
    labels_df: pd.DataFrame,
    label_col: str,
    scenarios: Dict[str, Dict[str, object]],
    cal_params: Dict[str, object],
    thresholds: Optional[dict],
    ece_bins: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Load model and determine feature order from X_test columns
    model = joblib.load(model_path)
    feat_names = [c for c in X_test.columns if c != "subject_id"]

    results_rows = []
    thresh_rows = []

    for sname, spec in scenarios.items():
        # Prepare X and y for this scenario
        Xs = apply_scenario_matrix(X_test, spec)
        ymap = _extract_label(labels_df, spec.get("label_override", label_col))  # type: ignore
        df = Xs[["subject_id"]].merge(ymap, on="subject_id", how="inner")
        if df.empty:
            continue
        Xmat = _align_features(Xs, feat_names)
        y = df[spec.get("label_override", label_col)].astype(int).values  # type: ignore

        # Predict + calibrate
        p_raw = model.predict_proba(Xmat)[:, 1]
        vmap = variants_from_cal(p_raw, cal_params if cal_params else {})

        # Metrics
        for vname, ps in vmap.items():
            m = _metrics(y, ps); m["ece"] = _ece(y, ps, n_bins=ece_bins)
            results_rows.append({"scenario": sname, "variant": vname, "label": spec.get("label_override", label_col), **m})  # type: ignore

        # Thresholds
        if thresholds:
            thmap = thresholds.get("thresholds", {})
            for vname, rules in thmap.items():
                if vname not in vmap:
                    continue
                for rule, thr in rules.items():
                    d = _confusion(y, vmap[vname], float(thr))
                    thresh_rows.append({"scenario": sname, "variant": vname, "rule": rule, "label": spec.get("label_override", label_col), **d})  # type: ignore

    return pd.DataFrame(results_rows), pd.DataFrame(thresh_rows)

def eval_mtl_scenarios(
    run_dir: Path,
    X_test: pd.DataFrame,
    labels_df: pd.DataFrame,
    task_cols: List[str],
    scenarios: Dict[str, Dict[str, object]],
    cal_params: Dict[str, object],
    thresholds: Optional[dict],
    ece_bins: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not _HAVE_TORCH:
        raise RuntimeError("PyTorch is required for MTL robustness checks.")
    # Load model
    cfg = json.loads((run_dir / "config.json").read_text())
    feat_names = json.loads((run_dir / "feature_names.json").read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_mtl(cfg, d_in=len(feat_names)).to(device)
    ckpt = torch.load(run_dir / "best.ckpt", map_location=device)
    model.load_state_dict(ckpt["model_state"]); model.eval()

    results_rows = []
    thresh_rows = []

    for sname, spec in scenarios.items():
        Xs = apply_scenario_matrix(X_test, spec)
        ymap = _extract_labels_multi(labels_df, spec.get("label_override", task_cols))  # type: ignore
        df = Xs[["subject_id"]].merge(ymap, on="subject_id", how="inner")
        if df.empty:
            continue

        Xmat = _align_features(Xs, feat_names)
        with torch.no_grad():
            xb = torch.from_numpy(Xmat).to(device).float()
            l1, l2 = model(xb)
            p1 = torch.sigmoid(l1).cpu().numpy()
            p2 = torch.sigmoid(l2).cpu().numpy()
        raw_map = {task_cols[0]: p1, task_cols[1]: p2}
        Y = df[spec.get("label_override", task_cols)].astype(int).to_numpy(copy=True)  # type: ignore

        for i, task in enumerate(task_cols):
            y = Y[:, i]
            vmap = variants_from_cal(raw_map[task], cal_params if cal_params else {})
            for vname, ps in vmap.items():
                m = _metrics(y, ps); m["ece"] = _ece(y, ps, n_bins=ece_bins)
                results_rows.append({"scenario": sname, "variant": vname, "label": task, **m})
            if thresholds:
                thmap = thresholds.get("thresholds", {})
                for vname, rules in thmap.items():
                    if vname not in vmap:
                        continue
                    for rule, thr in rules.items():
                        d = _confusion(y, vmap[vname], float(thr))
                        thresh_rows.append({"scenario": sname, "variant": vname, "rule": rule, "label": task, **d})

    return pd.DataFrame(results_rows), pd.DataFrame(thresh_rows)


# ----------------------------- MAIN ----------------------------------------- #

def main() -> int:
    args = parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    X_test = _load_parquet(Path(args.x_test))
    labels = _load_parquet(Path(args.labels))
    meta = _load_parquet(Path(args.meta_file)) if args.meta_file else None

    # Build scenarios
    scenarios = build_scenarios(X_test, meta, args)

    # Calibration & thresholds (optional)
    cal_params = _load_cal_params(Path(args.calibration_params)) if args.calibration_params else {}
    thresholds = json.loads(Path(args.thresholds_json).read_text()) if args.thresholds_json else None

    # Run
    if args.baseline_model_path:
        if not args.label_col:
            raise ValueError("--label-col is required for baseline mode.")
        res_df, thr_df = eval_baseline_scenarios(
            model_path=Path(args.baseline_model_path),
            X_test=X_test, labels_df=labels, label_col=args.label_col,
            scenarios=scenarios, cal_params=cal_params, thresholds=thresholds, ece_bins=int(args.ece_bins)
        )
    else:
        if not args.task_cols:
            raise ValueError("--task-cols is required for MTL mode.")
        task_cols = [s.strip() for s in args.task_cols.split(",") if s.strip()]
        if len(task_cols) != 2:
            raise ValueError("--task-cols must specify exactly two columns.")
        res_df, thr_df = eval_mtl_scenarios(
            run_dir=Path(args.mtl_run_dir),
            X_test=X_test, labels_df=labels, task_cols=task_cols,
            scenarios=scenarios, cal_params=cal_params, thresholds=thresholds, ece_bins=int(args.ece_bins)
        )

    # Persist
    if not res_df.empty:
        res_df.to_csv(outdir / "robustness_results.csv", index=False)
    if not thr_df.empty:
        thr_df.to_csv(outdir / "threshold_results.csv", index=False)

    # Scenario manifest
    manifest = {
        "generated_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "scenarios": scenarios
    }
    (outdir / "SCENARIOS.json").write_text(json.dumps(manifest, indent=2))

    # Summary
    lines: List[str] = []
    lines.append(f"# Robustness Checks Summary\n\nGenerated: {datetime.utcnow().isoformat(timespec='seconds')}Z\n\n")
    lines.append(f"- X_test: `{args.x_test}`\n- Labels: `{args.labels}`\n")
    if args.baseline_model_path:
        lines.append(f"- Model: baseline ({args.baseline_model_path}) | Label: {args.label_col}\n")
    else:
        lines.append(f"- Model: MTL ({args.mtl_run_dir}) | Tasks: {args.task_cols}\n")
    if args.calibration-params:
        pass  # placeholder to avoid linter; value shown below
    if args.calibration_params:
        lines.append(f"- Calibration: `{args.calibration_params}`\n")
    if args.thresholds_json:
        lines.append(f"- Thresholds: `{args.thresholds_json}`\n")
    # Scenario list
    lines.append("\n## Scenarios\n")
    for k, v in scenarios.items():
        if v["type"] == "ablate":
            lines.append(f"- **{k}**: ablate {len(v['features'])} features ({v['strategy']})\n")
        elif v["type"] == "noise":
            lines.append(f"- **{k}**: Gaussian noise σ={v['sigma']}\n")
        elif v["type"] == "time_slice":
            lines.append(f"- **{k}**: time slice = {v['slice_value']}\n")
        elif v["type"] == "alt_label":
            lines.append(f"- **{k}**: alternate label(s) = {v['label_override']}\n")
        else:
            lines.append(f"- **{k}**: base\n")
    if not res_df.empty:
        lines.append("\n## Output files\n- `robustness_results.csv`\n")
    if not thr_df.empty:
        lines.append("- `threshold_results.csv`\n")
    (outdir / "SUMMARY.md").write_text("".join(lines))

    print(f"Wrote robustness outputs to: {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

