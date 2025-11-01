#!/usr/bin/env python3
# 09_calibrate_and_thresholds.py
#
# Purpose
# -------
# Post-process model predictions:
#   1) Calibrate probabilities using VAL set (Platt sigmoid and/or Isotonic).
#   2) Apply the chosen calibrators to VAL and TEST.
#   3) Select decision thresholds on VAL (F1-max, Youden's J, fixed-sensitivity targets, match-prevalence).
#   4) Evaluate metrics and confusion tables on TEST at those thresholds.
#
# Inputs
# ------
# - --val-preds  : Path to validation predictions parquet
# - --test-preds : Path to test predictions parquet
#   Supported schemas:
#     (A) Baselines (script 06): columns = [subject_id, y_true, y_score]
#     (B) MTL (scripts 07/08):   columns = [subject_id, <task>_y_true, <task>_y_score]
#         In this case you must pass --task-col <task> (e.g., diab_incident_365d)
#
# Outputs (under {outdir}/)
# -------------------------
# - calibrated_val.parquet      # subject_id, y_true, y_score_raw, y_score_platt?, y_score_iso?
# - calibrated_test.parquet     # same columns as above
# - calibration_params.json     # exact parameters for Platt (A,B) and Isotonic (x,y knots), eps used
# - thresholds.json             # selected thresholds (per rule) chosen on VAL, applied to TEST
# - METRICS.md                  # human-readable summary
# - metrics.json                # full metrics pre/post calibration + thresholded results on TEST
#
# Examples
# --------
# # Calibrate a baseline model (val/test preds were written by 06_train_baselines.py)
# python scripts/09_calibrate_and_thresholds.py \
#   --val-preds artifacts/baselines/diab_incident_365d/logistic/val_preds.parquet \
#   --test-preds artifacts/baselines/diab_incident_365d/logistic/test_preds.parquet \
#   --methods platt,isotonic \
#   --fixed-sens 0.8,0.9 \
#   --outdir artifacts/calibration/diab_logistic
#
# # Calibrate an MTL head from 07/08 outputs
# python scripts/09_calibrate_and_thresholds.py \
#   --val-preds artifacts/mtl_shared_bottom/VAL_preds.parquet \
#   --test-preds artifacts/mtl_shared_bottom/TEST_preds.parquet \
#   --task-col diab_incident_365d \
#   --methods platt,isotonic \
#   --outdir artifacts/calibration/mtl_shared_bottom_diab

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss, log_loss
)

# ----------------------------- CLI ------------------------------------------ #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fit VAL-set calibration and choose thresholds; evaluate on TEST.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--val-preds", type=str, required=True, help="Path to validation predictions parquet.")
    p.add_argument("--test-preds", type=str, required=True, help="Path to test predictions parquet.")
    p.add_argument("--task-col", type=str, default=None,
                   help="If MTL-style preds, the task prefix (e.g., 'diab_incident_365d').")
    p.add_argument("--methods", type=str, default="platt,isotonic",
                   help="Comma-separated subset of: platt,isotonic,none")
    p.add_argument("--fixed-sens", type=str, default="0.8,0.9",
                   help="Comma-separated sensitivity targets for threshold selection (on VAL).")
    p.add_argument("--ece-bins", type=int, default=15, help="Bins for Expected Calibration Error.")
    p.add_argument("--outdir", type=str, default="artifacts/calibration")
    return p.parse_args()


# ----------------------------- Utilities ------------------------------------ #

def _extract_columns(df: pd.DataFrame, task_col: str | None) -> Tuple[pd.Series, pd.Series, pd.Series]:
    if task_col is None:
        # baseline schema
        need = {"subject_id", "y_true", "y_score"}
        missing = need - set(df.columns)
        if missing:
            raise KeyError(f"Missing columns for baseline schema: {missing}")
        return df["subject_id"], df["y_true"].astype(int), df["y_score"].astype(float)
    else:
        yt = f"{task_col}_y_true"
        ys = f"{task_col}_y_score"
        need = {"subject_id", yt, ys}
        missing = need - set(df.columns)
        if missing:
            raise KeyError(f"Missing columns for task '{task_col}': {missing}")
        return df["subject_id"], df[yt].astype(int), df[ys].astype(float)

def _clip_probs(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return np.clip(p, eps, 1.0 - eps)

def _logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = _clip_probs(p, eps)
    return np.log(p / (1.0 - p))

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

def _metrics(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    y = y.astype(int); p = _clip_probs(p)
    out = {}
    try: out["auroc"]  = float(roc_auc_score(y, p))
    except ValueError: out["auroc"] = float("nan")
    try: out["auprc"]  = float(average_precision_score(y, p))
    except ValueError: out["auprc"] = float("nan")
    out["brier"] = float(brier_score_loss(y, p))
    try: out["logloss"] = float(log_loss(y, p))
    except ValueError: out["logloss"] = float("nan")
    out["pos_rate"] = float(np.mean(y))
    return out

def _ece(y: np.ndarray, p: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error."""
    y = y.astype(int); p = _clip_probs(p)
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

def _select_thresholds_on_val(y_val: np.ndarray, p_val: np.ndarray, fixed_sens: List[float]) -> Dict[str, float]:
    # Sweep unique scores for robust selection
    scores = np.unique(p_val)
    # Safety: include 0/1 bounds
    scores = np.unique(np.concatenate([scores, np.array([0.0, 1.0])]))
    # Compute stats for each threshold
    stats = [_confusion_at_thresh(y_val, p_val, t) for t in scores]
    # F1-max
    t_f1 = max(stats, key=lambda d: (d["f1"], d["sensitivity"]))["threshold"]
    # Youden J
    t_j = max(stats, key=lambda d: (d["sensitivity"] + d["specificity"] - 1.0))["threshold"]
    # Match prevalence: choose threshold where predicted positive rate ~= empirical prevalence
    prev = float(np.mean(y_val))
    t_prev = min(stats, key=lambda d: abs((d["tp"] + d["fp"]) / max(1, d["tp"] + d["fp"] + d["tn"] + d["fn"]) - prev))["threshold"]
    # Fixed sensitivity targets
    fixed = {}
    for s in fixed_sens:
        # among thresholds with sens >= target, choose highest specificity; tiebreak on higher threshold
        cand = [d for d in stats if d["sensitivity"] >= s]
        if cand:
            chosen = max(cand, key=lambda d: (d["specificity"], d["threshold"]))
            fixed[f"fixed_sens_{int(round(100*s))}"] = float(chosen["threshold"])
        else:
            fixed[f"fixed_sens_{int(round(100*s))}"] = float(1.0)  # degenerate: classify none positive
    return {"f1_max": float(t_f1), "youden_j": float(t_j), "match_prevalence": float(t_prev), **fixed}

# ----------------------------- Calibration fits ----------------------------- #

@dataclass
class PlattParams:
    A: float  # multiplier on logit
    B: float  # intercept in logit space
    eps: float = 1e-6

def fit_platt(y: np.ndarray, p: np.ndarray, eps: float = 1e-6) -> PlattParams:
    # Fit logistic regression on logit(p) with intercept: sigma(B + A*logit(p))
    x = _logit(p, eps=eps)
    w = np.zeros(2, dtype=float)  # [B, A]
    for _ in range(50):
        z = w[0] + w[1] * x
        s = _sigmoid(z)
        g0 = np.sum(s - y)
        g1 = np.sum((s - y) * x)
        W = s * (1 - s)
        H00 = np.sum(W)
        H01 = np.sum(W * x)
        H11 = np.sum(W * x * x)
        det = H00 * H11 - H01 * H01
        if det <= 1e-12:
            break
        dB = -( H11 * g0 - H01 * g1) / det
        dA = -(-H01 * g0 + H00 * g1) / det
        w += np.array([dB, dA])
        if max(abs(dB), abs(dA)) < 1e-6:
            break
    return PlattParams(A=float(w[1]), B=float(w[0]), eps=float(eps))

def apply_platt(p: np.ndarray, prm: PlattParams) -> np.ndarray:
    z = prm.B + prm.A * _logit(p, eps=prm.eps)
    return _sigmoid(z)

@dataclass
class IsoParams:
    x: List[float]
    y: List[float]

def fit_isotonic(y: np.ndarray, p: np.ndarray) -> IsoParams:
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(p.astype(float), y.astype(int))
    return IsoParams(x=list(map(float, ir.X_thresholds_)), y=list(map(float, ir.y_thresholds_)))

def apply_isotonic(p: np.ndarray, prm: IsoParams) -> np.ndarray:
    x = np.array(prm.x, dtype=float)
    y = np.array(prm.y, dtype=float)
    idx = np.searchsorted(x, p, side="right") - 1
    idx = np.clip(idx, 0, len(y) - 1)
    return y[idx]

# ----------------------------- MAIN ----------------------------------------- #

def main() -> int:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    methods = [m.strip().lower() for m in args.methods.split(",") if m.strip()]
    methods = [m for m in methods if m in {"platt", "isotonic", "none"}]
    if not methods:
        raise ValueError("No valid methods requested; choose from platt,isotonic,none.")

    fixed_sens = [float(s) for s in [x.strip() for x in args.fixed_sens.split(",") if x.strip()]]
    fixed_sens = [s for s in fixed_sens if 0.0 < s < 1.0]

    # Load predictions
    val_df  = pd.read_parquet(args.val_preds)
    test_df = pd.read_parquet(args.test_preds)

    sid_v, y_v, p_v_raw = _extract_columns(val_df, args.task_col)
    sid_t, y_t, p_t_raw = _extract_columns(test_df, args.task_col)
    if len(set(sid_v)) != len(sid_v) or len(set(sid_t)) != len(sid_t):
        pass

    # Base metrics (uncalibrated)
    base_val = _metrics(y_v.values, p_v_raw.values) | {"ece": _ece(y_v.values, p_v_raw.values, n_bins=args.ece_bins)}
    base_tst = _metrics(y_t.values, p_t_raw.values) | {"ece": _ece(y_t.values, p_t_raw.values, n_bins=args.ece_bins)}

    # Fit calibrators on VAL
    params = {"epsilon": 1e-6, "fitted_on": "validation"}
    p_v = {"raw": p_v_raw.values.copy()}
    p_t = {"raw": p_t_raw.values.copy()}

    if "platt" in methods:
        platt = fit_platt(y_v.values.astype(int), p_v_raw.values.astype(float), eps=1e-6)
        p_v["platt"] = apply_platt(p_v_raw.values, platt)
        p_t["platt"] = apply_platt(p_t_raw.values, platt)
        params["platt"] = {"A": platt.A, "B": platt.B, "eps": platt.eps}

    if "isotonic" in methods:
        iso = fit_isotonic(y_v.values.astype(int), p_v_raw.values.astype(float))
        p_v["isotonic"] = apply_isotonic(p_v_raw.values, iso)
        p_t["isotonic"] = apply_isotonic(p_t_raw.values, iso)
        params["isotonic"] = {"x": iso.x, "y": iso.y}

    # Evaluate metrics per method
    metrics = {
        "validation": {"raw": base_val},
        "test": {"raw": base_tst},
    }
    for name in [m for m in methods if m != "none"]:
        metrics["validation"][name] = _metrics(y_v.values, p_v[name]) | {"ece": _ece(y_v.values, p_v[name], n_bins=args.ece_bins)}
        metrics["test"][name]       = _metrics(y_t.values, p_t[name]) | {"ece": _ece(y_t.values, p_t[name], n_bins=args.ece_bins)}

    # Choose thresholds on VAL for each available score variant
    chosen_thresholds: Dict[str, Dict[str, float]] = {}
    thres_tables_test: Dict[str, Dict[str, Dict[str, float]]] = {}
    for variant, pvec in p_v.items():
        ths = _select_thresholds_on_val(y_v.values, pvec, fixed_sens=fixed_sens)
        chosen_thresholds[variant] = ths
        thres_tables_test[variant] = {rule: _confusion_at_thresh(y_t.values, p_t[variant], t)
                                      for rule, t in ths.items()}

    # Write calibrated prediction files
    def write_calibrated(split: str, sids: pd.Series, y: pd.Series, pmap: Dict[str, np.ndarray]) -> None:
        df = pd.DataFrame({"subject_id": sids.values, "y_true": y.values.astype(int), "y_score_raw": pmap["raw"].astype(np.float32)})
        for k, v in pmap.items():
            if k == "raw":
                continue
            df[f"y_score_{k}"] = v.astype(np.float32)
        out_path = outdir / f"calibrated_{split}.parquet"
        if out_path.exists():
            out_path.unlink()
        df.to_parquet(out_path, index=False)

    write_calibrated("val",  sid_v, y_v, p_v)
    write_calibrated("test", sid_t, y_t, p_t)

    # Persist params, thresholds, and metrics
    (outdir / "calibration_params.json").write_text(json.dumps({
        "generated_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "params": params
    }, indent=2))

    (outdir / "thresholds.json").write_text(json.dumps({
        "selected_on": "validation",
        "thresholds": chosen_thresholds
    }, indent=2))

    (outdir / "metrics.json").write_text(json.dumps({
        "task_col": args.task_col,
        "ece_bins": args.ece_bins,
        "validation": metrics["validation"],
        "test": metrics["test"],
        "test_threshold_tables": thres_tables_test,
        "generated_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }, indent=2))

    # Markdown summary
    def fmt(m: Dict[str, float]) -> str:
        return (f"AUROC {m.get('auroc', float('nan')):.4f} | "
                f"AUPRC {m.get('auprc', float('nan')):.4f} | "
                f"Brier {m.get('brier', float('nan')):.4f} | "
                f"LogLoss {m.get('logloss', float('nan')):.4f} | "
                f"ECE {m.get('ece', float('nan')):.4f}")

    lines: List[str] = []
    lines.append(f"# Calibration & Thresholds Summary\n\nGenerated: {datetime.utcnow().isoformat(timespec='seconds')}Z\n\n")
    lines.append("## Validation metrics\n")
    lines.append(f"- Raw: {fmt(metrics['validation']['raw'])}\n")
    for k in [m for m in methods if m != "none"]:
        lines.append(f"- {k.title()}: {fmt(metrics['validation'][k])}\n")
    lines.append("\n## Test metrics\n")
    lines.append(f"- Raw: {fmt(metrics['test']['raw'])}\n")
    for k in [m for m in methods if m != "none"]:
        lines.append(f"- {k.title()}: {fmt(metrics['test'][k])}\n")
    lines.append("\n## Test confusion tables at VAL-selected thresholds\n")
    for variant, table in thres_tables_test.items():
        lines.append(f"### Variant: {variant}\n")
        for rule, d in table.items():
            lines.append(f"- {rule}: thr={d['threshold']:.4f} | "
                         f"sens={d['sensitivity']:.3f} spec={d['specificity']:.3f} "
                         f"ppv={d['ppv']:.3f} npv={d['npv']:.3f} f1={d['f1']:.3f} "
                         f"tp={d['tp']} fp={d['fp']} tn={d['tn']} fn={d['fn']}\n")
    (outdir / "METRICS.md").write_text("".join(lines))

    print("Validation (raw):", fmt(metrics["validation"]["raw"]))
    for k in [m for m in methods if m != "none"]:
        print(f"Validation ({k}):", fmt(metrics["validation"][k]))
    print("Test (raw):", fmt(metrics["test"]["raw"]))
    for k in [m for m in methods if m != "none"]:
        print(f"Test ({k}):", fmt(metrics["test"][k]))
    print(f"Wrote calibrated preds, params, thresholds, and metrics to: {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
