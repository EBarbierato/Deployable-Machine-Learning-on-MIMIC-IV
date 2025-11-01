#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[ ]:


#!/usr/bin/env python3
"""


Purpose
-------
Assemble a reproducible, paper-ready bundle from prior pipeline artifacts:
- Tables: cohort summary (optional), model metrics (VAL/TEST), thresholds, top features.
- Figures: PR and ROC curves (TEST), reliability diagrams (TEST), feature-importance bars.
- README.md: quick narrative and file map for manuscript integration.

Inputs (flexible; provide what you have)
---------------------------------------
--cohort-file            Parquet from 01_extract_cohort.py (optional; for cohort summary)
--labels-file            Parquet from 02_define_labels.py (optional; only to report available labels)
--baselines-root         Root of 06_train_baselines.py outputs (will scan */*/metrics.json, preds)
--mtl-dirs               Comma-separated dirs from 07/08 (each must contain TEST/VAL preds + metrics.json)
--calibration-dirs       Comma-separated dirs from 09 (each contains metrics.json, thresholds.json, calibrated_* files)
--explain-dirs           Comma-separated dirs from 10 (each contains explain_global.csv)
--tasks                  Comma-separated task names for MTL/Calibration (e.g., diab_incident_365d,cvd_incident_365d)
--topk-features          Top features to show in bar plots (default 20)
--outdir                 Output bundle dir (default artifacts/paper_bundle)

Examples
--------
python scripts/11_paper_bundle.py \
  --cohort-file artifacts/cohort/cohort.parquet \
  --labels-file artifacts/labels/labels.parquet \
  --baselines-root artifacts/baselines \
  --mtl-dirs artifacts/mtl_shared_bottom,artifacts/mtl_cross_stitch,artifacts/mtl_mmoe \
  --calibration-dirs artifacts/calibration/diab_logistic,artifacts/calibration/mtl_shared_bottom_diab \
  --explain-dirs artifacts/explain/diab_logistic,artifacts/explain/cvd_random_forest \
  --tasks diab_incident_365d,cvd_incident_365d \
  --outdir artifacts/paper_bundle
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, precision_recall_curve, average_precision_score, roc_auc_score
)


# ----------------------------- CLI ------------------------------------------ #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Assemble paper-ready bundle (tables + figures) from pipeline artifacts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--cohort-file", type=str, default=None)
    p.add_argument("--labels-file", type=str, default=None)

    p.add_argument("--baselines-root", type=str, default=None,
                   help="Root containing {label}/{model}/metrics.json and test/val preds.")
    p.add_argument("--mtl-dirs", type=str, default="",
                   help="Comma-separated list of 07/08 run dirs (each with metrics.json, TEST/VAL preds).")
    p.add_argument("--calibration-dirs", type=str, default="",
                   help="Comma-separated list of 09 dirs (each with metrics.json, thresholds.json, calibrated_*).")
    p.add_argument("--explain-dirs", type=str, default="",
                   help="Comma-separated list of 10 dirs (each with explain_global.csv).")
    p.add_argument("--tasks", type=str, default="diab_incident_365d,cvd_incident_365d")
    p.add_argument("--topk-features", type=int, default=20)
    p.add_argument("--outdir", type=str, default="artifacts/paper_bundle")
    return p.parse_args()


# ----------------------------- IO helpers ----------------------------------- #

def _safe_read_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None

def _safe_read_parquet(path: Path) -> Optional[pd.DataFrame]:
    try:
        return pd.read_parquet(path)
    except Exception:
        return None

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _split_paths(csv_like: str) -> List[Path]:
    items = [s.strip() for s in (csv_like or "").split(",") if s.strip()]
    return [Path(s) for s in items]


# ----------------------------- Cohort summary ------------------------------- #

def cohort_summary(cohort_path: Optional[Path]) -> Optional[pd.DataFrame]:
    if not cohort_path or not cohort_path.exists():
        return None
    df = pd.read_parquet(cohort_path)
    rows: List[Dict[str, object]] = []
    rows.append({"metric": "N patients", "value": int(df["subject_id"].nunique())})
    if "index_time" in df.columns:
        years = df["index_time"].astype("datetime64[ns]", errors="ignore")
        try:
            years = pd.to_datetime(df["index_time"], errors="coerce").dt.year
            rows.append({"metric": "Index years span", "value": f"{int(years.min())}–{int(years.max())}"})
        except Exception:
            pass
    if "age_at_admit" in df.columns:
        a = pd.to_numeric(df["age_at_admit"], errors="coerce")
        rows.extend([
            {"metric": "Age mean (sd)", "value": f"{a.mean():.1f} ({a.std():.1f})"},
            {"metric": "Age median [IQR]", "value": f"{a.median():.1f} [{a.quantile(0.25):.1f}, {a.quantile(0.75):.1f}]"},
        ])
    if "gender" in df.columns:
        g = df["gender"].astype(str).str.upper().str.strip()
        vc = g.value_counts(dropna=False)
        for k, v in vc.items():
            rows.append({"metric": f"Gender {k}", "value": int(v)})
    return pd.DataFrame(rows)


# ----------------------------- Metrics harvesting --------------------------- #

def harvest_baseline_metrics(root: Optional[Path]) -> pd.DataFrame:
    if not root or not root.exists():
        return pd.DataFrame()
    rows = []
    for label_dir in root.glob("*"):
        if not label_dir.is_dir():
            continue
        for model_dir in label_dir.glob("*"):
            if not model_dir.is_dir():
                continue
            mpath = model_dir / "metrics.json"
            meta = _safe_read_json(mpath)
            if not meta:
                continue
            rows.append({
                "family": "baseline",
                "label": meta.get("label", label_dir.name),
                "model": meta.get("model", model_dir.name),
                "select_metric": meta.get("select_metric"),
                "val_auroc": meta.get("val", {}).get("auroc"),
                "val_auprc": meta.get("val", {}).get("auprc"),
                "test_auroc": meta.get("test", {}).get("auroc"),
                "test_auprc": meta.get("test", {}).get("auprc"),
                "path": str(model_dir),
            })
    return pd.DataFrame(rows)

def harvest_mtl_metrics(mtl_dirs: List[Path]) -> pd.DataFrame:
    rows = []
    for d in mtl_dirs:
        m = _safe_read_json(d / "metrics.json")
        if not m:
            continue
        # Two-head report; store both
        tasks = m.get("task_cols", [])
        for i, t in enumerate(tasks):
            v = m.get("val", {}).get(t, {})
            te = m.get("test", {}).get(t, {})
            rows.append({
                "family": "mtl",
                "variant": Path(d).name,
                "label": t,
                "model": m.get("variant", "shared_bottom") if "variant" in m else "shared_bottom",
                "select_metric": m.get("select_metric"),
                "val_auroc": v.get("auroc"),
                "val_auprc": v.get("auprc"),
                "test_auroc": te.get("auroc"),
                "test_auprc": te.get("auprc"),
                "path": str(d),
            })
    return pd.DataFrame(rows)

def harvest_calibration_metrics(cal_dirs: List[Path]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      metrics_df: metrics.json flattened (validation/test per variant)
      thresholds_df: thresholds.json flattened
    """
    m_rows = []
    t_rows = []
    for d in cal_dirs:
        m = _safe_read_json(d / "metrics.json")
        th = _safe_read_json(d / "thresholds.json")
        if m:
            for split in ["validation", "test"]:
                split_m = m.get(split, {})
                for variant, md in split_m.items():
                    if not isinstance(md, dict):
                        continue
                    m_rows.append({
                        "family": "calibration",
                        "dir": str(d),
                        "split": split,
                        "variant": variant,  # raw/platt/isotonic
                        "auroc": md.get("auroc"),
                        "auprc": md.get("auprc"),
                        "brier": md.get("brier"),
                        "logloss": md.get("logloss"),
                        "ece": md.get("ece"),
                    })
        if th:
            thrmap = th.get("thresholds", {})
            for variant, rules in thrmap.items():
                for rule, val in rules.items():
                    t_rows.append({
                        "dir": str(d),
                        "variant": variant,
                        "rule": rule,
                        "threshold": float(val)
                    })
    return pd.DataFrame(m_rows), pd.DataFrame(t_rows)


# ----------------------------- Predictions loader --------------------------- #

def _load_preds_from_baseline_dir(model_dir: Path) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    """
    Returns list of tuples: (curve_label, y_true, y_score) for TEST.
    """
    out = []
    p = _safe_read_parquet(model_dir / "test_preds.parquet")
    if p is not None and {"y_true", "y_score"}.issubset(p.columns):
        y = p["y_true"].astype(int).values
        s = p["y_score"].astype(float).values
        out.append((f"{model_dir.parent.name}/{model_dir.name}", y, s))
    return out

def _load_preds_from_mtl_dir(mtl_dir: Path, tasks: List[str]) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    out = []
    p = _safe_read_parquet(mtl_dir / "TEST_preds.parquet")
    if p is None:
        return out
    for t in tasks:
        yt = f"{t}_y_true"
        ys = f"{t}_y_score"
        if {yt, ys}.issubset(p.columns):
            y = p[yt].astype(int).values
            s = p[ys].astype(float).values
            out.append((f"{mtl_dir.name}/{t}", y, s))
    return out

def _load_calibrated_preds(cal_dir: Path) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    out = []
    p = _safe_read_parquet(cal_dir / "calibrated_test.parquet")
    if p is None:
        return out
    base = ("raw", p["y_true"].astype(int).values, p["y_score_raw"].astype(float).values)
    out.append((f"{cal_dir.name}/raw", base[1], base[2]))
    for col in p.columns:
        if col.startswith("y_score_") and col != "y_score_raw":
            name = col.replace("y_score_", "")
            out.append((f"{cal_dir.name}/{name}", base[1], p[col].astype(float).values))
    return out


# ----------------------------- Curves & plots ------------------------------- #

def _pr_points(y: np.ndarray, s: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    P, R, _ = precision_recall_curve(y, s)
    ap = average_precision_score(y, s)
    return R, P, ap  # return recall (x), precision (y), AP

def _roc_points(y: np.ndarray, s: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    fpr, tpr, _ = roc_curve(y, s)
    au = roc_auc_score(y, s)
    return fpr, tpr, au

def plot_pr_curves(curves: List[Tuple[str, np.ndarray, np.ndarray]], outpath: Path, title: str) -> None:
    plt.figure(figsize=(6, 5), dpi=150)
    for label, y, s in curves:
        x, yv, ap = _pr_points(y, s)
        plt.plot(x, yv, label=f"{label} (AP={ap:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def plot_roc_curves(curves: List[Tuple[str, np.ndarray, np.ndarray]], outpath: Path, title: str) -> None:
    plt.figure(figsize=(6, 5), dpi=150)
    for label, y, s in curves:
        fpr, tpr, au = _roc_points(y, s)
        plt.plot(fpr, tpr, label=f"{label} (AUROC={au:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def plot_reliability(cal_dir: Path, outpath: Path, bins: int = 10) -> None:
    df = _safe_read_parquet(cal_dir / "calibrated_test.parquet")
    if df is None:
        return
    plt.figure(figsize=(6, 5), dpi=150)
    # Plot each available score variant
    score_cols = [c for c in df.columns if c.startswith("y_score")]
    for col in score_cols:
        p = df[col].astype(float).values
        y = df["y_true"].astype(int).values
        # bin
        edges = np.linspace(0, 1, bins + 1)
        inds = np.digitize(p, edges) - 1
        conf = []; acc = []
        for b in range(bins):
            m = inds == b
            if np.sum(m) == 0:
                continue
            conf.append(np.mean(p[m]))
            acc.append(np.mean(y[m]))
        if conf:
            plt.plot(conf, acc, marker="o", label=col.replace("y_score_", ""))
    # perfect calibration line
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="gray")
    plt.xlabel("Predicted probability (bin mean)")
    plt.ylabel("Observed frequency")
    plt.title(f"Reliability: {cal_dir.name}")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def plot_top_features(explain_dir: Path, outpath: Path, topk: int = 20) -> None:
    tbl = _safe_read_parquet(explain_dir / "explain_global.csv")
    if tbl is None:
        # CSV fallback
        csvp = explain_dir / "explain_global.csv"
        if csvp.exists():
            try:
                tbl = pd.read_csv(csvp)
            except Exception:
                return
        else:
            return
    # Choose a ranking column
    rank_col = None
    for c in ["shap_mean_abs", "perm_importance_mean", "value_abs", "value"]:
        if c in tbl.columns:
            rank_col = c
            break
    if rank_col is None:
        return
    sub = tbl[["feature", rank_col]].dropna().copy()
    sub = sub.sort_values(rank_col, ascending=False).head(topk)
    plt.figure(figsize=(7, max(3, 0.35 * len(sub))), dpi=150)
    plt.barh(range(len(sub))[::-1], sub[rank_col].values[::-1])
    plt.yticks(range(len(sub))[::-1], sub["feature"].values[::-1], fontsize=8)
    plt.xlabel(rank_col)
    plt.title(f"Top features: {Path(explain_dir).name}")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


# ----------------------------- Main orchestration --------------------------- #

def main() -> int:
    args = parse_args()
    outdir = Path(args.outdir)
    _ensure_dir(outdir)
    _ensure_dir(outdir / "tables")
    _ensure_dir(outdir / "figures")

    tasks = [s.strip() for s in args.tasks.split(",") if s.strip()]

    # 1) Cohort summary
    cohort_tbl = cohort_summary(Path(args.cohort_file)) if args.cohort_file else None
    if cohort_tbl is not None and not cohort_tbl.empty:
        cohort_tbl.to_csv(outdir / "tables" / "cohort_summary.csv", index=False)

    # 2) Harvest metrics
    base_df = harvest_baseline_metrics(Path(args.baselines_root)) if args.baselines_root else pd.DataFrame()
    mtl_df = harvest_mtl_metrics(_split_paths(args.mtl_dirs)) if args.mtl_dirs else pd.DataFrame()
    cal_metrics_df, thr_df = harvest_calibration_metrics(_split_paths(args.calibration_dirs)) if args.calibration_dirs else (pd.DataFrame(), pd.DataFrame())

    # Write metrics tables
    if not base_df.empty:
        base_df.to_csv(outdir / "tables" / "metrics_baselines.csv", index=False)
    if not mtl_df.empty:
        mtl_df.to_csv(outdir / "tables" / "metrics_mtl.csv", index=False)
    if not cal_metrics_df.empty:
        cal_metrics_df.to_csv(outdir / "tables" / "metrics_calibration.csv", index=False)
    if not thr_df.empty:
        thr_df.to_csv(outdir / "tables" / "thresholds_summary.csv", index=False)

    # Combined roll-up for quick manuscript table (TEST metrics preferred)
    parts = []
    if not base_df.empty:
        b = base_df.assign(method=lambda d: d["family"] + ":" + d["model"])
        parts.append(b[["label","method","test_auroc","test_auprc","path"]])
    if not mtl_df.empty:
        m = mtl_df.assign(method=lambda d: d["family"] + ":" + d["model"])
        parts.append(m[["label","method","test_auroc","test_auprc","path"]])
    if not cal_metrics_df.empty:
        # Use test + variant
        c = cal_metrics_df[cal_metrics_df["split"]=="test"].copy()
        c = c.assign(method=lambda d: "calibration:" + d["variant"])
        # calibration dirs often correspond to a single label/task; keep dir as path
        c = c.rename(columns={"auroc":"test_auroc","auprc":"test_auprc"})
        c["label"] = Path(c["dir"].iloc[0]).name if len(c)>0 else ""
        c["path"] = c["dir"]
        parts.append(c[["label","method","test_auroc","test_auprc","path"]])
    if parts:
        rollup = pd.concat(parts, ignore_index=True)
        rollup = rollup.sort_values(["label","method"])
        rollup.to_csv(outdir / "tables" / "metrics_rollup.csv", index=False)

    # 3) Curves from available predictions
    curves: List[Tuple[str, np.ndarray, np.ndarray]] = []

    # From baselines
    if args.baselines_root:
        for label_dir in Path(args.baselines_root).glob("*"):
            for model_dir in label_dir.glob("*"):
                curves.extend(_load_preds_from_baseline_dir(model_dir))

    # From MTL
    for d in _split_paths(args.mtl_dirs):
        curves.extend(_load_preds_from_mtl_dir(d, tasks))

    # From calibration (raw + calibrated)
    for d in _split_paths(args.calibration_dirs):
        curves.extend(_load_calibrated_preds(d))

    # Plot PR/ROC if we have at least one curve
    if curves:
        plot_pr_curves(curves, outdir / "figures" / "pr_curves_test.png", "Precision–Recall (TEST)")
        plot_roc_curves(curves, outdir / "figures" / "roc_curves_test.png", "ROC (TEST)")

    # 4) Reliability diagrams for each calibration dir
    for d in _split_paths(args.calibration_dirs):
        plot_reliability(d, outdir / "figures" / f"reliability_{Path(d).name}.png", bins=15)

    # 5) Feature importance bars
    for d in _split_paths(args.explain_dirs):
        plot_top_features(d, outdir / "figures" / f"top_features_{Path(d).name}.png", topk=args.topk_features)

    # 6) README/manifest
    lines: List[str] = []
    lines.append("# Paper Bundle\n\n")
    lines.append("This folder contains tables and figures assembled from the pipeline for manuscript use.\n\n")
    lines.append("## Contents\n")
    lines.append("- `tables/cohort_summary.csv` (if cohort provided)\n")
    lines.append("- `tables/metrics_baselines.csv` and/or `tables/metrics_mtl.csv`\n")
    lines.append("- `tables/metrics_calibration.csv` and `tables/thresholds_summary.csv` (if calibration provided)\n")
    lines.append("- `tables/metrics_rollup.csv` (compact summary table)\n")
    lines.append("- `figures/pr_curves_test.png`, `figures/roc_curves_test.png`\n")
    for d in _split_paths(args.calibration_dirs):
        lines.append(f"- `figures/reliability_{Path(d).name}.png`\n")
    for d in _split_paths(args.explain_dirs):
        lines.append(f"- `figures/top_features_{Path(d).name}.png`\n")
    lines.append("\nGenerated with 11_paper_bundle.py.\n")
    (outdir / "README.md").write_text("".join(lines))

    print(f"Wrote bundle to: {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

