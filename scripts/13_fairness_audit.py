#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python3
"""
13_fairness_audit.py

Purpose
-------
Evaluate subgroup performance and decision parity on TEST predictions.
Works with:
  - Baselines (06):     [subject_id, y_true, y_score]
  - MTL (07/08):        [subject_id, <task>_y_true, <task>_y_score]  (use --task-col)
  - Calibrated (09):    [subject_id, y_true, y_score_raw, y_score_platt?, y_score_isotonic?]

You provide a cohort/meta parquet with patient attributes (e.g., sex, age, ethnicity).
Specify which columns to audit via --group-cols. Numeric columns can be binned.

Outputs
-------
{outdir}/
  group_metrics.csv              # per variant x group value (AUROC, AUPRC, Brier, LogLoss, prevalence)
  group_threshold_metrics.csv    # per variant x rule x group value (TPR/FPR/PPV/NPV/Acc/F1 + counts)
  parity_gaps.csv                # per variant x rule: gaps (max-min) and ratios across groups
  SUMMARY.md                     # human-readable overview
  fairness_report.json           # machine-readable bundle (metrics + gaps + metadata)

Examples
--------
# Baseline predictions, groups by sex and 5-quantile age
python scripts/13_fairness_audit.py \
  --preds artifacts/baselines/diab_incident_365d/logistic/test_preds.parquet \
  --meta-file artifacts/cohort/cohort.parquet \
  --group-cols sex,age \
  --age-bins 5 \
  --outdir artifacts/fairness/diab_logistic

# Calibrated predictions with thresholds from 09
python scripts/13_fairness_audit.py \
  --preds artifacts/calibration/diab_logistic/calibrated_test.parquet \
  --thresholds-json artifacts/calibration/diab_logistic/thresholds.json \
  --meta-file artifacts/cohort/cohort.parquet \
  --group-cols sex,ethnicity \
  --outdir artifacts/fairness/diab_logistic_cal

# MTL head
python scripts/13_fairness_audit.py \
  --preds artifacts/mtl_shared_bottom/TEST_preds.parquet \
  --task-col diab_incident_365d \
  --meta-file artifacts/cohort/cohort.parquet \
  --group-cols sex \
  --outdir artifacts/fairness/mtl_shared_bottom_diab
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
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss, log_loss
)


# ----------------------------- CLI ------------------------------------------ #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Subgroup fairness audit on TEST predictions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--preds", type=str, required=True, help="Path to TEST predictions parquet.")
    p.add_argument("--task-col", type=str, default=None,
                   help="For MTL outputs, the task prefix (e.g., diab_incident_365d).")
    p.add_argument("--score-variants", type=str, default=None,
                   help="Comma-separated score columns to use. If omitted: "
                        "baseline->y_score; MTL-><task>_y_score; calibrated->all y_score_* columns.")
    p.add_argument("--thresholds-json", type=str, default=None,
                   help="Optional thresholds.json from 09; operating points evaluated per variant.")
    p.add_argument("--meta-file", type=str, required=True,
                   help="Parquet with subject attributes (must include subject_id and the group columns).")
    p.add_argument("--group-cols", type=str, required=True,
                   help="Comma-separated column names in meta-file to audit (e.g., sex,age,ethnicity).")
    p.add_argument("--age-bins", type=int, default=0,
                   help="If a 'age' column is provided and >0, bin into this many quantiles.")
    p.add_argument("--numeric-bins", type=int, default=0,
                   help="If >0, bin all other numeric group columns into this many quantiles.")
    p.add_argument("--bootstrap", type=int, default=0,
                   help="Bootstrap replicates for CIs (stratified by group value). 0 disables.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--outdir", type=str, default="artifacts/fairness")
    return p.parse_args()


# ----------------------------- IO & parsing --------------------------------- #

@dataclass
class YS:
    df: pd.DataFrame                # includes subject_id, y_true, and score columns
    y_col: str                      # 'y_true'
    score_cols: List[str]           # columns with scores (variants)
    variants: List[str]             # cleaned names for display (aligned to score_cols)

def load_preds(preds_path: Path, task_col: Optional[str], score_variants: Optional[List[str]]) -> YS:
    df = pd.read_parquet(preds_path)
    cols = set(df.columns)

    # MTL schema
    if task_col:
        yt = f"{task_col}_y_true"
        ys = f"{task_col}_y_score"
        if not {yt, ys}.issubset(cols):
            raise KeyError(f"MTL predictions missing '{yt}'/'{ys}'.")
        y_col = yt
        if score_variants:
            sc = [c for c in score_variants if c in df.columns]
            if not sc:
                raise KeyError("Requested score variants not present in file.")
            names = sc
        else:
            sc = [ys]
            names = ["y_score"]
        return YS(df=df[["subject_id", yt] + sc].copy(), y_col=yt, score_cols=sc, variants=names)

    # Calibrated schema
    if {"y_true"} <= cols and any(c.startswith("y_score") for c in df.columns):
        y_col = "y_true"
        cand = [c for c in df.columns if c.startswith("y_score")]
        if score_variants:
            sc = [v for v in score_variants if v in df.columns]
            if not sc:
                raise KeyError(f"Requested variants not found. Available: {cand}")
            names = [c.replace("y_score_", "") if c != "y_score" else "y_score" for c in sc]
        else:
            sc = cand
            names = [c.replace("y_score_", "") if c != "y_score" else "y_score" for c in sc]
        return YS(df=df[["subject_id", y_col] + sc].copy(), y_col=y_col, score_cols=sc, variants=names)

    # Baseline schema
    if {"y_true", "y_score"}.issubset(cols):
        y_col = "y_true"
        if score_variants:
            sc = [v for v in score_variants if v in df.columns]
            if not sc:
                raise KeyError("Requested score variants not present in baseline preds.")
            names = sc
        else:
            sc = ["y_score"]
            names = ["y_score"]
        return YS(df=df[["subject_id", y_col] + sc].copy(), y_col=y_col, score_cols=sc, variants=names)

    raise ValueError("Could not infer predictions schema. Provide outputs from 06/07-08/09.")

def prepare_groups(meta: pd.DataFrame, group_cols: List[str], age_bins: int, numeric_bins: int) -> pd.DataFrame:
    g = meta.copy()
    if "subject_id" not in g.columns:
        raise KeyError("meta-file must include 'subject_id'.")
    # Normalize typical categorical columns
    for c in group_cols:
        if c not in g.columns:
            raise KeyError(f"Group column '{c}' not found in meta-file.")
        if c.lower() in {"sex", "gender"}:
            g[c] = g[c].astype(str).str.strip().str.upper().replace({
                "FEMALE": "F", "MALE": "M", "F": "F", "M": "M"
            })
    # Age binning
    if "age" in group_cols and age_bins and age_bins > 0 and pd.api.types.is_numeric_dtype(g["age"]):
        try:
            g["age"] = pd.qcut(g["age"], q=age_bins, duplicates="drop")
        except Exception:
            pass
    # Other numeric bins
    if numeric_bins and numeric_bins > 0:
        for c in group_cols:
            if c == "age":
                continue
            if pd.api.types.is_numeric_dtype(g[c]):
                try:
                    g[c] = pd.qcut(g[c], q=numeric_bins, duplicates="drop")
                except Exception:
                    pass
    # Ensure string labels
    for c in group_cols:
        g[c] = g[c].astype(str)
        g.loc[g[c].isin(["nan", "NaT", "None"]), c] = "MISSING"
        g.loc[g[c].eq(""), c] = "MISSING"
    return g[["subject_id"] + group_cols]


# ----------------------------- Metrics utils -------------------------------- #

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
    out["prevalence"] = float(np.mean(y))
    out["n"] = int(len(y))
    return out

def _threshold_metrics(y: np.ndarray, p: np.ndarray, t: float) -> Dict[str, float]:
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
    return {"threshold": float(t), "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "tpr": sens, "fpr": 1 - spec, "ppv": ppv, "npv": npv, "accuracy": acc, "f1": f1, "n": int(len(y))}

def _stratified_boot(y: np.ndarray, p: np.ndarray, group: np.ndarray, reps: int, seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Bootstrap within each group value to preserve composition."""
    rng = np.random.default_rng(seed)
    values = np.unique(group)
    idx_groups = {v: np.where(group == v)[0] for v in values}
    bags = []
    n = len(y)
    for _ in range(reps):
        idx = []
        for v in values:
            gidx = idx_groups[v]
            if len(gidx) == 0:
                continue
            idx.extend(rng.choice(gidx, size=len(gidx), replace=True))
        idx = np.array(idx, dtype=int)
        bags.append((y[idx], p[idx]))
    return bags


# ----------------------------- Fairness logic -------------------------------- #

def compute_group_metrics(df: pd.DataFrame, y_col: str, s_col: str, group_cols: List[str],
                          bootstrap: int, seed: int) -> pd.DataFrame:
    rows = []
    for gc in group_cols:
        for val, sub in df.groupby(gc, dropna=False):
            y = sub[y_col].astype(int).values
            p = sub[s_col].astype(float).values
            m = _metrics(y, p)
            m.update({"group_col": gc, "group_val": str(val), "score_col": s_col})
            # Optional bootstrap CIs (simple percentile)
            if bootstrap and bootstrap > 0 and len(sub) > 1:
                bags = _stratified_boot(y, p, np.full_like(y, fill_value=1), reps=bootstrap, seed=seed)
                auroc = []; auprc = []; brier = []; logloss = []
                for yy, pp in bags:
                    mm = _metrics(yy, pp)
                    auroc.append(mm["auroc"]); auprc.append(mm["auprc"])
                    brier.append(mm["brier"]); logloss.append(mm["logloss"])
                def pct(a): 
                    a = np.array([x for x in a if np.isfinite(x)])
                    if a.size == 0: return (float("nan"), float("nan"))
                    return (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))
                m["auroc_ci_lo"], m["auroc_ci_hi"] = pct(auroc)
                m["auprc_ci_lo"], m["auprc_ci_hi"] = pct(auprc)
                m["brier_ci_lo"], m["brier_ci_hi"] = pct(brier)
                m["logloss_ci_lo"], m["logloss_ci_hi"] = pct(logloss)
            rows.append(m)
    return pd.DataFrame(rows)

def compute_group_threshold_metrics(df: pd.DataFrame, y_col: str, s_col: str, group_cols: List[str],
                                    thresholds: Dict[str, float], bootstrap: int, seed: int) -> pd.DataFrame:
    """
    thresholds: dict of rule->threshold (selected on VAL). Metrics computed per group on TEST.
    """
    rows = []
    for gc in group_cols:
        for val, sub in df.groupby(gc, dropna=False):
            y = sub[y_col].astype(int).values
            p = sub[s_col].astype(float).values
            for rule, thr in thresholds.items():
                tm = _threshold_metrics(y, p, thr)
                tm.update({"group_col": gc, "group_val": str(val), "score_col": s_col, "rule": rule})
                # Optional bootstrap CIs
                if bootstrap and bootstrap > 0 and len(sub) > 1:
                    bags = _stratified_boot(y, p, np.full_like(y, fill_value=1), reps=bootstrap, seed=seed)
                    tpr = []; fpr = []; ppv=[]; npv=[]; f1=[]
                    for yy, pp in bags:
                        mm = _threshold_metrics(yy, pp, thr)
                        tpr.append(mm["tpr"]); fpr.append(mm["fpr"]); ppv.append(mm["ppv"]); npv.append(mm["npv"]); f1.append(mm["f1"])
                    def pct(a):
                        a = np.array([x for x in a if np.isfinite(x)])
                        if a.size == 0: return (float("nan"), float("nan"))
                        return (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))
                    tm["tpr_ci_lo"], tm["tpr_ci_hi"] = pct(tpr)
                    tm["fpr_ci_lo"], tm["fpr_ci_hi"] = pct(fpr)
                    tm["ppv_ci_lo"], tm["ppv_ci_hi"] = pct(ppv)
                    tm["npv_ci_lo"], tm["npv_ci_hi"] = pct(npv)
                    tm["f1_ci_lo"],  tm["f1_ci_hi"]  = pct(f1)
                rows.append(tm)
    return pd.DataFrame(rows)

def parity_gaps(th_tbl: pd.DataFrame) -> pd.DataFrame:
    """
    For each (score_col, rule, group_col) compute gaps (max - min) and ratios (min / max) for TPR/FPR/PPV/NPV.
    """
    if th_tbl.empty:
        return pd.DataFrame()
    rows = []
    for (scol, rule, gcol), sub in th_tbl.groupby(["score_col", "rule", "group_col"]):
        def gap_ratio(series):
            vals = series.replace([np.inf, -np.inf], np.nan).dropna()
            if vals.empty:
                return float("nan"), float("nan")
            mx = float(vals.max()); mn = float(vals.min())
            gap = mx - mn
            ratio = (mn / mx) if mx > 0 else float("nan")
            return gap, ratio
        for name in ["tpr", "fpr", "ppv", "npv", "accuracy", "f1"]:
            gap, ratio = gap_ratio(sub[name])
            rows.append({"score_col": scol, "rule": rule, "group_col": gcol,
                         "metric": name, "gap_max_min": gap, "ratio_min_max": ratio})
    return pd.DataFrame(rows)


# ----------------------------- MAIN ----------------------------------------- #

def main() -> int:
    args = parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # Load predictions (TEST)
    variants_arg = None
    if args.score_variants:
        variants_arg = [v.strip() for v in args.score_variants.split(",") if v.strip()]
    ys = load_preds(Path(args.preds), args.task_col, variants_arg)
    preds_df = ys.df.rename(columns={ys.y_col: "y_true"})
    y_col = "y_true"

    # Load meta and prep groups
    meta = pd.read_parquet(args.meta_file)
    group_cols = [c.strip() for c in args.group_cols.split(",") if c.strip()]
    groups = prepare_groups(meta, group_cols, age_bins=args.age_bins, numeric_bins=args.numeric_bins)

    # Join
    df = preds_df.merge(groups, on="subject_id", how="inner")
    if df.empty:
        raise ValueError("Join of predictions and meta yielded 0 rows. Check 'subject_id' alignment and files.")

    # Load thresholds.json (optional)
    thresholds_all: Dict[str, Dict[str, float]] = {}
    if args.thresholds_json:
        T = json.loads(Path(args.thresholds_json).read_text())
        thresholds_all = T.get("thresholds", {})  # {variant: {rule: thr}}

    # Per-variant processing
    all_group_metrics = []
    all_group_thres = []
    all_parity = []

    for s_col, name in zip(ys.score_cols, ys.variants):
        # Rename working score column to a unified name; keep original for identification
        df["_score"] = df[s_col].astype(float)

        # Group metrics (rank-based + proper scoring)
        gm = compute_group_metrics(df, y_col="y_true", s_col="_score",
                                   group_cols=group_cols, bootstrap=int(args.bootstrap), seed=int(args.seed))
        gm["variant"] = name
        gm["score_col"] = name
        all_group_metrics.append(gm)

        # Thresholded metrics if we have thresholds for this variant
        th_map = None
        if thresholds_all:
            # Threshold keys from 09 are: raw, platt, isotonic (or y_score)
            # Try variant name, and try fallback to exact column name if present.
            if name in thresholds_all:
                th_map = thresholds_all[name]
            elif s_col in thresholds_all:
                th_map = thresholds_all[s_col]
            elif name == "y_score" and "raw" in thresholds_all:
                th_map = thresholds_all["raw"]

        if th_map:
            gt = compute_group_threshold_metrics(df, y_col="y_true", s_col="_score",
                                                 group_cols=group_cols, thresholds=th_map,
                                                 bootstrap=int(args.bootstrap), seed=int(args.seed))
            gt["variant"] = name
            gt["score_col"] = name
            all_group_thres.append(gt)
            all_parity.append(parity_gaps(gt))

        # Cleanup temp
        df.drop(columns=["_score"], inplace=True)

    # Concatenate and persist
    gm_df = pd.concat(all_group_metrics, ignore_index=True) if all_group_metrics else pd.DataFrame()
    gt_df = pd.concat(all_group_thres, ignore_index=True) if all_group_thres else pd.DataFrame()
    pg_df = pd.concat(all_parity, ignore_index=True) if all_parity else pd.DataFrame()

    if not gm_df.empty:
        gm_df.to_csv(outdir / "group_metrics.csv", index=False)
    if not gt_df.empty:
        gt_df.to_csv(outdir / "group_threshold_metrics.csv", index=False)
    if not pg_df.empty:
        pg_df.to_csv(outdir / "parity_gaps.csv", index=False)

    # Summary
    lines: List[str] = []
    lines.append(f"# Fairness Audit Summary\n\nGenerated: {datetime.utcnow().isoformat(timespec='seconds')}Z\n\n")
    lines.append(f"- Predictions: `{args.preds}`\n")
    lines.append(f"- Meta file: `{args.meta_file}`\n")
    if args.task_col:
        lines.append(f"- Task column: `{args.task_col}`\n")
    lines.append(f"- Variants evaluated: {', '.join(ys.variants)}\n")
    lines.append(f"- Group columns: {', '.join(group_cols)}\n")
    if args.thresholds_json:
        lines.append(f"- Thresholds: `{args.thresholds_json}` (VAL-selected)\n")
    if args.bootstrap and args.bootstrap > 0:
        lines.append(f"- Bootstrap replicates: {args.bootstrap} (stratified within groups; seed={args.seed})\n")
    lines.append("\n## Files written\n")
    if not gm_df.empty:
        lines.append("- `group_metrics.csv` (AUROC, AUPRC, Brier, LogLoss, prevalence, n per group)\n")
    if not gt_df.empty:
        lines.append("- `group_threshold_metrics.csv` (TPR/FPR/PPV/NPV/F1/Acc per group at operating points)\n")
    if not pg_df.empty:
        lines.append("- `parity_gaps.csv` (gaps and ratios across groups for each rule)\n")
    (outdir / "SUMMARY.md").write_text("".join(lines))

    # JSON bundle
    bundle = {
        "generated_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "preds": str(Path(args.preds)),
        "meta_file": str(Path(args.meta_file)),
        "task_col": args.task_col,
        "variants": ys.variants,
        "group_cols": group_cols,
        "bootstrap": int(args.bootstrap),
        "thresholds_json": str(Path(args.thresholds_json)) if args.thresholds_json else None,
        "paths": {
            "group_metrics_csv": str(outdir / "group_metrics.csv") if not gm_df.empty else None,
            "group_threshold_metrics_csv": str(outdir / "group_threshold_metrics.csv") if not gt_df.empty else None,
            "parity_gaps_csv": str(outdir / "parity_gaps.csv") if not pg_df.empty else None,
            "summary_md": str(outdir / "SUMMARY.md")
        }
    }
    (outdir / "fairness_report.json").write_text(json.dumps(bundle, indent=2))

    print(f"Wrote fairness outputs to: {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

