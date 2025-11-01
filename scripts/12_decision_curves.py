#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python3
"""
12_decision_curves.py

Purpose
-------
Compute Decision Curve Analysis (DCA) for binary risk models on TEST data.

- For a threshold probability pt in (0,1), classify y_score >= pt as positive.
- Net benefit: NB = (TP/N) - (FP/N) * (pt / (1-pt))
- "Treat none" NB = 0
- "Treat all"  NB_all = prevalence - (1 - prevalence) * (pt / (1-pt))
- Net reduction in interventions per 100: NRI100 = 100 * (NB_model - NB_all) / (pt / (1-pt))

Features
--------
- Accepts baseline preds (06), MTL preds (07/08; needs --task-col), or calibrated preds (09).
- Supports multiple score variants at once (e.g., raw/platt/isotonic from 09).
- Optional bootstrap CIs for NB curves.
- Reports NB at specific operating thresholds from a thresholds.json (09).

Inputs
------
--preds               Path to predictions parquet.
                      Supported schemas:
                        A) Baseline (06): [subject_id, y_true, y_score]
                        B) MTL (07/08):   [subject_id, <task>_y_true, <task>_y_score]  -> pass --task-col
                        C) Calibrated(09): [subject_id?, y_true, y_score_raw, y_score_platt?, y_score_isotonic?]
--task-col            Required for MTL schema (e.g., diab_incident_365d). Ignored otherwise.
--score-variants      Comma list of columns to use from preds file.
                      Omit to auto-detect:
                       - Baseline: uses 'y_score'
                       - MTL: uses '<task>_y_score'
                       - Calibrated: uses all columns prefixed with 'y_score'
--thresholds-json     Optional path to thresholds.json from 09; computes NB at those op points.
--grid                Threshold grid spec as start:end:step (default 0.01:0.99:0.01).
--bootstrap           Number of bootstrap replicates for NB CIs (default 0 = off).
--seed                RNG seed for bootstrap.
--outdir              Output directory (default artifacts/dca)

Outputs
-------
{outdir}/
  dca_curve.csv                # variant, threshold, nb_model, nb_all, nb_none(=0), nri100, n, (ci if bootstrapped)
  operating_points.csv         # NB at named thresholds from thresholds.json (if provided)
  SUMMARY.md                   # human-readable summary
  decision_curve.png           # NB vs threshold for all variants (+ treat-all/none)
  net_reduction_per100.png     # NRI/100 vs threshold (if multiple variants, all plotted)

Examples
--------
# Baseline logistic model
python scripts/12_decision_curves.py \
  --preds artifacts/baselines/diab_incident_365d/logistic/test_preds.parquet \
  --outdir artifacts/dca/diab_logistic

# Calibrated predictions (raw+platt+isotonic)
python scripts/12_decision_curves.py \
  --preds artifacts/calibration/diab_logistic/calibrated_test.parquet \
  --outdir artifacts/dca/diab_logistic_cal

# MTL head
python scripts/12_decision_curves.py \
  --preds artifacts/mtl_shared_bottom/TEST_preds.parquet \
  --task-col diab_incident_365d \
  --outdir artifacts/dca/mtl_shared_bottom_diab
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------- CLI ------------------------------------------ #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Decision Curve Analysis (DCA) on TEST predictions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--preds", type=str, required=True, help="Path to predictions parquet.")
    p.add_argument("--task-col", type=str, default=None, help="MTL: name of task column (e.g., diab_incident_365d).")
    p.add_argument("--score-variants", type=str, default=None,
                   help="Comma-separated list of score columns to use (auto-detect if omitted).")
    p.add_argument("--thresholds-json", type=str, default=None,
                   help="Optional thresholds.json from 09 to evaluate NB at named operating points.")
    p.add_argument("--grid", type=str, default="0.01:0.99:0.01", help="start:end:step for threshold grid.")
    p.add_argument("--bootstrap", type=int, default=0, help="Number of bootstrap replicates for NB CIs (0=off).")
    p.add_argument("--seed", type=int, default=42, help="Random seed for bootstrap.")
    p.add_argument("--outdir", type=str, default="artifacts/dca")
    return p.parse_args()


# ----------------------------- IO helpers ----------------------------------- #

def _safe_read_parquet(p: Path) -> pd.DataFrame:
    return pd.read_parquet(p)

def _safe_read_json(p: Optional[Path]) -> Optional[dict]:
    if not p:
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


# ----------------------------- Preds parsing -------------------------------- #

@dataclass
class YS:
    y: np.ndarray
    scores: Dict[str, np.ndarray]  # variant -> scores


def load_predictions(preds_path: Path, task_col: Optional[str], score_variants: Optional[List[str]]) -> YS:
    df = _safe_read_parquet(preds_path)

    # Identify schema
    cols = set(df.columns)
    variants: Dict[str, np.ndarray] = {}

    if {"y_true", "y_score"}.issubset(cols):
        # Baseline schema
        y = df["y_true"].astype(int).values
        if score_variants:
            for v in score_variants:
                if v == "y_score":
                    variants["y_score"] = df["y_score"].astype(float).values
                elif v in df.columns:
                    variants[v] = df[v].astype(float).values
                else:
                    raise KeyError(f"Requested score variant '{v}' not in file.")
        else:
            variants["y_score"] = df["y_score"].astype(float).values
        return YS(y=y, scores=variants)

    # Calibrated schema
    if {"y_true"} <= cols and any(c.startswith("y_score") for c in df.columns):
        y = df["y_true"].astype(int).values
        cand = [c for c in df.columns if c.startswith("y_score")]
        if score_variants:
            use = [v for v in score_variants if v in cand]
            if not use:
                raise KeyError(f"None of requested score variants found. Available: {cand}")
        else:
            use = cand
        for c in use:
            name = c.replace("y_score_", "") if c != "y_score" else "y_score"
            variants[name] = df[c].astype(float).values
        return YS(y=y, scores=variants)

    # MTL schema
    if task_col:
        yt = f"{task_col}_y_true"
        ys = f"{task_col}_y_score"
        need = {yt, ys}
        if not need.issubset(cols):
            raise KeyError(f"MTL predictions missing columns: {need - cols}")
        y = df[yt].astype(int).values
        if score_variants:
            use = [v for v in score_variants if v in df.columns]
            if not use:
                raise KeyError(f"Requested variants not found in MTL file.")
            for c in use:
                variants[c] = df[c].astype(float).values
        else:
            variants["y_score"] = df[ys].astype(float).values
        return YS(y=y, scores=variants)

    raise ValueError("Could not infer predictions schema. Provide a supported preds file (06/07-08/09 outputs).")


# ----------------------------- DCA core ------------------------------------- #

def parse_grid(spec: str) -> np.ndarray:
    try:
        s, e, step = spec.split(":")
        s = float(s); e = float(e); step = float(step)
        if not (0.0 < s < e < 1.0) or step <= 0:
            raise ValueError
        # inclusive end if aligns to grid
        grid = np.arange(s, e + 1e-12, step)
        grid = grid[(grid > 0) & (grid < 1)]
        return grid
    except Exception:
        raise ValueError(f"Invalid --grid spec '{spec}'. Use start:end:step with 0<start<end<1.")

def net_benefit(y: np.ndarray, s: np.ndarray, pt: float) -> Tuple[float, float, float, float]:
    """
    Returns (NB_model, NB_all, NB_none, NRI100)
    """
    y = y.astype(int); s = s.astype(float)
    n = len(y)
    pred = (s >= pt).astype(int)
    tp = float(np.sum((pred == 1) & (y == 1)))
    fp = float(np.sum((pred == 1) & (y == 0)))
    prev = float(np.mean(y))
    wt = pt / (1.0 - pt)
    nb_model = (tp / n) - (fp / n) * wt
    nb_all = prev - (1.0 - prev) * wt
    nb_none = 0.0
    nri100 = 100.0 * (nb_model - nb_all) / wt
    return nb_model, nb_all, nb_none, nri100

def dca_curve(y: np.ndarray, s: np.ndarray, grid: np.ndarray) -> pd.DataFrame:
    rows = []
    n = len(y)
    for pt in grid:
        nb_m, nb_a, nb_n, nri = net_benefit(y, s, float(pt))
        rows.append({"threshold": float(pt), "n": int(n),
                     "nb_model": float(nb_m), "nb_all": float(nb_a), "nb_none": float(nb_n),
                     "nri100": float(nri)})
    return pd.DataFrame(rows)

def bootstrap_ci(y: np.ndarray, s: np.ndarray, grid: np.ndarray, reps: int, seed: int = 42,
                 q: Tuple[float,float]=(2.5,97.5)) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(y)
    curves = []
    for _ in range(reps):
        idx = rng.integers(0, n, size=n)
        yy = y[idx]; ss = s[idx]
        curves.append(dca_curve(yy, ss, grid)["nb_model"].values)
    arr = np.vstack(curves)  # (reps, len(grid))
    lo, hi = np.percentile(arr, q=q, axis=0)
    return pd.DataFrame({"threshold": grid, "nb_model_lo": lo, "nb_model_hi": hi})


# ----------------------------- Plotting ------------------------------------- #

def plot_dca(all_curves: Dict[str, pd.DataFrame], outpath: Path, title: str) -> None:
    plt.figure(figsize=(7, 5), dpi=150)
    # All model variants
    for name, df in all_curves.items():
        plt.plot(df["threshold"], df["nb_model"], label=name)
        # Add CI ribbon if present
        if {"nb_model_lo", "nb_model_hi"} <= set(df.columns):
            plt.fill_between(df["threshold"], df["nb_model_lo"], df["nb_model_hi"], alpha=0.15)
    # Reference lines: treat-all/none from the first variant
    any_df = next(iter(all_curves.values()))
    plt.plot(any_df["threshold"], any_df["nb_all"], linestyle="--", linewidth=1, label="treat_all")
    plt.plot(any_df["threshold"], np.zeros_like(any_df["threshold"]), linestyle="--", linewidth=1, label="treat_none")
    plt.xlabel("Threshold probability")
    plt.ylabel("Net benefit")
    plt.title(title)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def plot_nri(all_curves: Dict[str, pd.DataFrame], outpath: Path, title: str) -> None:
    plt.figure(figsize=(7, 5), dpi=150)
    for name, df in all_curves.items():
        plt.plot(df["threshold"], df["nri100"], label=name)
    plt.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    plt.xlabel("Threshold probability")
    plt.ylabel("Net reduction in interventions / 100")
    plt.title(title)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


# ----------------------------- Operating points ----------------------------- #

def evaluate_operating_points(y: np.ndarray, s_map: Dict[str, np.ndarray], thresholds_json: Optional[Path]) -> pd.DataFrame:
    meta = _safe_read_json(thresholds_json) if thresholds_json else None
    if not meta:
        return pd.DataFrame()
    th_map = meta.get("thresholds", {})
    rows = []
    for variant, rules in th_map.items():
        # Map variant name to our score key
        # In calibrated files we used names like 'raw','platt','isotonic'
        # If user provided different keys, skip gracefully.
        if variant not in s_map:
            continue
        ss = s_map[variant]
        for rule, t in rules.items():
            nb_m, nb_a, nb_n, nri = net_benefit(y, ss, float(t))
            rows.append({
                "variant": variant, "rule": rule, "threshold": float(t),
                "nb_model": float(nb_m), "nb_all": float(nb_a), "nb_none": float(nb_n), "nri100": float(nri),
                "n": int(len(y))
            })
    return pd.DataFrame(rows)


# ----------------------------- MAIN ----------------------------------------- #

def main() -> int:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load predictions
    variants_arg = None
    if args.score_variants:
        variants_arg = [v.strip() for v in args.score_variants.split(",") if v.strip()]
    ys = load_predictions(Path(args.preds), args.task_col, variants_arg)
    y = ys.y
    score_map = ys.scores  # name -> np.ndarray

    # Threshold grid
    grid = parse_grid(args.grid)

    # Compute DCA curves (+ bootstrap CIs if requested)
    curves: Dict[str, pd.DataFrame] = {}
    for name, s in score_map.items():
        df = dca_curve(y, s, grid)
        if args.bootstrap and args.bootstrap > 0:
            ci = bootstrap_ci(y, s, grid, reps=int(args.bootstrap), seed=int(args.seed))
            df = df.merge(ci, on="threshold", how="left")
        curves[name] = df

    # Persist curves
    # Long-form table concatenating all variants
    parts = []
    for name, df in curves.items():
        tmp = df.copy()
        tmp.insert(0, "variant", name)
        parts.append(tmp)
    dca_tbl = pd.concat(parts, ignore_index=True)
    dca_tbl.to_csv(outdir / "dca_curve.csv", index=False)

    # Operating points (from thresholds.json)
    op_tbl = evaluate_operating_points(y, score_map, Path(args.thresholds_json) if args.thresholds_json else None)
    if not op_tbl.empty:
        op_tbl.to_csv(outdir / "operating_points.csv", index=False)

    # Plots
    plot_dca(curves, outdir / "decision_curve.png", title=Path(args.preds).parent.name)
    plot_nri(curves, outdir / "net_reduction_per100.png", title=Path(args.preds).parent.name)

    # Summary
    lines: List[str] = []
    lines.append(f"# Decision Curve Analysis\n\nGenerated: {datetime.utcnow().isoformat(timespec='seconds')}Z\n\n")
    lines.append(f"- Predictions file: `{args.preds}`\n")
    if args.task_col:
        lines.append(f"- Task column: `{args.task_col}`\n")
    lines.append(f"- Variants: {', '.join(curves.keys())}\n")
    lines.append(f"- Grid: {args.grid}\n")
    if args.bootstrap and args.bootstrap > 0:
        lines.append(f"- Bootstrap replicates: {args.bootstrap} (seed={args.seed})\n")
    prev = float(np.mean(y))
    lines.append(f"\nEmpirical prevalence on TEST: **{prev:.4f}**\n\n")
    lines.append("Outputs:\n")
    lines.append("- `dca_curve.csv` (net benefit & net reduction across thresholds for each variant)\n")
    if not op_tbl.empty:
        lines.append("- `operating_points.csv` (NB at VAL-selected thresholds from 09)\n")
    lines.append("- `decision_curve.png` (NB vs threshold)\n")
    lines.append("- `net_reduction_per100.png` (net reduction vs threshold)\n")
    (outdir / "SUMMARY.md").write_text("".join(lines))

    print(f"Wrote: {outdir / 'dca_curve.csv'}")
    if not op_tbl.empty:
        print(f"Wrote: {outdir / 'operating_points.csv'}")
    print(f"Wrote: {outdir / 'decision_curve.png'}")
    print(f"Wrote: {outdir / 'net_reduction_per100.png'}")
    print(f"Wrote: {outdir / 'SUMMARY.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

