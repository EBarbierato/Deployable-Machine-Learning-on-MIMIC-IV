#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Global explainability for baseline classifiers (coeffs/importances, permutation, SHAP).

Usage example:
python scripts/10_explainability.py \
  --x-val "$OUT/imputed/X_val.parquet" \
  --labels "$OUT/labels/labels.parquet" \
  --label-col diab_incident_365d \
  --model-type logistic \
  --model-path "$OUT/baselines/diab_incident_365d/logistic/model.joblib" \
  --metric auprc \
  --top-k 30 \
  --shap True --shap-n-sample 2000 --shap-bg 200 \
  --outdir "$OUT/explain/diab_incident_365d_logistic"
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

# sklearn
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    brier_score_loss,
    log_loss,
    make_scorer,
)
from sklearn.inspection import permutation_importance
from sklearn.utils.validation import check_is_fitted

import joblib
import pyarrow.parquet as pq
import pyarrow as pa


# -----------------------
# Utilities
# -----------------------
def read_parquet_to_pandas(path: Path, columns: Optional[list[str]] = None) -> pd.DataFrame:
    table = pq.read_table(path, columns=columns)
    return table.to_pandas()


def align_X_to_model_features(X: pd.DataFrame, model) -> pd.DataFrame:
    """Align validation matrix to the model's training-time columns, if available."""
    cols = getattr(model, "feature_names_in_", None)
    if cols is None:
        # best effort: ensure only numeric columns and sort columns for stability
        Xn = X.select_dtypes(include=[np.number]).copy()
        Xn = Xn.fillna(0.0)
        return Xn
    # add any missing columns as 0 and reorder
    X_aligned = pd.DataFrame(index=X.index)
    for c in cols:
        if c in X.columns:
            X_aligned[c] = X[c]
        else:
            X_aligned[c] = 0.0
    # drop any extras
    return X_aligned.fillna(0.0)


def build_scorer(name: str):
    """Return a sklearn scorer without unsupported kwargs across versions."""
    name = name.lower()
    if name == "auroc":
        return make_scorer(roc_auc_score, needs_threshold=True)
    if name == "auprc":
        # average_precision_score expects probabilities or confidence scores.
        # Don't pass needs_proba/needs_threshold to avoid version errors.
        return make_scorer(average_precision_score)
    if name == "brier":
        return make_scorer(brier_score_loss, greater_is_better=False)
    if name == "logloss":
        return make_scorer(log_loss, greater_is_better=False)
    raise ValueError(f"Unknown metric: {name}")


def compute_permutation_importance(
    model,
    X_df: pd.DataFrame,
    y: np.ndarray,
    metric: str = "auprc",
    n_repeats: int = 10,
    random_state: int = 42,
) -> Optional[pd.DataFrame]:
    scorer = build_scorer(metric)
    try:
        r = permutation_importance(
            model, X_df, y, scoring=scorer, n_repeats=n_repeats, random_state=random_state, n_jobs=-1
        )
        imp = pd.DataFrame(
            {"feature": X_df.columns, "importances_mean": r.importances_mean, "importances_std": r.importances_std}
        ).sort_values("importances_mean", ascending=False)
        return imp
    except Exception as e:
        print(f"[warn] permutation importance failed: {e}")
        return None


# -----------------------
# SHAP helpers
# -----------------------
def _to_numeric_matrix(X: pd.DataFrame) -> np.ndarray:
    """
    Robustly convert a (potentially mixed-dtype) DataFrame to a float64 numpy matrix.
    Handles pandas nullable dtypes (Int64/boolean), coercing non-numerics to NaN then 0.
    """
    # Ensure index alignment and no stray non-feature columns slip through here.
    X = X.copy()
    # Convert every column to numeric where possible; non-numerics -> NaN
    X_num = X.apply(pd.to_numeric, errors="coerce")
    # Fill any NaNs that may remain (imputation should have been done earlier)
    X_num = X_num.fillna(0.0)
    # Export contiguous float64 matrix
    return X_num.to_numpy(dtype=np.float64, copy=False)



def _safe_shap_values(explainer, X):
    try:
        return explainer.shap_values(X, check_additivity=False)
    except TypeError:
        return explainer.shap_values(X)


def compute_shap(model, X_df: pd.DataFrame, model_type: str, max_eval=2000, bg_size=200):
    try:
        import shap  # import inside to avoid hard dependency during CI
    except Exception as e:
        print(f"[warn] SHAP not available: {e}")
        return None, None, None, None

    X_eval = X_df.iloc[: min(max_eval, len(X_df))].copy()
    X_eval_np = _to_numeric_matrix(X_eval)

    # Background for linear/kernel explainers
    bg = X_df.sample(min(bg_size, len(X_df)), random_state=42)
    bg_np = _to_numeric_matrix(bg)

    shap_values = None
    base_value = None
    method_used = None

    try:
        if model_type == "random_forest":
            expl = shap.TreeExplainer(model)
            shap_values = _safe_shap_values(expl, X_eval_np)
            if isinstance(shap_values, list) and len(shap_values) == 2:
                shap_values = shap_values[1]
            base = expl.expected_value
            base_value = np.mean(base) if isinstance(base, (list, tuple, np.ndarray)) else base
            method_used = "TreeExplainer"
        elif model_type == "logistic":
            # try linear first
            try:
                expl = shap.LinearExplainer(model, bg_np)
                shap_values = _safe_shap_values(expl, X_eval_np)
                base_value = expl.expected_value
                method_used = "LinearExplainer"
            except Exception:
                # fallback to kernel
                f = getattr(model, "predict_proba", None)
                if f is not None:
                    fx = lambda z: f(z)[:, 1]
                else:
                    fx = model.predict
                expl = shap.KernelExplainer(fx, bg_np)
                shap_values = _safe_shap_values(expl, X_eval_np)
                base_value = expl.expected_value
                method_used = "KernelExplainer"
        else:
            # generic fallback
            f = getattr(model, "predict_proba", None)
            if f is not None:
                fx = lambda z: f(z)[:, 1]
            else:
                fx = model.predict
            expl = shap.KernelExplainer(fx, bg_np)
            shap_values = _safe_shap_values(expl, X_eval_np)
            base_value = expl.expected_value
            method_used = "KernelExplainer"

        return shap_values, base_value, method_used, X_eval
    except Exception as e:
        print(f"[warn] SHAP computation failed: {e}")
        return None, None, None, None


def save_shap_outputs(shap_values, X_eval: pd.DataFrame, outdir: Path, name: str, top_k: int = 30):
    outdir.mkdir(parents=True, exist_ok=True)
    # Write global CSV by mean |SHAP|
    try:
        mean_abs = np.mean(np.abs(shap_values), axis=0)
        tbl = pd.DataFrame({"feature": list(X_eval.columns), "mean_abs_shap": mean_abs}) \
                .sort_values("mean_abs_shap", ascending=False)
        tbl.to_csv(outdir / "shap_global.csv", index=False)
    except Exception as e:
        print(f"[warn] Failed to save shap_global.csv: {e}")

    # Save plots
    try:
        import shap
        import matplotlib.pyplot as plt
        plt.figure()
        shap.summary_plot(shap_values, X_eval, show=False, plot_type="bar")
        plt.tight_layout()
        plt.savefig(outdir / f"shap_bar_{name}.png", dpi=180, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"[warn] Failed to save shap_bar: {e}")

    try:
        import shap
        import matplotlib.pyplot as plt
        plt.figure()
        shap.summary_plot(shap_values, X_eval, show=False)
        plt.tight_layout()
        plt.savefig(outdir / f"shap_summary_{name}.png", dpi=180, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"[warn] Failed to save shap_summary: {e}")

    # Minimal EXPLAIN.md
    try:
        head = pd.read_csv(outdir / "shap_global.csv").head(top_k)
        lines = ["# Global explainability", "", "Top features by mean |SHAP|:", ""]
        for _, r in head.iterrows():
            lines.append(f"- {r['feature']}: {r['mean_abs_shap']:.6f}")
        (outdir / "EXPLAIN.md").write_text("\n".join(lines), encoding="utf-8")
    except Exception as e:
        print(f"[warn] Failed to write EXPLAIN.md: {e}")


# -----------------------
# Main
# -----------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Global explainability for baseline classifiers (coeffs/importances, permutation, SHAP)."
    )
    p.add_argument("--x-val", required=True, help="Validation feature matrix (Parquet) with 'subject_id'.")
    p.add_argument("--labels", required=True, help="Labels parquet with 'subject_id' and the label column.")
    p.add_argument("--label-col", required=True)
    p.add_argument("--model-type", required=True, choices=["logistic", "random_forest", "xgboost"])
    p.add_argument("--model-path", required=True, help="Path to model.joblib saved by 06_train_baselines.py")
    p.add_argument("--metric", default="auprc", choices=["auprc", "auroc", "brier", "logloss"],
                   help="Metric for permutation importance (higher is better).")
    p.add_argument("--top-k", type=int, default=30, help="Top features to list in EXPLAIN.md")
    p.add_argument("--shap", type=lambda s: s.lower() in {"1", "true", "yes"}, default=True,
                   help="Compute SHAP summaries (if shap available).")
    p.add_argument("--shap-n-sample", type=int, default=2000, help="Max rows from VAL to use for SHAP.")
    p.add_argument("--shap-bg", type=int, default=200, help="Background sample size for SHAP.")
    p.add_argument("--outdir", default="artifacts/explain", help="Output directory.")
    return p.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load data
    X_val = read_parquet_to_pandas(Path(args.x_val))
    y_df = read_parquet_to_pandas(Path(args.labels), columns=["subject_id", args.label_col])
    if "subject_id" not in X_val.columns:
        raise SystemExit("x-val parquet must contain 'subject_id' column.")
    # Join on subject_id
    X_val = X_val.set_index("subject_id").sort_index()
    y_df = y_df.set_index("subject_id").sort_index()
    common = X_val.index.intersection(y_df.index)
    X_val = X_val.loc[common]
    y = y_df.loc[common, args.label_col].astype(int).values

    # Load model
    model = joblib.load(args.model_path)

    # Align columns
    X_aligned = align_X_to_model_features(X_val, model)

    # Save a small header for traceability
    (outdir / "explain_global.csv").write_text("feature,importance\n", encoding="utf-8")

    # Permutation importance (best-effort)
    perm = compute_permutation_importance(model, X_aligned, y, metric=args.metric)
    if perm is not None:
        # save PNG bar (top-30) and CSV
        try:
            import matplotlib.pyplot as plt
            top = perm.head(args.top_k)
            plt.figure(figsize=(7, 5))
            plt.barh(top["feature"][::-1], top["importances_mean"][::-1])
            plt.xlabel(f"Permutation importance ({args.metric})")
            plt.tight_layout()
            plt.savefig(outdir / ("top30_perm_importance_random_forest.png" if args.model_type == "random_forest"
                                  else "top30_permutation_importance.png"),
                        dpi=180, bbox_inches="tight")
            plt.close()
        except Exception as e:
            print(f"[warn] Failed to save permutation bar plot: {e}")
        try:
            perm.to_csv(outdir / "explain_global.csv", index=False)
        except Exception as e:
            print(f"[warn] Failed to save explain_global.csv from permutation: {e}")

    # SHAP summaries
    if args.shap:
        shap_values, base_value, method_used, X_eval = compute_shap(
            model, X_aligned, args.model_type, max_eval=args.shap_n_sample, bg_size=args.shap_bg
        )
        if shap_values is not None:
            save_shap_outputs(shap_values, X_eval, outdir, name=args.model_type, top_k=args.top_k)
        else:
            print("[warn] SHAP returned no values; skipping plots.")

    # Small footer
    print(f"[ok] wrote outputs under: {outdir}")


if __name__ == "__main__":
    # Quiet common warnings that confused earlier runs
    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.utils.validation")
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
