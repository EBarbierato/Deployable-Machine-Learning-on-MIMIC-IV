#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python3
"""
05_impute_and_scale.py

Purpose
-------
Train imputers/scalers on TRAIN only and transform VAL/TEST without leakage.
Drops columns that are unusable (all-NaN on train) or non-informative (zero variance on train).
Optionally adds missingness indicators per feature (computed on TRAIN pattern).

Inputs
------
- features.parquet  (from 03_build_features.py) with columns: subject_id, <features...>
- splits.parquet    (from 04_train_val_test_split.py) with columns: subject_id, fold

Outputs
-------
- {outdir}/X_train.parquet
- {outdir}/X_val.parquet
- {outdir}/X_test.parquet
- {outdir}/transforms.joblib           # dict with fitted imputers/scalers and metadata
- {outdir}/TRANSFORMS.md               # human-readable summary
- {outdir}/transforms_hashes.json      # integrity hashes of column sets

Examples
--------
python scripts/05_impute_and_scale.py \
  --features-file artifacts/features/features.parquet \
  --splits-file artifacts/splits/splits.parquet \
  --imputer median \
  --add-missing-indicators true \
  --scaler standard \
  --no-scale-binary true \
  --seed 42 \
  --outdir artifacts/imputed
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer

# IterativeImputer is experimental in some sklearn versions
try:
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer
    _HAVE_ITER = True
except Exception:
    IterativeImputer = None  # type: ignore
    _HAVE_ITER = False

try:
    from sklearn.linear_model import BayesianRidge
    from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
except Exception:
    BayesianRidge = None  # type: ignore
    ExtraTreesRegressor = None  # type: ignore
    RandomForestRegressor = None  # type: ignore


# ----------------------------- CLI ------------------------------------------ #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Impute and scale features with strict train-only fitting.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--features-file", type=str, required=True)
    p.add_argument("--splits-file", type=str, required=True)
    p.add_argument("--outdir", type=str, default="artifacts/imputed")

    p.add_argument("--imputer", type=str, choices=["median", "most_frequent", "iterative"], default="median")
    p.add_argument("--iterative-estimator", type=str, choices=["bayesian_ridge", "extra_trees", "random_forest"],
                   default="bayesian_ridge", help="Used only when --imputer iterative.")
    p.add_argument("--iterative-max-iter", type=int, default=10)

    p.add_argument("--add-missing-indicators", type=str2bool, default=True,
                   help="Add _mis flags based on missingness pattern in TRAIN.")
    p.add_argument("--scaler", type=str, choices=["standard", "robust", "quantile", "none"], default="standard")
    p.add_argument("--no-scale-binary", type=str2bool, default=True,
                   help="Do not scale columns that are binary (0/1) in TRAIN after imputation.")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--float32", type=str2bool, default=True,
                   help="Store float features as float32 to save space.")
    return p.parse_args()

def str2bool(v: str) -> bool:
    return str(v).lower() in {"1", "true", "t", "yes", "y"}


# --------------------------- Utilities -------------------------------------- #

def _hash_columns(cols: List[str]) -> str:
    # stable-ish short hash
    s = "|".join(cols)
    return str(pd.util.hash_pandas_object(pd.Series([s]), index=False).astype(str).iloc[0])[:32]

def _is_binary_col(series: pd.Series) -> bool:
    vals = pd.unique(series.dropna())
    if len(vals) == 0:
        return False
    try:
        vals = pd.Series(vals).astype(float)
    except Exception:
        return False
    return set(vals.tolist()).issubset({0.0, 1.0})

def _select_numeric_features(df: pd.DataFrame, exclude: List[str]) -> List[str]:
    return [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]

def _drop_all_nan_and_zero_var(X_tr: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
    all_nan = [c for c in X_tr.columns if X_tr[c].isna().all()]
    X = X_tr.drop(columns=all_nan) if all_nan else X_tr
    zero_var = []
    for c in X.columns:
        # consider only non-NaN variance
        s = X[c].dropna()
        if len(s) == 0 or s.nunique(dropna=True) <= 1:
            zero_var.append(c)
    X = X.drop(columns=zero_var) if zero_var else X
    return X, all_nan, zero_var

def _fit_imputer(name: str, X_tr: pd.DataFrame, args: argparse.Namespace):
    if name in {"median", "most_frequent"}:
        imp = SimpleImputer(strategy=name)
        imp.fit(X_tr.values)
        return imp
    if name == "iterative":
        if not _HAVE_ITER:
            raise RuntimeError("IterativeImputer not available in your sklearn version.")
        # choose estimator
        est = None
        if args.iterative_estimator == "bayesian_ridge":
            if BayesianRidge is None:
                raise RuntimeError("BayesianRidge not available.")
            est = BayesianRidge()
        elif args.iterative_estimator == "extra_trees":
            if ExtraTreesRegressor is None:
                raise RuntimeError("ExtraTreesRegressor not available.")
            est = ExtraTreesRegressor(n_estimators=100, random_state=args.seed, n_jobs=-1)
        else:
            if RandomForestRegressor is None:
                raise RuntimeError("RandomForestRegressor not available.")
            est = RandomForestRegressor(n_estimators=200, random_state=args.seed, n_jobs=-1)
        imp = IterativeImputer(
            random_state=args.seed, max_iter=args.iterative_max_iter,
            estimator=est, sample_posterior=False, skip_complete=True
        )
        imp.fit(X_tr.values)
        return imp
    raise ValueError(f"Unknown imputer: {name}")

def _fit_scaler(name: str, X_tr_scaled_cols: pd.DataFrame, seed: int):
    if name == "none":
        return None
    if name == "standard":
        sc = StandardScaler(with_mean=True, with_std=True)
    elif name == "robust":
        sc = RobustScaler(with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0))
    else:  # quantile
        sc = QuantileTransformer(n_quantiles=min(1000, max(10, X_tr_scaled_cols.shape[0] // 2)),
                                 output_distribution="normal", random_state=seed, subsample=int(1e9))
    sc.fit(X_tr_scaled_cols.values)
    return sc

def _add_missing_indicators(train_missing_mask: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
    out = X.copy()
    for c in train_missing_mask.columns:
        if train_missing_mask[c].any():
            out[f"{c}__mis"] = X[c].isna().astype(np.int8)
    return out

def _cast_float32(df: pd.DataFrame, float32: bool) -> pd.DataFrame:
    if not float32:
        return df
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].astype(np.float32)
    return df


# ------------------------------ MAIN ---------------------------------------- #

def main() -> int:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    feats = pd.read_parquet(args.features_file)
    splits = pd.read_parquet(args.splits_file)

    # Merge and basic integrity
    df = feats.merge(splits[["subject_id", "fold"]], on="subject_id", how="inner")
    if df["subject_id"].duplicated().any():
        raise AssertionError("Duplicate subject_id after merge. Check inputs for one-row-per-subject invariant.")
    if set(df["fold"].unique()) != {"train", "val", "test"}:
        raise AssertionError("Splits file must contain train/val/test folds.")

    id_col = "subject_id"
    exclude_cols = [id_col, "fold"]
    feat_cols_all = _select_numeric_features(df, exclude_cols)
    if not feat_cols_all:
        raise ValueError("No numeric features found to impute/scale.")

    # Partition by fold
    tr = df[df["fold"] == "train"].reset_index(drop=True)
    va = df[df["fold"] == "val"].reset_index(drop=True)
    te = df[df["fold"] == "test"].reset_index(drop=True)

    X_tr = tr[feat_cols_all].copy()
    X_va = va[feat_cols_all].copy()
    X_te = te[feat_cols_all].copy()

    # Missingness indicators (computed from TRAIN pattern only)
    if args.add_missing_indicators:
        train_missing_mask = X_tr.isna()
        X_tr = _add_missing_indicators(train_missing_mask, X_tr)
        X_va = _add_missing_indicators(train_missing_mask, X_va)
        X_te = _add_missing_indicators(train_missing_mask, X_te)

    # Drop all-NaN and zero-variance (on TRAIN), propagate to VAL/TEST
    X_tr, dropped_all_nan, dropped_zero_var = _drop_all_nan_and_zero_var(X_tr)
    keep_cols = list(X_tr.columns)
    X_va = X_va[keep_cols].copy()
    X_te = X_te[keep_cols].copy()

    # Impute
    imputer = _fit_imputer(args.imputer, X_tr, args)
    X_tr_imp = pd.DataFrame(imputer.transform(X_tr.values), columns=keep_cols, index=X_tr.index)
    X_va_imp = pd.DataFrame(imputer.transform(X_va.values), columns=keep_cols, index=X_va.index)
    X_te_imp = pd.DataFrame(imputer.transform(X_te.values), columns=keep_cols, index=X_te.index)

    # If some columns were binary before imputation, try to coerce tiny numeric drift back to {0,1}
    # (only for columns that were binary on TRAIN prior to imputation)
    was_binary = {c: _is_binary_col(tr[c]) for c in keep_cols if c in tr.columns}
    for c, is_bin in was_binary.items():
        if is_bin and args.no_scale_binary:
            # After imputation they are floats; clip/round
            for X in (X_tr_imp, X_va_imp, X_te_imp):
                X[c] = np.where(np.isnan(X[c]), 0.0, X[c])
                X[c] = (X[c] >= 0.5).astype(np.float32)  # keep as float for consistent dtypes

    # Split columns for scaling vs passthrough
    if args.no_scale_binary:
        scale_cols = [c for c in keep_cols if not was_binary.get(c, False)]
        passthrough_cols = [c for c in keep_cols if was_binary.get(c, False)]
    else:
        scale_cols = keep_cols
        passthrough_cols = []

    # Fit scaler on TRAIN (only on selected columns)
    if args.scaler == "none" or len(scale_cols) == 0:
        scaler = None
        X_tr_s = X_tr_imp.copy()
        X_va_s = X_va_imp.copy()
        X_te_s = X_te_imp.copy()
    else:
        scaler = _fit_scaler(args.scaler, X_tr_imp[scale_cols], args.seed)
        # Transform and reconstruct frames in original column order
        def apply_scale(X_imp: pd.DataFrame) -> pd.DataFrame:
            X_scaled = X_imp.copy()
            X_scaled.loc[:, scale_cols] = scaler.transform(X_imp[scale_cols].values)
            return X_scaled
        X_tr_s = apply_scale(X_tr_imp)
        X_va_s = apply_scale(X_va_imp)
        X_te_s = apply_scale(X_te_imp)

    # Cast floats if requested
    X_tr_s = _cast_float32(X_tr_s, args.float32)
    X_va_s = _cast_float32(X_va_s, args.float32)
    X_te_s = _cast_float32(X_te_s, args.float32)

    # Reattach subject_id
    X_train = pd.concat([tr[[id_col]].reset_index(drop=True), X_tr_s.reset_index(drop=True)], axis=1)
    X_val   = pd.concat([va[[id_col]].reset_index(drop=True), X_va_s.reset_index(drop=True)], axis=1)
    X_test  = pd.concat([te[[id_col]].reset_index(drop=True), X_te_s.reset_index(drop=True)], axis=1)

    # Persist matrices
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    X_train_path = outdir / "X_train.parquet"
    X_val_path   = outdir / "X_val.parquet"
    X_test_path  = outdir / "X_test.parquet"
    X_train.to_parquet(X_train_path, index=False)
    X_val.to_parquet(X_val_path, index=False)
    X_test.to_parquet(X_test_path, index=False)

    # Persist transformers and metadata
    meta = {
        "seed": args.seed,
        "imputer": args.imputer,
        "iterative_estimator": args.iterative_estimator if args.imputer == "iterative" else None,
        "iterative_max_iter": args.iterative_max_iter if args.imputer == "iterative" else None,
        "scaler": args.scaler,
        "no_scale_binary": args.no_scale_binary,
        "add_missing_indicators": args.add_missing_indicators,
        "float32": args.float32,
        "dropped_all_nan_cols": dropped_all_nan,
        "dropped_zero_var_cols": dropped_zero_var,
        "kept_columns_order": keep_cols,
        "train_shape": list(X_train.shape),
        "val_shape": list(X_val.shape),
        "test_shape": list(X_test.shape),
        "generated_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    joblib.dump(
        {
            "imputer": imputer,
            "scaler": scaler,
            "meta": meta,
        },
        outdir / "transforms.joblib"
    )

    # Human summary
    md = []
    md.append(f"# Transforms Summary\n\nGenerated: {meta['generated_utc']}\n\n")
    md.append(f"Imputer: **{args.imputer}**")
    if args.imputer == "iterative":
        md.append(f" (estimator: {args.iterative_estimator}, max_iter: {args.iterative_max_iter})")
    md.append("\n")
    md.append(f"Scaler: **{args.scaler}** (no_scale_binary={args.no_scale_binary})\n\n")
    md.append(f"Add missing indicators: {args.add_missing_indicators}\n\n")
    md.append(f"Dropped (all-NaN on train): {len(dropped_all_nan)}\n")
    md.append(f"Dropped (zero-variance on train): {len(dropped_zero_var)}\n\n")
    md.append(f"Kept feature count: {len(keep_cols)}\n")
    (outdir / "TRANSFORMS.md").write_text("".join(md))

    # Hashes for integrity
    hashes = {
        "kept_cols_hash32": _hash_columns(keep_cols),
        "train_cols_hash32": _hash_columns(list(X_train.columns)),
        "val_cols_hash32": _hash_columns(list(X_val.columns)),
        "test_cols_hash32": _hash_columns(list(X_test.columns)),
        "n_kept_cols": len(keep_cols),
        "n_train_rows": len(X_train),
        "n_val_rows": len(X_val),
        "n_test_rows": len(X_test),
    }
    (outdir / "transforms_hashes.json").write_text(json.dumps(hashes, indent=2))

    # Console
    print(f"Wrote: {X_train_path}")
    print(f"Wrote: {X_val_path}")
    print(f"Wrote: {X_test_path}")
    print(f"Wrote: {outdir / 'transforms.joblib'}")
    print(f"Wrote: {outdir / 'TRANSFORMS.md'}")
    print(f"Wrote: {outdir / 'transforms_hashes.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

