#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python3
"""
06_train_baselines.py

Purpose
-------
Train baseline single-task classifiers on leakage-safe, preprocessed data:
- Logistic Regression (class_weight balanced)
- Random Forest (class_weight balanced)
- XGBoost (if available; scale_pos_weight; early stopping)

Model selection is done on the validation set using a small hyperparameter grid
and AUPRC (default) or AUROC as the selection metric. The script saves:
  - Trained model (.joblib)
  - Validation and test predictions (parquet)
  - Metrics (JSON + Markdown)
  - Coefficients / feature importances (CSV when available)

Inputs
------
- X_train.parquet, X_val.parquet, X_test.parquet from 05_impute_and_scale.py
- labels.parquet from 02_define_labels.py
- A label column to train (e.g., diab_incident_365d)

Outputs
-------
- {outdir}/{label}/{model}/model.joblib
- {outdir}/{label}/{model}/val_preds.parquet
- {outdir}/{label}/{model}/test_preds.parquet
- {outdir}/{label}/{model}/metrics.json
- {outdir}/{label}/{model}/METRICS.md
- {outdir}/{label}/{model}/feature_importance.csv (when applicable)

Example
-------
python scripts/06_train_baselines.py \
  --x-train artifacts/imputed/X_train.parquet \
  --x-val   artifacts/imputed/X_val.parquet \
  --x-test  artifacts/imputed/X_test.parquet \
  --labels  artifacts/labels/labels.parquet \
  --label-col diab_incident_365d \
  --models logistic,random_forest,xgboost \
  --select-metric auprc \
  --seed 42 \
  --outdir artifacts/baselines
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss, log_loss
)
from sklearn.utils.class_weight import compute_class_weight

# Optional xgboost
try:
    import xgboost as xgb  # type: ignore
    _HAVE_XGB = True
except Exception:
    xgb = None  # type: ignore
    _HAVE_XGB = False


# ----------------------------- CLI ------------------------------------------ #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train baseline models (logistic, RF, XGBoost) on a specified label.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--x-train", type=str, required=True)
    p.add_argument("--x-val",   type=str, required=True)
    p.add_argument("--x-test",  type=str, required=True)
    p.add_argument("--labels",  type=str, required=True,
                   help="labels.parquet from 02_define_labels.py")
    p.add_argument("--label-col", type=str, required=True,
                   help="Name of the boolean label column to train, e.g., diab_incident_365d")
    p.add_argument("--models", type=str, default="logistic,random_forest,xgboost",
                   help="Comma-separated subset of: logistic,random_forest,xgboost")
    p.add_argument("--select-metric", type=str, choices=["auprc", "auroc"], default="auprc",
                   help="Metric used to pick the best hyperparameters on validation set.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--outdir", type=str, default="artifacts/baselines")
    return p.parse_args()


# ----------------------------- Utilities ------------------------------------ #

@dataclass
class Dataset:
    X: pd.DataFrame  # includes subject_id then features
    y: pd.Series     # aligned to X rows
    id_col: str = "subject_id"

def _load_matrix(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)

def _load_label_map(labels_path: Path, label_col: str) -> pd.DataFrame:
    lab = pd.read_parquet(labels_path)
    if "subject_id" not in lab.columns:
        raise KeyError("labels file must contain 'subject_id'.")
    if label_col not in lab.columns:
        raise KeyError(f"labels file missing requested column '{label_col}'.")
    y = lab[["subject_id", label_col]].drop_duplicates("subject_id")
    y[label_col] = y[label_col].astype(bool).astype(int)
    return y

def _join_Xy(Xdf: pd.DataFrame, ymap: pd.DataFrame, label_col: str) -> Dataset:
    if "subject_id" not in Xdf.columns:
        raise KeyError("Design matrix must contain 'subject_id'.")
    df = Xdf.merge(ymap, on="subject_id", how="inner")
    if df.empty:
        raise ValueError("Join of features and labels is empty. Check subject_id alignment.")
    y = df[label_col].astype(int)
    X = df.drop(columns=[label_col])
    return Dataset(X=X, y=y)

def _features_only(df_with_id: pd.DataFrame, id_col: str = "subject_id") -> np.ndarray:
    cols = [c for c in df_with_id.columns if c != id_col]
    return df_with_id[cols].values, cols

def _ensure_binary(y: pd.Series) -> None:
    vals = set(pd.unique(y))
    if not vals.issubset({0, 1}):
        raise ValueError(f"Label is not binary 0/1. Found values: {sorted(list(vals))}")

def _class_weights(y: pd.Series) -> Dict[int, float]:
    classes = np.array([0, 1])
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    return {int(c): float(w) for c, w in zip(classes, weights)}

def _pos_rate(y: pd.Series) -> float:
    return float(y.mean())

def _metrics(y_true: np.ndarray, p: np.ndarray) -> Dict[str, float]:
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

def _write_metrics_md(path: Path, label_col: str, model_name: str,
                      m_val: Dict[str, float], m_test: Dict[str, float],
                      n_val: int, n_test: int, pos_val: float, pos_test: float) -> None:
    lines = []
    lines.append(f"# Baseline metrics: {model_name}\n\n")
    lines.append(f"Label: **{label_col}**\n\n")
    lines.append(f"Validation (n={n_val}, pos_rate={pos_val:.4f})\n\n")
    lines.append(f"- AUROC: {m_val.get('auroc', float('nan')):.4f}\n")
    lines.append(f"- AUPRC: {m_val.get('auprc', float('nan')):.4f}\n")
    lines.append(f"- Brier: {m_val.get('brier', float('nan')):.4f}\n")
    lines.append(f"- LogLoss: {m_val.get('logloss', float('nan')):.4f}\n\n")
    lines.append(f"Test (n={n_test}, pos_rate={pos_test:.4f})\n\n")
    lines.append(f"- AUROC: {m_test.get('auroc', float('nan')):.4f}\n")
    lines.append(f"- AUPRC: {m_test.get('auprc', float('nan')):.4f}\n")
    lines.append(f"- Brier: {m_test.get('brier', float('nan')):.4f}\n")
    lines.append(f"- LogLoss: {m_test.get('logloss', float('nan')):.4f}\n")
    path.write_text("".join(lines))


# ----------------------------- Training Loops ------------------------------- #

def train_logistic(ds_tr: Dataset, ds_va: Dataset, seed: int, select_metric: str
                   ) -> Tuple[LogisticRegression, Dict[str, float]]:
    cw = _class_weights(ds_tr.y)
    Xtr, cols = _features_only(ds_tr.X)
    Xva, _ = _features_only(ds_va.X)

    # Small grid over C and penalty
    grid = [
        {"penalty": "l2", "C": c, "solver": "liblinear", "class_weight": cw, "random_state": seed}
        for c in [0.01, 0.1, 1.0, 10.0]
    ] + [
        {"penalty": "l1", "C": c, "solver": "liblinear", "class_weight": cw, "random_state": seed}
        for c in [0.01, 0.1, 1.0]
    ]

    best_m = -np.inf
    best_model = None
    for prm in grid:
        m = LogisticRegression(**prm, max_iter=2000)
        m.fit(Xtr, ds_tr.y.values)
        pva = m.predict_proba(Xva)[:, 1]
        score = average_precision_score(ds_va.y, pva) if select_metric == "auprc" else roc_auc_score(ds_va.y, pva)
        if score > best_m:
            best_m = score
            best_model = m
    if best_model is None:
        raise RuntimeError("Failed to fit any LogisticRegression model.")
    return best_model, {"val_score": float(best_m), "metric": select_metric, "features": cols}

def train_random_forest(ds_tr: Dataset, ds_va: Dataset, seed: int, select_metric: str
                        ) -> Tuple[RandomForestClassifier, Dict[str, float]]:
    cw = "balanced"
    Xtr, cols = _features_only(ds_tr.X)
    Xva, _ = _features_only(ds_va.X)

    grid = [
        {"n_estimators": n, "max_depth": d, "min_samples_leaf": m, "random_state": seed,
         "n_jobs": -1, "class_weight": cw, "max_features": "sqrt"}
        for n in [500, 1000]
        for d in [None, 6, 10]
        for m in [1, 5]
    ]

    best_m = -np.inf
    best_model = None
    for prm in grid:
        m = RandomForestClassifier(**prm)
        m.fit(Xtr, ds_tr.y.values)
        pva = m.predict_proba(Xva)[:, 1]
        score = average_precision_score(ds_va.y, pva) if select_metric == "auprc" else roc_auc_score(ds_va.y, pva)
        if score > best_m:
            best_m = score
            best_model = m
    if best_model is None:
        raise RuntimeError("Failed to fit any RandomForest model.")
    return best_model, {"val_score": float(best_m), "metric": select_metric, "features": cols}

def _xgb_tree_method() -> str:
    # Prefer GPU if likely available; otherwise 'hist'
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        return "gpu_hist"
    return "hist"

def train_xgboost(ds_tr: Dataset, ds_va: Dataset, seed: int, select_metric: str
                  ) -> Tuple["xgb.XGBClassifier", Dict[str, float]]:
    if not _HAVE_XGB:
        raise RuntimeError("XGBoost is not installed.")

    Xtr, cols = _features_only(ds_tr.X)
    Xva, _ = _features_only(ds_va.X)
    ytr = ds_tr.y.values
    yva = ds_va.y.values

    # scale_pos_weight = neg/pos on TRAIN
    pos = float(ytr.sum())
    neg = float(len(ytr) - ytr.sum())
    spw = (neg / max(1.0, pos)) if pos > 0 else 1.0

    base = dict(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method=_xgb_tree_method(),
        subsample=1.0,
        colsample_bytree=1.0,
        reg_lambda=1.0,
        random_state=seed,
        n_estimators=1000,
        early_stopping_rounds=50,
        scale_pos_weight=spw,
        n_jobs=0,  # let xgb decide threads
        verbosity=0,
    )
    grid = [
        dict(base, learning_rate=eta, max_depth=md, subsample=sub, colsample_bytree=col)
        for eta in [0.01, 0.05, 0.1]
        for md in [3, 5, 7]
        for sub in [0.8, 1.0]
        for col in [0.6, 0.8, 1.0]
    ]

    best_m = -np.inf
    best_model = None
    dval = [(Xva, yva)]
    for prm in grid:
        model = xgb.XGBClassifier(**prm)
        model.fit(Xtr, ytr, eval_set=dval, verbose=False)
        pva = model.predict_proba(Xva)[:, 1]
        score = average_precision_score(yva, pva) if select_metric == "auprc" else roc_auc_score(yva, pva)
        if score > best_m:
            best_m = score
            best_model = model

    if best_model is None:
        raise RuntimeError("Failed to fit any XGBoost model.")
    return best_model, {"val_score": float(best_m), "metric": select_metric, "features": cols, "scale_pos_weight": spw}


# ----------------------------- Orchestration -------------------------------- #

def main() -> int:
    args = parse_args()
    outroot = Path(args.outdir)
    outroot.mkdir(parents=True, exist_ok=True)

    # Load features
    Xtr = _load_matrix(Path(args.x_train))
    Xva = _load_matrix(Path(args.x_val))
    Xte = _load_matrix(Path(args.x_test))

    # Load labels and join
    ymap = _load_label_map(Path(args.labels), args.label_col)
    ds_tr = _join_Xy(Xtr, ymap, args.label_col)
    ds_va = _join_Xy(Xva, ymap, args.label_col)
    ds_te = _join_Xy(Xte, ymap, args.label_col)
    _ensure_binary(ds_tr.y); _ensure_binary(ds_va.y); _ensure_binary(ds_te.y)

    # Selection metric
    select_metric = args.select_metric

    # Which models
    req = [m.strip().lower() for m in args.models.split(",") if m.strip()]
    allowed = {"logistic", "random_forest", "xgboost"}
    models = [m for m in req if m in allowed]
    if not models:
        raise ValueError(f"No valid models requested. Choose subset of {sorted(list(allowed))}.")

    # Train each model
    report_index: Dict[str, Dict[str, float]] = {}
    for model_name in models:
        label_dir = outroot / args.label_col / model_name
        label_dir.mkdir(parents=True, exist_ok=True)

        if model_name == "logistic":
            model, aux = train_logistic(ds_tr, ds_va, args.seed, select_metric)
        elif model_name == "random_forest":
            model, aux = train_random_forest(ds_tr, ds_va, args.seed, select_metric)
        elif model_name == "xgboost":
            if not _HAVE_XGB:
                print("Skipping XGBoost: package not installed.")
                continue
            model, aux = train_xgboost(ds_tr, ds_va, args.seed, select_metric)
        else:
            continue

        # Save model
        joblib.dump(model, label_dir / "model.joblib")

        # Predictions and metrics
        def preds_metrics(ds: Dataset) -> Tuple[pd.DataFrame, Dict[str, float]]:
            Xmat, feat_cols = _features_only(ds.X)
            prob = model.predict_proba(Xmat)[:, 1]
            m = _metrics(ds.y.values, prob)
            df = pd.DataFrame({
                "subject_id": ds.X["subject_id"].values,
                "y_true": ds.y.values.astype(int),
                "y_score": prob.astype(np.float32)
            })
            return df, m

        val_df, m_val = preds_metrics(ds_va)
        test_df, m_test = preds_metrics(ds_te)

        # Persist predictions
        val_df.to_parquet(label_dir / "val_preds.parquet", index=False)
        test_df.to_parquet(label_dir / "test_preds.parquet", index=False)

        # Feature importance / coefficients
        try:
            if hasattr(model, "coef_"):
                coef = np.ravel(model.coef_)
                _, feat_cols = _features_only(ds_tr.X)
                imp = pd.DataFrame({"feature": feat_cols, "coefficient": coef})
                imp.to_csv(label_dir / "feature_importance.csv", index=False)
            elif hasattr(model, "feature_importances_"):
                _, feat_cols = _features_only(ds_tr.X)
                imp = pd.DataFrame({"feature": feat_cols, "importance": model.feature_importances_})
                imp.to_csv(label_dir / "feature_importance.csv", index=False)
            elif _HAVE_XGB and isinstance(model, xgb.XGBClassifier):
                _, feat_cols = _features_only(ds_tr.X)
                # Gain-based importance for stability
                fmap = {f"f{i}": feat_cols[i] for i in range(len(feat_cols))}
                booster = model.get_booster()
                score_map = booster.get_score(importance_type="gain")
                rows = []
                for k, v in score_map.items():
                    fname = fmap.get(k, k)
                    rows.append({"feature": fname, "gain": v})
                imp = pd.DataFrame(rows).sort_values("gain", ascending=False)
                imp.to_csv(label_dir / "feature_importance.csv", index=False)
        except Exception as e:
            # Non-fatal
            (label_dir / "feature_importance_error.txt").write_text(str(e))

        # Save metrics
        payload = {
            "label": args.label_col,
            "model": model_name,
            "select_metric": select_metric,
            "selection_val_score": aux.get("val_score"),
            "train_pos_rate": _pos_rate(ds_tr.y),
            "val_pos_rate": _pos_rate(ds_va.y),
            "test_pos_rate": _pos_rate(ds_te.y),
            "val": m_val,
            "test": m_test,
            "generated_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        }
        if "scale_pos_weight" in aux:
            payload["scale_pos_weight"] = aux["scale_pos_weight"]
        (label_dir / "metrics.json").write_text(json.dumps(payload, indent=2))

        _write_metrics_md(
            label_dir / "METRICS.md",
            args.label_col, model_name,
            m_val, m_test,
            n_val=len(val_df), n_test=len(test_df),
            pos_val=_pos_rate(ds_va.y), pos_test=_pos_rate(ds_te.y)
        )

        # Index for console
        report_index[model_name] = {
            "val_" + select_metric: aux.get("val_score", float("nan")),
            "test_auroc": m_test.get("auroc", float("nan")),
            "test_auprc": m_test.get("auprc", float("nan")),
        }

        print(f"Model {model_name} | val {select_metric}={aux.get('val_score'):.4f} | "
              f"test AUROC={m_test.get('auroc', float('nan')):.4f} AUPRC={m_test.get('auprc', float('nan')):.4f}")

    # Summary index JSON at root/label
    label_root = outroot / args.label_col
    label_root.mkdir(parents=True, exist_ok=True)
    (label_root / "models_summary.json").write_text(json.dumps(report_index, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

