#!/usr/bin/env python3
from __future__ import annotations
import argparse, joblib, pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np

def load_scores(model, X):
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        return proba[:, 1] if proba.ndim == 2 else proba
    if hasattr(model, "decision_function"):
        s = model.decision_function(X)
        return 1.0 / (1.0 + np.exp(-s))
    pred = model.predict(X)
    return pd.Series(pred, index=X.index, dtype=float)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--x-file",     required=True)
    ap.add_argument("--labels",     required=True)
    ap.add_argument("--label-col",  required=True)
    ap.add_argument("--out",        required=True)
    args = ap.parse_args()

    model = joblib.load(args.model_path)
    X = pd.read_parquet(args.x_file)
    y = pd.read_parquet(args.labels)[["subject_id", args.label_col]]

    df = X.merge(y, on="subject_id", how="left")
    feat = [c for c in df.columns if c not in {"subject_id", args.label_col}]
    df[feat] = df[feat].fillna(0)

    scores = load_scores(model, df[feat])
    out = pd.DataFrame({"subject_id": df["subject_id"],
                        "y_true":     df[args.label_col].astype(int).values,
                        "y_score":    scores})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)

    auroc = roc_auc_score(out.y_true, out.y_score)
    auprc = average_precision_score(out.y_true, out.y_score)
    print(f"[dump_preds] {args.out}  AUROC={auroc:.4f}  AUPRC={auprc:.4f}")

if __name__ == "__main__":
    main()
