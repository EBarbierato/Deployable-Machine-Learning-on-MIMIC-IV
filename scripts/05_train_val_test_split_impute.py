#!/usr/bin/env python3
from __future__ import annotations
import argparse, pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

def main()->int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--label-col", default="diab_incident_365d")
    ap.add_argument("--seed", type=int, default=42)
    args=ap.parse_args()

    out=Path(args.out_root); out.mkdir(parents=True, exist_ok=True)
    X=pd.read_parquet(args.features)
    y=pd.read_parquet(args.labels)[["subject_id", args.label_col]]
    df=X.merge(y,on="subject_id",how="inner")
    feat=[c for c in df.columns if c not in {"subject_id", args.label_col}]
    df[feat]=df[feat].fillna(0)

    sids=df["subject_id"].unique()
    tr, te = train_test_split(sids, test_size=0.2, random_state=args.seed, shuffle=True)
    tr, va = train_test_split(tr,   test_size=0.25, random_state=args.seed, shuffle=True)

    def dump(sids, name):
        d=df[df.subject_id.isin(sids)].copy()
        d[["subject_id"]+feat].to_parquet(out/f"X_{name}.parquet", index=False)
        print(f"[split] X_{name} rows={len(d)}")

    dump(tr,"train"); dump(va,"val"); dump(te,"test")
    print("[split] labels at:", args.labels)
    return 0

if __name__=="__main__":
    raise SystemExit(main())
