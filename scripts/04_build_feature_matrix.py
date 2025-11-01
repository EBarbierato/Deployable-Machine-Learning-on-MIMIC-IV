#!/usr/bin/env python3
from __future__ import annotations
import argparse, pandas as pd
from pathlib import Path

def simplify_eth(e):
    s=str(e).lower() if pd.notna(e) else ""
    if "white" in s: return "white"
    if "black" in s or "african" in s: return "black"
    if "asian" in s: return "asian"
    if "hispanic" in s or "latino" in s: return "hispanic"
    return "other"

def main()->int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--raw-root", required=True)
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--out", required=True)
    args=ap.parse_args()

    raw=Path(args.raw_root)
    cohort=pd.read_parquet(args.cohort)
    adm=pd.read_parquet(raw/"admissions")[["subject_id","hadm_id","admittime","dischtime"]]
    for c in ("admittime","dischtime"): adm[c]=pd.to_datetime(adm[c],errors="coerce")

    j=adm.merge(cohort[["subject_id","index_time"]], on="subject_id", how="inner")
    m=(j.admittime<j.index_time) & (j.admittime>=j.index_time-pd.Timedelta(days=365))
    prev=j.loc[m].groupby("subject_id").size().rename("prev_adm_365d").astype("int64")

    idx=adm.merge(cohort[["subject_id","index_hadm_id"]], left_on=["subject_id","hadm_id"], right_on=["subject_id","index_hadm_id"], how="inner")
    los=((idx.dischtime-idx.admittime).dt.total_seconds()/86400.0).rename("los_days_index")
    los.index=idx.subject_id

    dx=pd.read_parquet(raw/"diagnoses_icd")[["subject_id","hadm_id","icd_code","icd_version"]]
    dx=dx.merge(adm[["hadm_id","subject_id","admittime"]], on=["hadm_id","subject_id"], how="left").rename(columns={"admittime":"dx_time"})
    dx=dx.merge(cohort[["subject_id","index_time"]], on="subject_id", how="left")
    prior=dx.loc[dx.dx_time<dx.index_time].copy()
    def is_diab(c,v):
        if not isinstance(c,str): return False
        c=c.upper(); return (v==9 and c.startswith("250")) or (v==10 and c[:3] in {"E10","E11","E13","E14"})
    def is_cvd(c,v):
        if not isinstance(c,str): return False
        c=c.upper()
        return (v==9 and c[:3].isdigit() and (410<=int(c[:3])<=414 or int(c[:3]) in {428,433,434,436})) or \
               (v==10 and c.startswith(("I20","I21","I22","I23","I24","I25","I50","I63","I64")))
    prior["prior_diab"]=[is_diab(c,int(v)) for c,v in zip(prior.icd_code, prior.icd_version)]
    prior["prior_cvd"] =[is_cvd(c,int(v)) for c,v in zip(prior.icd_code, prior.icd_version)]
    pr=prior.groupby("subject_id").agg(prior_diab=("prior_diab","any"), prior_cvd=("prior_cvd","any")).astype(int)

    feats=cohort.set_index("subject_id")[["sex","age","ethnicity"]].copy()
    feats["sex_female"]=(feats["sex"].str.upper()=="F").astype(int)
    feats["eth_simplified"]=feats["ethnicity"].map(simplify_eth).fillna("other")
    for g in ["white","black","asian","hispanic","other"]:
        feats[f"eth_{g}"]=(feats["eth_simplified"]==g).astype(int)
    feats.drop(columns=["sex","ethnicity","eth_simplified"], inplace=True)
    feats["prev_adm_365d"]=prev.reindex(feats.index).fillna(0).astype(int)
    feats["los_days_index"]=los.reindex(feats.index).fillna(0.0)
    feats=feats.join(pr, how="left").fillna({"prior_diab":0,"prior_cvd":0}).astype({"prior_diab":int,"prior_cvd":int})

    feats.reset_index().to_parquet(args.out, index=False)
    print(f"[features] wrote {args.out} rows={len(feats)} cols={feats.shape[1]}")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
