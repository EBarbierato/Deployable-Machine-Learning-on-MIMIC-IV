#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

def is_diab(code:str, ver:int)->bool:
    if not isinstance(code,str): return False
    c=code.upper()
    return (ver==9 and c.startswith("250")) or (ver==10 and c[:3] in {"E10","E11","E13","E14"})

def is_cvd(code:str, ver:int)->bool:
    if not isinstance(code,str): return False
    c=code.upper()
    if ver==9 and c[:3].isdigit():
        k=int(c[:3]); return (410<=k<=414) or k in {428,433,434,436}
    if ver==10:
        return c.startswith(("I20","I21","I22","I23","I24","I25","I50","I63","I64"))
    return False

def main()->int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--raw-root", required=True)
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--horizon-days", type=int, default=365)
    args=ap.parse_args()

    raw=Path(args.raw_root)
    cohort=pd.read_parquet(args.cohort)[["subject_id","index_time","index_hadm_id"]]
    admissions=pd.read_parquet(raw/"admissions")[["hadm_id","subject_id","admittime"]]
    admissions["admittime"]=pd.to_datetime(admissions["admittime"],errors="coerce")
    dx=pd.read_parquet(raw/"diagnoses_icd")[["subject_id","hadm_id","icd_code","icd_version"]]
    dx=dx.merge(admissions,on=["hadm_id","subject_id"],how="left").rename(columns={"admittime":"dx_time"})
    cohort["horizon_to"]=pd.to_datetime(cohort["index_time"])+pd.Timedelta(days=args.horizon_days)
    dx=dx.merge(cohort[["subject_id","index_time","horizon_to"]],on="subject_id",how="left")
    m=(dx["dx_time"]>=dx["index_time"]) & (dx["dx_time"]<=dx["horizon_to"])
    w=dx.loc[m].copy()
    w["diab"]=[is_diab(c,int(v)) for c,v in zip(w.icd_code,w.icd_version)]
    w["cvd"] =[is_cvd(c,int(v))  for c,v in zip(w.icd_code,w.icd_version)]
    lab=w.groupby("subject_id").agg(diab_incident_365d=("diab","any"), cvd_incident_365d=("cvd","any")).reset_index()
    lab=lab.astype({"diab_incident_365d":int,"cvd_incident_365d":int})
    labels=cohort[["subject_id"]].merge(lab,on="subject_id",how="left").fillna(0).astype({"diab_incident_365d":int,"cvd_incident_365d":int})
    Path(args.out).parent.mkdir(parents=True,exist_ok=True)
    labels.to_parquet(args.out,index=False)
    print(f"[labels] wrote {args.out} rows={len(labels)}")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
