#!/usr/bin/env python3
# coding: utf-8
"""
03_build_features.py  (updated)

Purpose
-------
Produce leakage-safe, patient-level features using only data strictly prior to each
patient's index_time. Adds:
  • Recency features: days since last lab / vital / prescription
  • Density features: counts normalized by lookback length
  • Medication recent-exposure windows (30d/90d) per drug class
  • Optional winsorization of continuous aggregates to improve robustness

Inputs
------
1) Cohort from 01_extract_cohort.py with:
   subject_id, hadm_id (optional), index_time, feature_time_upper_bound, age_at_admit, gender
2) MIMIC-IV tables via --mimic-root or explicit files:
   - hosp/labevents, hosp/d_labitems
   - hosp/prescriptions
   - icu/chartevents (optional; large)

Leakage policy
--------------
All source rows must satisfy event_time < feature_time_upper_bound.
For labs:   event_time = charttime
For meds:   event_time = starttime
For vitals: event_time = charttime

Outputs
-------
- {outdir}/features.parquet
- {outdir}/FEATURES.md
- {outdir}/feature_schema.json
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ----------------------------- CLI ------------------------------------------ #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build leakage-safe tabular features prior to index_time (with recency & winsorization).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--cohort-file", type=str, required=True,
                   help="Parquet from 01_extract_cohort.py")
    p.add_argument("--mimic-root", type=str, default=None,
                   help="Root of MIMIC-IV (expects hosp/* and optionally icu/*)")
    p.add_argument("--labevents-file", type=str, default=None)
    p.add_argument("--d-labitems-file", type=str, default=None)
    p.add_argument("--prescriptions-file", type=str, default=None)
    p.add_argument("--chartevents-file", type=str, default=None)

    p.add_argument("--include-labs", type=str2bool, default=True)
    p.add_argument("--include-meds", type=str2bool, default=True)
    p.add_argument("--include-vitals", type=str2bool, default=False,
                   help="Chartevents is very large; enable only if you have filtered extracts.")
    p.add_argument("--lookback-days", type=int, default=365,
                   help="Only include events whose timestamp is within this window before index_time.")
    p.add_argument("--labs-by-label-regex", type=str, default="",
                   help="Comma-separated regex labels to keep from d_labitems.label (e.g., 'glucose,creatinine,wbc').")

    # New options
    p.add_argument("--winsorize-continuous", type=str2bool, default=True,
                   help="Winsorize continuous aggregates (min/max/mean/last) for robustness.")
    p.add_argument("--winsor-lower", type=float, default=0.005, help="Lower tail for winsorization (0-0.5).")
    p.add_argument("--winsor-upper", type=float, default=0.995, help="Upper tail for winsorization (0.5-1).")
    p.add_argument("--recent-windows", type=str, default="30,90",
                   help="Comma-separated day windows for recent medication exposure (e.g., '30,90').")

    p.add_argument("--outdir", type=str, default="artifacts/features")
    p.add_argument("--nrows", type=int, default=None,
                   help="Optional row cap for fast dry runs (applies to source tables).")
    return p.parse_args()

def str2bool(v: str) -> bool:
    return str(v).lower() in {"1", "true", "t", "yes", "y"}


# ----------------------------- IO ------------------------------------------- #

def read_auto(path: Path, nrows: Optional[int] = None) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, nrows=nrows, low_memory=False)

def resolve_paths(args: argparse.Namespace) -> Dict[str, Optional[Path]]:
    paths: Dict[str, Optional[Path]] = {
        "labevents": None, "d_labitems": None, "prescriptions": None, "chartevents": None
    }
    if args.mimic_root:
        root = Path(args.mimic_root)
        def first_exist(rel_list: List[str]) -> Optional[Path]:
            for rel in rel_list:
                p = root / rel
                if p.exists():
                    return p
            return None
        paths["labevents"]    = first_exist(["hosp/labevents.parquet","hosp/labevents.csv.gz","hosp/labevents.csv"])
        paths["d_labitems"]   = first_exist(["hosp/d_labitems.parquet","hosp/d_labitems.csv.gz","hosp/d_labitems.csv"])
        paths["prescriptions"]= first_exist(["hosp/prescriptions.parquet","hosp/prescriptions.csv.gz","hosp/prescriptions.csv"])
        paths["chartevents"]  = first_exist(["icu/chartevents.parquet","icu/chartevents.csv.gz","icu/chartevents.csv"])
    # explicit overrides
    if args.labevents_file:     paths["labevents"]     = Path(args.labevents_file)
    if args.d_l
abitems_file:    paths["d_labitems"]    = Path(args.d_labitems_file)
    if args.prescriptions_file: paths["prescriptions"] = Path(args.prescriptions_file)
    if args.chartevents_file:   paths["chartevents"]   = Path(args.chartevents_file)
    return paths


# -------------------------- Feature schema ---------------------------------- #

@dataclass
class FeatureMeta:
    name: str
    dtype: str
    source: str
    description: str
    units: Optional[str] = None
    lookback_days: Optional[int] = None
    time_window: str = "(-lookback, index_time)"
    agg: Optional[str] = None
    leakage_guard: str = "strictly before index_time"

def add_meta(meta: List[FeatureMeta], **kwargs) -> None:
    meta.append(FeatureMeta(**kwargs))


# -------------------------- Core utilities ---------------------------------- #

def enforce_pre_index(joined: pd.DataFrame, event_col: str, upper_bound_col: str) -> pd.DataFrame:
    m = (joined[event_col] < joined[upper_bound_col])
    return joined[m]

def within_lookback(joined: pd.DataFrame, event_col: str, index_col: str, lookback_days: int) -> pd.DataFrame:
    lower = joined[index_col] - pd.to_datetime(pd.Timedelta(days=lookback_days))
    m = (joined[event_col] >= lower)
    return joined[m]

def agg_stats(x: pd.Series) -> Dict[str, float]:
    return {
        "min": float(np.nanmin(x)) if len(x) else np.nan,
        "max": float(np.nanmax(x)) if len(x) else np.nan,
        "mean": float(np.nanmean(x)) if len(x) else np.nan,
        "last": float(x.iloc[-1]) if len(x) else np.nan,
        "count": float(x.notna().sum())
    }

def regex_set(pattern_csv: str) -> List[re.Pattern]:
    items = [s.strip() for s in pattern_csv.split(",") if s.strip()]
    return [re.compile(s, flags=re.IGNORECASE) for s in items]

def sanitize(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "_", str(s).strip())
    s = re.sub(r"_+", "_", s).strip("_")
    return s.lower()

def winsorize_series(s: pd.Series, lo: float, hi: float) -> pd.Series:
    """Clip to empirical quantiles (lo, hi)."""
    if s.dropna().empty:
        return s
    lo_q, hi_q = s.quantile([lo, hi])
    return s.clip(lower=lo_q, upper=hi_q)


# -------------------------- LAB FEATURES ------------------------------------ #

def build_lab_features(cohort: pd.DataFrame,
                       labevents: pd.DataFrame,
                       d_labitems: Optional[pd.DataFrame],
                       lookback_days: int,
                       label_regexes: List[re.Pattern],
                       meta: List[FeatureMeta],
                       winsor: bool,
                       w_lo: float,
                       w_hi: float) -> pd.DataFrame:
    if labevents is None or labevents.empty:
        return pd.DataFrame({"subject_id": cohort["subject_id"].values})

    labs = labevents.copy()
    need = {"subject_id", "hadm_id", "charttime", "itemid", "valuenum"}
    missing = need - set(labs.columns)
    if missing:
        raise KeyError(f"labevents missing columns: {missing}")
    labs["charttime"] = pd.to_datetime(labs["charttime"], errors="coerce")
    labs = labs.dropna(subset=["subject_id", "hadm_id", "charttime"])
    # Attach labels if available
    if d_labitems is not None and not d_labitems.empty and "itemid" in d_labitems.columns:
        labs = labs.merge(d_labitems[["itemid", "label", "unitname"]].drop_duplicates("itemid"),
                          on="itemid", how="left")
    else:
        labs["label"] = labs["itemid"].astype(str)
        labs["unitname"] = None

    if label_regexes:
        def keep_lbl(s: str) -> bool:
            s = str(s) if s is not None else ""
            return any(rx.search(s) for rx in label_regexes)
        labs = labs[labs["label"].map(keep_lbl)]

    base = cohort[["subject_id", "index_time", "feature_time_upper_bound"]].copy()
    j = labs.merge(base, on="subject_id", how="inner")
    j = j.dropna(subset=["index_time", "feature_time_upper_bound"])

    j = enforce_pre_index(j, "charttime", "feature_time_upper_bound")
    j = within_lookback(j, "charttime", "index_time", lookback_days)
    if j.empty:
        return pd.DataFrame({"subject_id": cohort["subject_id"].values})

    j = j.sort_values(["subject_id", "label", "charttime"])

    # Aggregate per subject, per label
    aggs = []
    rec = []  # recency in days since last measurement (per label)
    for (sid, lbl), df in j.groupby(["subject_id", "label"], sort=False):
        s = df["valuenum"].astype(float)
        stats = agg_stats(s)
        stats["subject_id"] = sid
        stats["label"] = lbl
        stats["unit"] = df["unitname"].ffill().bfill().iloc[-1] if "unitname" in df.columns else None
        aggs.append(stats)
        # Recency
        idx_time = df["index_time"].iloc[-1]
        last_time = df["charttime"].iloc[-1]
        rec.append({"subject_id": sid, "label": lbl,
                    "recency_days": float((idx_time - last_time).total_seconds() / 86400.0)})
    A = pd.DataFrame(aggs)
    R = pd.DataFrame(rec)

    # Pivot to wide stats
    pieces = []
    cont_cols = []
    for stat in ["min", "max", "mean", "last", "count"]:
        wide = A.pivot_table(index="subject_id", columns="label", values=stat)
        wide.columns = [f"lab_{sanitize(lbl)}_{stat}" for lbl in wide.columns]
        # Winsorize continuous aggregates if requested (skip counts)
        if winsor and stat in {"min","max","mean","last"}:
            for c in wide.columns:
                wide[c] = winsorize_series(wide[c], w_lo, w_hi)
                cont_cols.append(c)
        pieces.append(wide)
    lab_wide = pd.concat(pieces, axis=1).reset_index()

    # Recency wide
    rec_wide = R.pivot_table(index="subject_id", columns="label", values="recency_days")
    if not rec_wide.empty:
        rec_wide.columns = [f"lab_{sanitize(lbl)}_recency_days" for lbl in rec_wide.columns]
        lab_wide = lab_wide.merge(rec_wide.reset_index(), on="subject_id", how="left")
        for c in rec_wide.columns:
            add_meta(meta,
                name=c, dtype="float64", source="hosp.labevents",
                description=f"Days since last {c.replace('lab_','').replace('_recency_days','')} measurement at index",
                units="days", lookback_days=lookback_days, agg="recency_days"
            )

    # Density features (counts per day of lookback)
    for c in [col for col in lab_wide.columns if col.endswith("_count")]:
        dens = c.replace("_count", "_per_day")
        lab_wide[dens] = lab_wide[c] / float(max(1, lookback_days))
        add_meta(meta, name=dens, dtype="float64", source="hosp.labevents",
                 description=f"{c} normalized by lookback_days", units="count/day",
                 lookback_days=lookback_days, agg="count_per_day")

    # Missingness indicators
    miss_cols = [c for c in lab_wide.columns if c.startswith("lab_") and c.endswith("_last")]
    for c in miss_cols:
        ind = c.replace("_last", "_missing")
        lab_wide[ind] = lab_wide[c].isna().astype(int)
        add_meta(meta,
            name=ind, dtype="int8", source="hosp.labevents",
            description=f"Missing indicator for {ind.replace('lab_','').replace('_missing','')} in lookback window",
            units=None, lookback_days=lookback_days, agg="missing_flag"
        )

    # Update meta for aggregates
    if not A.empty:
        for lbl, grp in A.groupby("label"):
            unit = str(grp["unit"].dropna().unique()[0]) if grp["unit"].notna().any() else None
            for stat in ["min", "max", "mean", "last", "count"]:
                add_meta(meta,
                    name=f"lab_{sanitize(lbl)}_{stat}",
                    dtype="float64" if stat != "count" else "float64",
                    source="hosp.labevents",
                    description=f"{lbl} ({stat}) over last {lookback_days} days",
                    units=unit if stat != "count" else None,
                    lookback_days=lookback_days,
                    agg=stat
                )

    return lab_wide


# -------------------------- MED FEATURES ------------------------------------ #

_DRUG_CLASSES = {
    "insulin": r"\binsulin\b",
    "metformin": r"\bmetformin\b",
    "sulfonylurea": r"glyburide|glipizide|glimepiride",
    "statin": r"atorvastatin|rosuvastatin|simvastatin|pravastatin|lovastatin|pitavastatin|fluvastatin",
    "beta_blocker": r"metoprolol|carvedilol|atenolol|bisoprolol|propranolol|labetalol|nebivolol",
    "ace_inhibitor": r"\b(lisinopril|enalapril|captopril|ramipril|benazepril|quinapril|perindopril)\b",
    "arb": r"\b(losartan|valsartan|olmesartan|candesartan|irbesartan|telmisartan|azilsartan)\b",
    "diuretic": r"furosemide|bumetanide|torsemide|hydrochlorothiazide|chlorthalidone|indapamide|spironolactone|eplerenone",
}

def build_med_features(cohort: pd.DataFrame,
                       rx: pd.DataFrame,
                       lookback_days: int,
                       meta: List[FeatureMeta],
                       recent_windows: List[int]) -> pd.DataFrame:
    if rx is None or rx.empty:
        return pd.DataFrame({"subject_id": cohort["subject_id"].values})

    tbl = rx.copy()
    need = {"subject_id", "hadm_id", "drug", "starttime"}
    missing = need - set(tbl.columns)
    if missing:
        raise KeyError(f"prescriptions missing columns: {missing}")
    tbl["starttime"] = pd.to_datetime(tbl["starttime"], errors="coerce")
    tbl = tbl.dropna(subset=["subject_id", "hadm_id", "starttime"])

    base = cohort[["subject_id", "index_time", "feature_time_upper_bound"]]
    j = tbl.merge(base, on="subject_id", how="inner")
    j = j.dropna(subset=["index_time", "feature_time_upper_bound"])

    j = enforce_pre_index(j, "starttime", "feature_time_upper_bound")
    j = within_lookback(j, "starttime", "index_time", lookback_days)
    if j.empty:
        return pd.DataFrame({"subject_id": cohort["subject_id"].values})

    j["drug_lc"] = j["drug"].astype(str).str.lower()

    feats = j.groupby("subject_id").size().rename("med_any_count").to_frame().reset_index()
    add_meta(meta, name="med_any_count", dtype="int32", source="hosp.prescriptions",
             description=f"Any prescriptions count in last {lookback_days} days",
             lookback_days=lookback_days, agg="count")

    # Days since last prescription
    rec = j.groupby("subject_id")["starttime"].max().rename("med_last_start").reset_index()
    feats = feats.merge(rec, on="subject_id", how="left")
    feats["med_recency_days"] = (j[["subject_id","index_time"]].drop_duplicates("subject_id")
                                 .set_index("subject_id")["index_time"]
                                 .reindex(feats["subject_id"].values).reset_index(drop=True))
    feats["med_recency_days"] = (feats["med_recency_days"] - feats["med_last_start"]).dt.total_seconds()/86400.0
    feats.drop(columns=["med_last_start"], inplace=True)
    add_meta(meta, name="med_recency_days", dtype="float64", source="hosp.prescriptions",
             description="Days since last prescription start at index", units="days",
             lookback_days=lookback_days, agg="recency_days")

    # Class-specific counts and recent windows
    for cls, pattern in _DRUG_CLASSES.items():
        m = j["drug_lc"].str.contains(pattern, regex=True)
        g = j[m].groupby("subject_id").size().rename(f"med_{cls}_count")
        feats = feats.merge(g.to_frame().reset_index(), on="subject_id", how="left")
        feats[f"med_{cls}_count"] = feats[f"med_{cls}_count"].fillna(0).astype(int)
        add_meta(meta, name=f"med_{cls}_count", dtype="int32", source="hosp.prescriptions",
                 description=f"Count of {cls.replace('_',' ')} prescriptions in last {lookback_days} days",
                 lookback_days=lookback_days, agg="count")

        feats[f"med_{cls}_any"] = (feats[f"med_{cls}_count"] > 0).astype(int)
        add_meta(meta, name=f"med_{cls}_any", dtype="int8", source="hosp.prescriptions",
                 description=f"Any {cls.replace('_',' ')} exposure in last {lookback_days} days",
                 lookback_days=lookback_days, agg="any")

        # Recent windows
        for W in recent_windows:
            rec_flag = j[m].copy()
            lower = rec_flag["index_time"] - pd.to_timedelta(W, unit="D")
            in_win = (rec_flag["starttime"] >= lower)
            flag = (rec_flag[in_win].groupby("subject_id").size() > 0).rename(f"med_{cls}_recent_{W}d").astype(int)
            feats = feats.merge(flag.to_frame().reset_index(), on="subject_id", how="left")
            feats[f"med_{cls}_recent_{W}d"] = feats[f"med_{cls}_recent_{W}d"].fillna(0).astype(int)
            add_meta(meta, name=f"med_{cls}_recent_{W}d", dtype="int8", source="hosp.prescriptions",
                     description=f"Any {cls.replace('_',' ')} exposure within {W} days before index",
                     lookback_days=W, agg="recent_any")

    # Density
    feats["med_any_per_day"] = feats["med_any_count"] / float(max(1, lookback_days))
    add_meta(meta, name="med_any_per_day", dtype="float64", source="hosp.prescriptions",
             description="Any prescriptions per day over lookback", units="count/day",
             lookback_days=lookback_days, agg="count_per_day")

    return feats


# -------------------------- VITALS (optional) -------------------------------- #

_VITAL_LABELS = {
    "heartrate": r"heart rate|heartrate",
    "sbp": r"(systolic )?bp systolic|sbp|arterial line systolic",
    "dbp": r"(diastolic )?bp diastolic|dbp|arterial line diastolic",
    "spo2": r"spo2|oxygen saturation",
    "resp_rate": r"respiratory rate|resp rate",
    "temp": r"temperature|temp",
}

def build_vital_features(cohort: pd.DataFrame,
                         chartevents: pd.DataFrame,
                         lookback_days: int,
                         meta: List[FeatureMeta],
                         winsor: bool,
                         w_lo: float,
                         w_hi: float) -> pd.DataFrame:
    if chartevents is None or chartevents.empty:
        return pd.DataFrame({"subject_id": cohort["subject_id"].values})

    ce = chartevents.copy()
    need = {"subject_id", "hadm_id", "charttime", "itemid", "valuenum", "label"}
    missing = need - set(ce.columns)
    if missing:
        # Fallback: derive label from itemid if label missing
        if {"subject_id", "hadm_id", "charttime", "valuenum", "itemid"}.issubset(ce.columns):
            ce["label"] = ce["itemid"].astype(str)
        else:
            raise KeyError(f"chartevents missing columns: {missing}")

    ce["charttime"] = pd.to_datetime(ce["charttime"], errors="coerce")
    ce = ce.dropna(subset=["subject_id", "hadm_id", "charttime"])

    base = cohort[["subject_id", "index_time", "feature_time_upper_bound"]]
    j = ce.merge(base, on="subject_id", how="inner").dropna(subset=["index_time", "feature_time_upper_bound"])

    j = enforce_pre_index(j, "charttime", "feature_time_upper_bound")
    j = within_lookback(j, "charttime", "index_time", lookback_days)
    if j.empty:
        return pd.DataFrame({"subject_id": cohort["subject_id"].values})

    j = j.sort_values(["subject_id", "charttime"])

    out = pd.DataFrame({"subject_id": cohort["subject_id"].values})
    for vit, pattern in _VITAL_LABELS.items():
        m = j["label"].astype(str).str.lower().str.contains(pattern, regex=True)
        sub = j[m].groupby("subject_id")["valuenum"].agg(["min","max","mean","last"]).reset_index()
        sub.columns = ["subject_id"] + [f"vital_{vit}_{c}" for c in ["min","max","mean","last"]]
        if winsor:
            for c in sub.columns:
                if c.startswith("vital_") and any(c.endswith(s) for s in ["min","max","mean","last"]):
                    sub[c] = winsorize_series(sub[c], w_lo, w_hi)
        out = out.merge(sub, on="subject_id", how="left")
        # Recency
        last_time = j[m].groupby("subject_id")["charttime"].max().rename(f"vital_{vit}_last_time")
        tmp = pd.DataFrame({"subject_id": cohort["subject_id"].values}).merge(last_time.reset_index(),
                                                                              on="subject_id", how="left")
        idx = cohort[["subject_id","index_time"]].set_index("subject_id")["index_time"]
        tmp["vital_{vit}_recency_days".format(vit=vit)] = (idx.reindex(tmp["subject_id"]).reset_index(drop=True) - tmp[f"vital_{vit}_last_time"]).dt.total_seconds()/86400.0
        out = out.merge(tmp[["subject_id", f"vital_{vit}_recency_days"]], on="subject_id", how="left")
        for stat in ["min","max","mean","last"]:
            add_meta(meta, name=f"vital_{vit}_{stat}", dtype="float64", source="icu.chartevents",
                     description=f"{vit} ({stat}) over last {lookback_days} days",
                     units=None, lookback_days=lookback_days, agg=stat)
        add_meta(meta, name=f"vital_{vit}_recency_days", dtype="float64", source="icu.chartevents",
                 description=f"Days since last {vit} measurement at index", units="days",
                 lookback_days=lookback_days, agg="recency_days")
        # Missing flag
        out[f"vital_{vit}_missing"] = out[f"vital_{vit}_last"].isna().astype(int)
        add_meta(meta, name=f"vital_{vit}_missing", dtype="int8", source="icu.chartevents",
                 description=f"Missing indicator for {vit} in lookback window",
                 units=None, lookback_days=lookback_days, agg="missing_flag")

    return out


# ----------------------------- MAIN ----------------------------------------- #

def write_markdown_report(out_md: Path, feats: pd.DataFrame, meta: List[FeatureMeta],
                          lookback_days: int) -> None:
    n = len(feats)
    lines = []
    lines.append(f"# Features Data Sheet\n\nGenerated: {datetime.utcnow().isoformat(timespec='seconds')}Z\n")
    lines.append(f"Rows (patients): {n}\n\n")
    lines.append(f"Lookback window: last {lookback_days} days strictly before index_time\n\n")
    lines.append(f"Columns: {feats.shape[1] - 1} (excluding subject_id)\n\n")
    lines.append("## Head (first 5 rows)\n\n")
    lines.append(feats.head(5).to_markdown(index=False))
    out_md.write_text("".join(lines))

def main() -> int:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cohort = pd.read_parquet(args.cohort_file)
    need = {"subject_id", "index_time", "feature_time_upper_bound"}
    miss = need - set(cohort.columns)
    if miss:
        raise KeyError(f"Cohort missing required columns: {miss}")
    cohort["index_time"] = pd.to_datetime(cohort["index_time"], errors="coerce")
    cohort["feature_time_upper_bound"] = pd.to_datetime(cohort["feature_time_upper_bound"], errors="coerce")
    if cohort[["index_time","feature_time_upper_bound"]].isna().any().any():
        raise ValueError("Cohort contains NaT in index_time/feature_time_upper_bound.")

    paths = resolve_paths(args)
    meta: List[FeatureMeta] = []

    feats = cohort[["subject_id"]].copy()
    # Static demographics
    if "age_at_admit" in cohort.columns:
        feats = feats.merge(cohort[["subject_id","age_at_admit"]], on="subject_id", how="left")
        add_meta(meta, name="age_at_admit", dtype=str(cohort["age_at_admit"].dtype),
                 source="cohort", description="Age (years) at index admission")
    if "gender" in cohort.columns:
        g = cohort[["subject_id","gender"]].copy()
        g["gender"] = g["gender"].astype(str).str.upper().str.strip()
        for val in sorted(g["gender"].dropna().unique()):
            col = f"gender_{sanitize(val)}"
            feats = feats.merge((g.assign(**{col: (g["gender"]==val).astype(int)})[["subject_id", col]]),
                                on="subject_id", how="left")
            add_meta(meta, name=col, dtype="int8", source="cohort",
                     description=f"Gender indicator: {val}")

    lab_regexes = regex_set(args.labs_by_label_regex)

    # Labs
    if args.include_labs and paths["labevents"] is not None:
        labevents = read_auto(paths["labevents"], nrows=args.nrows)
        d_labitems = read_auto(paths["d_labitems"], nrows=None) if paths["d_labitems"] else None
        lab_wide = build_lab_features(cohort, labevents, d_labitems, args.lookback_days, lab_regexes, meta,
                                      winsor=args.winsorize_continuous, w_lo=args.winsor_lower, w_hi=args.winsor_upper)
        feats = feats.merge(lab_wide, on="subject_id", how="left")

    # Meds
    recent_windows = [int(s) for s in str(args.recent_windows).split(",") if str(s).strip().isdigit()]
    if args.include_meds and paths["prescriptions"] is not None:
        rx = read_auto(paths["prescriptions"], nrows=args.nrows)
        med_feats = build_med_features(cohort, rx, args.lookback_days, meta, recent_windows=recent_windows)
        feats = feats.merge(med_feats, on="subject_id", how="left")

    # Vitals (optional)
    if args.include_vitals and paths["chartevents"] is not None:
        ce = read_auto(paths["chartevents"], nrows=args.nrows)
        vital_feats = build_vital_features(cohort, ce, args.lookback_days, meta,
                                           winsor=args.winsorize_continuous, w_lo=args.winsor_lower, w_hi=args.winsor_upper)
        feats = feats.merge(vital_feats, on="subject_id", how="left")

    # Fill NaNs for counts/flags with 0; keep continuous as NaN where truly missing
    for c in feats.columns:
        if c == "subject_id":
            continue
        if c.endswith("_count") or c.endswith("_any") or c.endswith("_missing") or c.endswith("_recent_30d") or c.endswith("_recent_90d") or c.startswith("gender_"):
            feats[c] = feats[c].fillna(0).astype(int)

    # Density sanity: replace inf with nan (in case of weird lookback)
    feats.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Persist
    feats_path = outdir / "features.parquet"
    feats.to_parquet(feats_path, index=False)

    schema = [asdict(m) for m in meta]
    (outdir / "feature_schema.json").write_text(json.dumps(schema, indent=2))

    write_markdown_report(out_md=outdir / "FEATURES.md", feats=feats, meta=meta,
                          lookback_days=args.lookback_days)

    print(f"Wrote: {feats_path}")
    print(f"Wrote: {outdir / 'FEATURES.md'}")
    print(f"Wrote: {outdir / 'feature_schema.json'}")
    return 0


if __name__ == "__main__":
    def str2bool(v: str) -> bool:
        return str(v).lower() in {"1", "true", "t", "yes", "y"}
    raise SystemExit(main())
