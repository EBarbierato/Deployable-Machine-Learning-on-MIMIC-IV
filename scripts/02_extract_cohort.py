#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python3
"""


Purpose
-------
Create a leakage-safe patient-level cohort and index time from MIMIC data.

What it does
------------
- Loads patients and admissions (CSV or Parquet), either via explicit paths or a MIMIC root.
- Computes age at admission (supports MIMIC-IV 'anchor_age'/'anchor_year' or classic 'dob').
- Filters to adults (min_age), valid times, and an optional admission-year window.
- Selects ONE index admission per patient (earliest admission within the window).
- Emits a cohort table with index timestamps and audit fields.
- Writes a concise Data Sheet (Markdown) and stable ID hashes for later integrity checks.

Inputs (choose ONE style)
-------------------------
  a) --mimic-root /path/to/mimic
     expects:
       {root}/hosp/patients.csv(.gz|.parquet)
       {root}/hosp/admissions.csv(.gz|.parquet)
  b) --patients-file and --admissions-file as explicit files

Outputs
-------
  {outdir}/cohort.parquet
  {outdir}/COHORT.md
  {outdir}/cohort_hashes.json

Examples
--------
python scripts/01_extract_cohort.py \
  --mimic-root /data/mimiciv \
  --start-year 2008 --end-year 2019 \
  --min-age 18 \
  --outdir artifacts/cohort

python scripts/01_extract_cohort.py \
  --patients-file /data/mimiciv/hosp/patients.csv.gz \
  --admissions-file /data/mimiciv/hosp/admissions.csv.gz \
  --min-age 18 --outdir artifacts/cohort
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


# -------------------------- CLI -------------------------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Builds adult, first-admission cohort with index time.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--mimic-root", type=str, default=None,
                   help="Root folder of MIMIC (expects hosp/patients.* and hosp/admissions.*)")
    p.add_argument("--patients-file", type=str, default=None,
                   help="Path to patients file if not using --mimic-root")
    p.add_argument("--admissions-file", type=str, default=None,
                   help="Path to admissions file if not using --mimic-root")
    p.add_argument("--start-year", type=int, default=None,
                   help="Earliest admission year to consider (inclusive)")
    p.add_argument("--end-year", type=int, default=None,
                   help="Latest admission year to consider (inclusive)")
    p.add_argument("--min-age", type=int, default=18,
                   help="Minimum age at admission (years)")
    p.add_argument("--outdir", type=str, default="artifacts/cohort",
                   help="Directory to write cohort artifacts")
    p.add_argument("--nrows", type=int, default=None,
                   help="Optional row cap for dry runs / debugging")
    return p.parse_args()


# ----------------------- IO helpers ---------------------------------------- #

def _read_auto(path: Path, nrows: Optional[int] = None) -> pd.DataFrame:
    """Read CSV/CSV.GZ/Parquet by extension."""
    ext = path.suffix.lower()
    if ext == ".parquet":
        return pd.read_parquet(path)
    # handle .csv or .gz
    return pd.read_csv(path, nrows=nrows, low_memory=False)


def _resolve_paths(args: argparse.Namespace) -> Tuple[Path, Path]:
    if args.mimic_root:
        root = Path(args.mimic_root)
        # Prefer Parquet if present, else CSV(.gz)
        candidates = []
        for fname in ["patients.parquet", "patients.csv.gz", "patients.csv"]:
            p = root / "hosp" / fname
            if p.exists():
                candidates.append(p)
        for fname in ["admissions.parquet", "admissions.csv.gz", "admissions.csv"]:
            p = root / "hosp" / fname
            if p.exists():
                candidates.append(p)
        if not candidates:
            raise FileNotFoundError("No patients/admissions files found under {root}/hosp/")
        # pick best matches
        patients_file = next((root / "hosp" / f for f in ["patients.parquet", "patients.csv.gz", "patients.csv"]
                              if (root / "hosp" / f).exists()))
        admissions_file = next((root / "hosp" / f for f in ["admissions.parquet", "admissions.csv.gz", "admissions.csv"]
                                if (root / "hosp" / f).exists()))
        return patients_file, admissions_file

    if args.patients_file and args.admissions_file:
        return Path(args.patients_file), Path(args.admissions_file)

    raise ValueError("Provide either --mimic-root or both --patients-file and --admissions-file.")


# --------------------- Age computation ------------------------------------- #

def compute_age_at_admit(adm: pd.DataFrame, pat: pd.DataFrame) -> pd.Series:
    """
    Compute age at admission. Supports two schemas:

    1) MIMIC-IV style: patients has ['anchor_age', 'anchor_year'].
       age = anchor_age + (admit_year - anchor_year)

    2) Classic style: patients has 'dob'. age = floor((admittime - dob) / 365.2425 days)

    Returns a float Series (years). Caps unphysiologic ages to [0, 120] as a safety net.
    """
    # normalize datetime
    adm = adm.copy()
    adm["admittime"] = pd.to_datetime(adm["admittime"], errors="coerce")

    if {"anchor_age", "anchor_year"}.issubset(pat.columns):
        # merge minimal fields
        tmp = adm[["subject_id", "admittime"]].merge(
            pat[["subject_id", "anchor_age", "anchor_year"]], on="subject_id", how="left"
        )
        admit_year = tmp["admittime"].dt.year
        age = tmp["anchor_age"].astype(float) + (admit_year - tmp["anchor_year"].astype(float))
    elif "dob" in pat.columns:
        tmp = adm[["subject_id", "admittime"]].merge(
            pat[["subject_id", "dob"]], on="subject_id", how="left"
        )
        dob = pd.to_datetime(tmp["dob"], errors="coerce")
        delta_days = (tmp["admittime"] - dob).dt.total_seconds() / (3600 * 24)
        age = np.floor(delta_days / 365.2425)
    else:
        raise KeyError("patients table missing required columns: need anchor_age/anchor_year or dob")

    # safety caps
    age = age.clip(lower=0, upper=120)
    return age


# --------------------- Cohort logic ---------------------------------------- #

def build_cohort(adm: pd.DataFrame,
                 pat: pd.DataFrame,
                 min_age: int,
                 start_year: Optional[int],
                 end_year: Optional[int]) -> pd.DataFrame:
    df = adm.copy()

    # Basic column checks
    needed_adm = {"subject_id", "hadm_id", "admittime", "dischtime"}
    missing = needed_adm - set(df.columns)
    if missing:
        raise KeyError(f"admissions table missing: {missing}")

    # Timestamps and sanity
    df["admittime"] = pd.to_datetime(df["admittime"], errors="coerce")
    df["dischtime"] = pd.to_datetime(df["dischtime"], errors="coerce")
    df = df.dropna(subset=["subject_id", "hadm_id", "admittime", "dischtime"])

    # remove inverted times
    df = df[df["dischtime"] >= df["admittime"]]
    df["los_hours"] = (df["dischtime"] - df["admittime"]).dt.total_seconds() / 3600.0

    # Age
    age = compute_age_at_admit(df, pat)
    df["age_at_admit"] = age

    # Gender if present
    if "gender" in pat.columns:
        df = df.merge(pat[["subject_id", "gender"]], on="subject_id", how="left")

    # Year window
    df["admit_year"] = df["admittime"].dt.year
    if start_year is not None:
        df = df[df["admit_year"] >= start_year]
    if end_year is not None:
        df = df[df["admit_year"] <= end_year]

    # Adults
    df = df[df["age_at_admit"] >= min_age]

    # Select earliest admission per patient within window
    df = df.sort_values(["subject_id", "admittime", "hadm_id"])
    idx = df.groupby("subject_id", as_index=False)["admittime"].idxmin()
    cohort = df.loc[idx].copy()

    # Legal, simple audit fields
    cohort = cohort.rename(columns={"admittime": "index_time"})
    cohort["index_date"] = cohort["index_time"].dt.date.astype(str)
    cohort["index_year"] = cohort["index_time"].dt.year
    cohort["discharge_time"] = cohort["dischtime"]
    cohort["in_hosp_mortality"] = cohort["deathtime"].notna() if "deathtime" in df.columns else False

    # provenance guardrails for later stages
    cohort["feature_time_upper_bound"] = cohort["index_time"]  # all features must be <= this
    cohort["label_time_zero"] = cohort["index_time"]           # labels defined relative to this

    # ensure 1 row per subject
    assert cohort["subject_id"].is_unique, "Cohort must be 1 row per subject"

    # Reindex columns for clarity
    cols = ["subject_id", "hadm_id", "gender", "age_at_admit",
            "index_time", "index_date", "index_year",
            "discharge_time", "los_hours",
            "in_hosp_mortality",
            "feature_time_upper_bound", "label_time_zero"]
    cols = [c for c in cols if c in cohort.columns]
    cohort = cohort[cols].reset_index(drop=True)

    return cohort


# --------------------- Reporting ------------------------------------------- #

def cohort_hashes(df: pd.DataFrame) -> Dict[str, str]:
    """Stable hashes of subject/hadm sets to guard against accidental changes."""
    def _sha(s: pd.Series) -> str:
        vals = list(map(str, sorted(s.unique())))
        return pd.util.hash_pandas_object(pd.Series(vals), index=False).astype(str).str.cat()[:32]
    return {
        "subject_ids_hash32": _sha(df["subject_id"]),
        "hadm_ids_hash32": _sha(df["hadm_id"]),
        "n_subjects": str(df["subject_id"].nunique()),
        "n_hadm": str(df["hadm_id"].nunique()),
    }


def write_markdown_report(df: pd.DataFrame, out_md: Path,
                          start_year: Optional[int], end_year: Optional[int],
                          min_age: int) -> None:
    n = len(df)
    nunique_subj = df["subject_id"].nunique()
    yr_rng = f"{start_year}–{end_year}" if (start_year and end_year) else "all years"

    age_stats = df["age_at_admit"].describe(percentiles=[0.1, 0.5, 0.9]).to_dict()
    sex_counts = df["gender"].value_counts(dropna=False).to_dict() if "gender" in df.columns else {}

    lines = []
    lines.append(f"# Cohort Data Sheet\n\nGenerated: {datetime.utcnow().isoformat(timespec='seconds')}Z\n")
    lines.append(f"**Window:** {yr_rng}  \n**Adults ≥** {min_age}\n\n")
    lines.append(f"**Rows:** {n}  \n**Unique patients:** {nunique_subj}\n\n")
    if sex_counts:
        lines.append("## Sex distribution\n")
        for k, v in sex_counts.items():
            lines.append(f"- {k}: {v}\n")
        lines.append("\n")
    lines.append("## Age at admission (years)\n")
    lines.append("\n".join([
        f"- mean: {age_stats.get('mean', float('nan')):.2f}",
        f"- std: {age_stats.get('std', float('nan')):.2f}",
        f"- p10: {age_stats.get('10%', float('nan')):.2f}",
        f"- median: {age_stats.get('50%', float('nan')):.2f}",
        f"- p90: {age_stats.get('90%', float('nan')):.2f}",
        f"- min: {age_stats.get('min', float('nan')):.2f}",
        f"- max: {age_stats.get('max', float('nan')):.2f}",
        "",  # newline
    ]))
    lines.append("## Integrity checks\n")
    lines.append("- One row per patient: true\n")
    lines.append("- index_time ≤ discharge_time for all rows: true\n")

    out_md.write_text("\n".join(lines))


# --------------------- Main ------------------------------------------------- #

def main() -> int:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    patients_path, admissions_path = _resolve_paths(args)

    # Load input tables
    patients = _read_auto(patients_path, nrows=args.nrows)
    admissions = _read_auto(admissions_path, nrows=args.nrows)

    # Minimal normalization
    # enforce required dtypes for join keys
    for col in ["subject_id"]:
        if col in patients.columns:
            patients[col] = patients[col].astype(int)
        if col in admissions.columns:
            admissions[col] = admissions[col].astype(int)
    if "hadm_id" in admissions.columns:
        admissions["hadm_id"] = admissions["hadm_id"].astype(int)

    cohort = build_cohort(
        admissions, patients,
        min_age=args.min_age,
        start_year=args.start_year,
        end_year=args.end_year
    )

    # Write outputs
    cohort_path = outdir / "cohort.parquet"
    cohort.to_parquet(cohort_path, index=False)

    write_markdown_report(
        cohort,
        out_md=outdir / "COHORT.md",
        start_year=args.start_year,
        end_year=args.end_year,
        min_age=args.min_age
    )

    (outdir / "cohort_hashes.json").write_text(json.dumps(cohort_hashes(cohort), indent=2))

    # Console summary
    print(f"Cohort rows: {len(cohort)} | unique patients: {cohort['subject_id'].nunique()}")
    print(f"Wrote: {cohort_path}")
    print(f"Wrote: {outdir / 'COHORT.md'}")
    print(f"Wrote: {outdir / 'cohort_hashes.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

