#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02_define_labels.py  —  Leakage-safe label definition for incident outcomes

This script builds prevalence and incident labels for diabetes and cardiovascular
disease (CVD) in a leakage-safe way, anchored at each subject's index admission
time. It is a drop-in replacement for earlier versions, with clearer structure,
more defensive checks, and reproducible outputs.

-------------------------------------------------------------------------------
Inputs
-------------------------------------------------------------------------------
1) --cohort-file : Parquet produced by 01_extract_cohort.py
                   required columns: subject_id, hadm_id, index_time
2) Either --mimic-root pointing to MIMIC-IV (expects hosp/diagnoses_icd.* and
   hosp/admissions.*) OR explicit --diagnoses-file and --admissions-file.
3) Optional YAML files to override ICD prefix sets for diabetes / CVD.

-------------------------------------------------------------------------------
Outputs (in --outdir, default: artifacts/labels)
-------------------------------------------------------------------------------
- labels.parquet                  : final labels joined to cohort keys
- LABELS.md                       : human-readable summary
- icd_codebook_used.json          : ICD prefixes, horizons, switches used
- labels_hashes.json              : short hashes/counts for change detection

-------------------------------------------------------------------------------
Label semantics
-------------------------------------------------------------------------------
For a condition (e.g., "diab") we compute the earliest admission time whose
diagnoses match the ICD prefix set (ICD-9 and/or ICD-10). With index_time τ_u:

Prevalence:   prevalent_at_index = 1  if earliest_event_time < τ_u
Incident(H):  incident_Hd        = 1  if earliest_event_time ∈ (τ_u, τ_u + H]
                                OR    earliest_event_time == τ_u and same hadm
                                       (the latter controlled by the CLI switch)

Temporal guards:
- We ONLY use admission start time to anchor diagnosis timing.
- We NEVER look past τ_u for prevalence; incident is restricted to (τ_u, τ_u+H].

-------------------------------------------------------------------------------
Example
-------------------------------------------------------------------------------
python scripts/02_define_labels.py \
  --cohort-file /path/cohort.parquet \
  --mimic-root  /data/mimiciv \
  --incident-horizons 30,90,365 \
  --count-index-admission-as-incident true \
  --outdir /path/artifacts/labels
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import yaml


# ----------------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------------

def setup_logging(level: str = "INFO") -> None:
    lvl = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Define leakage-safe prevalence and incident labels for diabetes and CVD.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Core I/O
    p.add_argument("--cohort-file", type=str, required=True,
                   help="Parquet from 01_extract_cohort.py with subject_id, hadm_id, index_time")
    p.add_argument("--mimic-root", type=str, default=None,
                   help="Root of MIMIC-IV; expects hosp/diagnoses_icd.* and hosp/admissions.*")
    p.add_argument("--diagnoses-file", type=str, default=None,
                   help="Explicit path to hosp/diagnoses_icd if not using --mimic-root")
    p.add_argument("--admissions-file", type=str, default=None,
                   help="Explicit path to hosp/admissions if not using --mimic-root")

    # ICD configuration
    p.add_argument("--diabetes-codes-yaml", type=str, default=None,
                   help="YAML with icd9_prefixes/icd10_prefixes for diabetes")
    p.add_argument("--cvd-codes-yaml", type=str, default=None,
                   help="YAML with icd9_prefixes/icd10_prefixes for CVD")

    # Label semantics
    p.add_argument("--incident-horizons", type=str, default="30,90,365",
                   help="Comma-separated day horizons for incident labels (positive integers)")
    p.add_argument("--count-index-admission-as-incident", type=str2bool, default=True,
                   help="If true, a diagnosis on the index admission counts as incident")

    # Misc
    p.add_argument("--outdir", type=str, default="artifacts/labels",
                   help="Directory to write outputs")
    p.add_argument("--nrows", type=int, default=None,
                   help="Optional cap for quick trial runs (read head of CSVs)")
    p.add_argument("--log-level", type=str, default="INFO",
                   help="Logging level (DEBUG, INFO, WARNING, ERROR)")

    return p.parse_args()


def str2bool(x: str) -> bool:
    return str(x).strip().lower() in {"1", "true", "t", "yes", "y"}


# ----------------------------------------------------------------------------
# ICD utilities
# ----------------------------------------------------------------------------

@dataclass
class ICDCodebook:
    icd9_prefixes: List[str]
    icd10_prefixes: List[str]


def _normalize_icd(s: str) -> str:
    # Keep alnum, uppercase; strip punctuation/spaces e.g. "E11.9" -> "E119"
    return "".join(ch for ch in str(s).upper() if ch.isalnum())


def _normalize_prefixes(prefixes: Iterable[str]) -> List[str]:
    return [_normalize_icd(p) for p in prefixes]


def default_diabetes() -> ICDCodebook:
    return ICDCodebook(
        icd9_prefixes=["249", "250"],
        icd10_prefixes=["E08", "E09", "E10", "E11", "E13"],
    )


def default_cvd() -> ICDCodebook:
    # 410–414, 428, 433–438, 440 (ICD-9); I20–I25, I50, I60–I69, I70 (ICD-10)
    icd9 = [str(x) for x in range(410, 415)] + ["428"] + [str(x) for x in range(433, 439)] + ["440"]
    icd10 = [f"I{n}" for n in range(20, 26)] + ["I50"] + [f"I{n}" for n in range(60, 70)] + ["I70"]
    return ICDCodebook(icd9_prefixes=icd9, icd10_prefixes=icd10)


def load_codebook(yaml_path: Optional[str], fallback: ICDCodebook) -> ICDCodebook:
    if not yaml_path:
        return fallback
    cfg = yaml.safe_load(Path(yaml_path).read_text())
    nine = cfg.get("icd9_prefixes", []) or []
    ten = cfg.get("icd10_prefixes", []) or []
    nine = _normalize_prefixes(nine) if nine else fallback.icd9_prefixes
    ten = _normalize_prefixes(ten) if ten else fallback.icd10_prefixes
    return ICDCodebook(icd9_prefixes=nine, icd10_prefixes=ten)


def _code_matches_any_prefix(code: str, prefixes: List[str]) -> bool:
    c = _normalize_icd(code)
    return any(c.startswith(p) for p in prefixes)


# ----------------------------------------------------------------------------
# I/O helpers
# ----------------------------------------------------------------------------

def _resolve_inputs(args: argparse.Namespace) -> Tuple[Path, Path]:
    if args.mimic_root:
        root = Path(args.mimic_root) / "hosp"
        diag = _first_existing(root, ["diagnoses_icd.parquet", "diagnoses_icd.csv.gz", "diagnoses_icd.csv"])
        adm = _first_existing(root, ["admissions.parquet", "admissions.csv.gz", "admissions.csv"])
        if not (diag and adm):
            raise FileNotFoundError("Could not find hosp/diagnoses_icd.* and/or hosp/admissions.* under --mimic-root")
        return diag, adm

    if args.diagnoses_file and args.admissions_file:
        return Path(args.diagnoses_file), Path(args.admissions_file)

    raise ValueError("Provide either --mimic-root OR both --diagnoses-file and --admissions-file.")


def _first_existing(base: Path, candidates: List[str]) -> Optional[Path]:
    for name in candidates:
        p = base / name
        if p.exists():
            return p
    return None


def _read_auto(path: Path, nrows: Optional[int] = None) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    # CSV
    return pd.read_csv(path, nrows=nrows, low_memory=False)


# ----------------------------------------------------------------------------
# Core labeling logic
# ----------------------------------------------------------------------------

def earliest_event_times(
    diagnoses: pd.DataFrame,
    admissions: pd.DataFrame,
    cohort: pd.DataFrame,
    cb: ICDCodebook,
    label_name: str,
) -> pd.DataFrame:
    """
    For a given ICD codebook, return earliest admission time with a matching diagnosis,
    per subject. Columns returned:
      subject_id,
      {label}_first_event_time (datetime64[ns]),
      {label}_first_event_hadm_id (Int64)
    """
    need_diag = {"subject_id", "hadm_id", "icd_code", "icd_version"}
    missing = need_diag - set(diagnoses.columns)
    if missing:
        raise KeyError(f"diagnoses_icd missing columns: {missing}")

    d = diagnoses.loc[:, ["subject_id", "hadm_id", "icd_code", "icd_version"]].copy()
    d["icd_code"] = d["icd_code"].astype(str)
    # icd_version may come as object in some dumps
    d["icd_version"] = pd.to_numeric(d["icd_version"], errors="coerce").astype("Int64")

    mask9 = (d["icd_version"] == 9) & d["icd_code"].map(lambda c: _code_matches_any_prefix(c, cb.icd9_prefixes))
    mask10 = (d["icd_version"] == 10) & d["icd_code"].map(lambda c: _code_matches_any_prefix(c, cb.icd10_prefixes))
    d = d[mask9 | mask10]

    if d.empty:
        logging.warning("No matching %s codes found; returning NaT events.", label_name)
        return cohort[["subject_id"]].assign(
            **{
                f"{label_name}_first_event_time": pd.NaT,
                f"{label_name}_first_event_hadm_id": pd.Series(pd.array([pd.NA], dtype='Int64')).iloc[0],
            }
        )

    a = admissions.loc[:, ["hadm_id", "admittime"]].copy()
    a["admittime"] = pd.to_datetime(a["admittime"], errors="coerce")
    a = a.dropna(subset=["admittime"])

    # Enforce int64 hadm_id where present
    for df in (d, a, cohort):
        if "hadm_id" in df.columns:
            df["hadm_id"] = pd.to_numeric(df["hadm_id"], errors="coerce").astype("Int64")

    d = d.merge(a, on="hadm_id", how="left").dropna(subset=["admittime"])

    if d.empty:
        logging.warning("Matched diagnoses had no valid admittime; returning NaT events for %s.", label_name)
        return cohort[["subject_id"]].assign(
            **{
                f"{label_name}_first_event_time": pd.NaT,
                f"{label_name}_first_event_hadm_id": pd.Series(pd.array([pd.NA], dtype='Int64')).iloc[0],
            }
        )

    d = d.sort_values(["subject_id", "admittime", "hadm_id"])
    grp = d.groupby("subject_id", as_index=False).first()
    out = grp.loc[:, ["subject_id", "hadm_id", "admittime"]].rename(
        columns={
            "hadm_id": f"{label_name}_first_event_hadm_id",
            "admittime": f"{label_name}_first_event_time",
        }
    )

    # Ensure every cohort subject appears
    out = cohort[["subject_id"]].merge(out, on="subject_id", how="left")
    return out


def label_from_event_time(
    cohort: pd.DataFrame,
    event_df: pd.DataFrame,
    label_name: str,
    horizons_days: List[int],
    count_index_as_incident: bool,
) -> pd.DataFrame:
    """
    Using earliest event times, produce:
      - {label}_prevalent_at_index (bool)
      - {label}_incident_{H}d (bool for each H in horizons_days)
    """
    df = cohort.merge(event_df, on="subject_id", how="left")

    idx_t = pd.to_datetime(df["index_time"], errors="coerce")
    e_t = pd.to_datetime(df[f"{label_name}_first_event_time"], errors="coerce")

    # Prevalent strictly before index time
    prevalent = e_t.notna() & (e_t < idx_t)
    df[f"{label_name}_prevalent_at_index"] = prevalent

    # Incident base: after index OR (equal and same hadm if allowed)
    same_hadm = (
        pd.to_numeric(df[f"{label_name}_first_event_hadm_id"], errors="coerce").astype("Int64")
        == pd.to_numeric(df["hadm_id"], errors="coerce").astype("Int64")
    )
    if count_index_as_incident:
        incident_base = e_t.notna() & ((e_t > idx_t) | ((e_t == idx_t) & same_hadm.fillna(False)))
    else:
        incident_base = e_t.notna() & (e_t > idx_t)

    for H in horizons_days:
        if H <= 0:
            continue
        cut = idx_t + pd.to_timedelta(H, unit="D")
        df[f"{label_name}_incident_{H}d"] = incident_base & (e_t <= cut)

    return df


# ----------------------------------------------------------------------------
# Reporting & hashes
# ----------------------------------------------------------------------------

def write_markdown_report(
    out_md: Path,
    horizons: List[int],
    labels: pd.DataFrame,
    label_prefixes: Dict[str, List[str]],
) -> None:
    N = len(labels)
    now = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    lines = []
    lines.append(f"# Labels Data Sheet\n\nGenerated: {now}\n\n")
    lines.append(f"Rows (patients): {N}\n\n")

    # Diabetes
    lines.append("## Diabetes\n")
    lines.append(f"Prevalent at/before index: {int(labels['diab_prevalent_at_index'].sum())}\n")
    for H in horizons:
        col = f"diab_incident_{H}d"
        if col in labels:
            lines.append(f"Incident within {H} days: {int(labels[col].sum())}\n")
    lines.append("\n")

    # CVD
    lines.append("## Cardiovascular disease (CVD)\n")
    lines.append(f"Prevalent at/before index: {int(labels['cvd_prevalent_at_index'].sum())}\n")
    for H in horizons:
        col = f"cvd_incident_{H}d"
        if col in labels:
            lines.append(f"Incident within {H} days: {int(labels[col].sum())}\n")
    lines.append("\n")

    # Codebooks used
    lines.append("## ICD prefixes used\n")
    lines.append("Diabetes ICD-9: " + ", ".join(label_prefixes["diab_icd9"]) + "\n")
    lines.append("Diabetes ICD-10: " + ", ".join(label_prefixes["diab_icd10"]) + "\n")
    lines.append("CVD ICD-9: " + ", ".join(label_prefixes["cvd_icd9"]) + "\n")
    lines.append("CVD ICD-10: " + ", ".join(label_prefixes["cvd_icd10"]) + "\n")

    out_md.write_text("".join(lines))


def _short_hash_bool_index(series: pd.Series) -> str:
    # Stable short hash of positive indices only (for drift detection)
    idx = series[series.fillna(False)].index.to_series().astype("int64").sort_values()
    if idx.empty:
        return "0"*16
    return pd.util.hash_pandas_object(idx, index=False).astype(str).str.cat()[:32]


def labels_hashes(labels: pd.DataFrame, horizons: List[int]) -> Dict[str, str]:
    payload = {
        "n_patients": int(len(labels)),
        "n_diab_prev": int(labels["diab_prevalent_at_index"].sum()),
        "n_cvd_prev": int(labels["cvd_prevalent_at_index"].sum()),
    }
    for H in horizons:
        d_col = f"diab_incident_{H}d"
        c_col = f"cvd_incident_{H}d"
        if d_col in labels:
            payload[f"n_diab_incident_{H}d"] = int(labels[d_col].sum())
            payload[f"hash_diab_incident_{H}d"] = _short_hash_bool_index(labels[d_col])
        if c_col in labels:
            payload[f"n_cvd_incident_{H}d"] = int(labels[c_col].sum())
            payload[f"hash_cvd_incident_{H}d"] = _short_hash_bool_index(labels[c_col])
    return payload


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    setup_logging(args.log_level)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Read cohort
    cohort = pd.read_parquet(args.cohort_file)
    need = {"subject_id", "hadm_id", "index_time"}
    miss = need - set(cohort.columns)
    if miss:
        raise KeyError(f"Cohort missing required columns: {miss}")

    cohort["index_time"] = pd.to_datetime(cohort["index_time"], errors="coerce")
    if cohort["index_time"].isna().any():
        raise ValueError("Cohort contains invalid index_time (NaT).")

    # Standardize hadm_id dtype to Int64 across frames we control
    cohort["hadm_id"] = pd.to_numeric(cohort["hadm_id"], errors="coerce").astype("Int64")

    # Resolve and read MIMIC tables
    diag_path, adm_path = _resolve_inputs(args)
    logging.info("Reading diagnoses from %s", diag_path)
    diagnoses = _read_auto(diag_path, nrows=args.nrows)
    logging.info("Reading admissions from %s", adm_path)
    admissions = _read_auto(adm_path, nrows=args.nrows)

    if "admittime" not in admissions.columns:
        raise KeyError("Admissions table must contain 'admittime'.")
    admissions["admittime"] = pd.to_datetime(admissions["admittime"], errors="coerce")
    admissions = admissions.dropna(subset=["admittime"])
    admissions["hadm_id"] = pd.to_numeric(admissions["hadm_id"], errors="coerce").astype("Int64")

    # ICD codebooks
    diab_cb = load_codebook(args.diabetes_codes_yaml, default_diabetes())
    cvd_cb = load_codebook(args.cvd_codes_yaml, default_cvd())

    # Earliest events
    logging.info("Computing earliest diabetes events")
    diab_events = earliest_event_times(diagnoses, admissions, cohort, diab_cb, "diab")
    logging.info("Computing earliest CVD events")
    cvd_events = earliest_event_times(diagnoses, admissions, cohort, cvd_cb, "cvd")

    # Horizons
    horizons = sorted({int(h) for h in (s.strip() for s in args.incident_horizons.split(",")) if s.strip()})
    horizons = [h for h in horizons if h > 0]
    if not horizons:
        raise ValueError("--incident-horizons produced no positive integers.")

    # Build labels
    df = cohort.copy()
    df = df.merge(diab_events, on="subject_id", how="left")
    df = df.merge(cvd_events, on="subject_id", how="left")

    df = label_from_event_time(df, diab_events, "diab", horizons, args.count_index_admission_as_incident)
    df = label_from_event_time(df, cvd_events,  "cvd",  horizons, args.count_index_admission_as_incident)

    # Select output columns in a stable order
    keep = [
        "subject_id", "hadm_id", "index_time",
        "diab_first_event_time", "diab_first_event_hadm_id", "diab_prevalent_at_index",
        "cvd_first_event_time",  "cvd_first_event_hadm_id",  "cvd_prevalent_at_index",
    ] + [f"diab_incident_{h}d" for h in horizons] + [f"cvd_incident_{h}d" for h in horizons]

    keep = [c for c in keep if c in df.columns]
    labels = df.loc[:, keep].reset_index(drop=True)

    # Write artifacts
    labels_path = outdir / "labels.parquet"
    labels.to_parquet(labels_path, index=False)

    write_markdown_report(
        out_md=outdir / "LABELS.md",
        horizons=horizons,
        labels=labels,
        label_prefixes={
            "diab_icd9": diab_cb.icd9_prefixes,
            "diab_icd10": diab_cb.icd10_prefixes,
            "cvd_icd9": cvd_cb.icd9_prefixes,
            "cvd_icd10": cvd_cb.icd10_prefixes,
        },
    )

    (outdir / "icd_codebook_used.json").write_text(json.dumps({
        "diabetes": {"icd9_prefixes": diab_cb.icd9_prefixes, "icd10_prefixes": diab_cb.icd10_prefixes},
        "cvd": {"icd9_prefixes": cvd_cb.icd9_prefixes, "icd10_prefixes": cvd_cb.icd10_prefixes},
        "incident_horizons_days": horizons,
        "count_index_admission_as_incident": bool(args.count_index_admission_as_incident),
    }, indent=2))

    (outdir / "labels_hashes.json").write_text(json.dumps(labels_hashes(labels, horizons), indent=2))

    logging.info("Wrote: %s", labels_path)
    logging.info("Wrote: %s", outdir / "LABELS.md")
    logging.info("Wrote: %s", outdir / "icd_codebook_used.json")
    logging.info("Wrote: %s", outdir / "labels_hashes.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
