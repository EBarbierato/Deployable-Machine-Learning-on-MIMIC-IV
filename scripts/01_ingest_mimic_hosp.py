#!/usr/bin/env python3
"""
01_ingest_mimic_hosp.py

Purpose
-------
Stream-ingest MIMIC-IV "hosp" CSVs (already decompressed in your case) into
typed, queryable Parquet datasets. Handles very large tables via chunked IO.

Features
--------
- Prefers .csv, falls back to .csv.gz if .csv is absent.
- Chunked read with pandas; date parsing for common time columns (when present).
- Light schema normalization: subject_id / hadm_id / stay_id / itemid as nullable ints.
- Writes to Parquet "dataset" layout: one folder per table, with part files.
- Optional partitioning for labevents by year or month (from charttime).
- Summaries per table: rows, unique IDs, min/max dates for parsed cols.
- Safe re-runs: skip existing output unless --overwrite is provided.

Example
-------
python scripts/01_ingest_mimic_hosp.py \
  --input-root "K:\\physionet.org\\files\\mimiciv\\3.1\\hosp" \
  --tables admissions,patients,diagnoses_icd,procedures_icd,labevents \
  --out-root artifacts/raw_parquet/hosp \
  --chunk-rows 1500000 \
  --partition-labevents month

Notes
-----
- Requires pandas + pyarrow installed.
- Date parsing is best-effort: only columns present are parsed.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except Exception as e:  # pragma: no cover
    raise SystemExit("pyarrow is required for this script (pip install pyarrow).") from e


# ----------------------------- Config & CLI --------------------------------- #

DEFAULT_TABLES_ORDER = [
    # Small-ish first
    "patients",
    "d_icd_diagnoses",
    "d_icd_procedures",
    "d_labitems",
    "admissions",
    "transfers",
    "diagnoses_icd",
    "procedures_icd",
    "pharmacy",
    "prescriptions",
    "microbiologyevents",
    # Very large
    "labevents",
]

# Columns we will TRY to parse as datetimes if present
DATE_CANDIDATES = {
    "admissions": ["admittime", "dischtime", "deathtime"],
    "transfers": ["intime", "outtime"],
    "prescriptions": ["starttime", "stoptime"],
    "pharmacy": ["starttime", "stoptime", "verifiedtime"],
    "labevents": ["charttime", "storetime"],
    "microbiologyevents": ["charttime", "storetime", "chartdate", "ab_charttime", "charttime_utc"],
    # dictionaries and code tables: none
}

# Columns we will TRY to cast to integers if present
INT_ID_CANDIDATES = ["subject_id", "hadm_id", "stay_id", "itemid", "specimen_id"]


@dataclass
class TableSpec:
    name: str
    filename: str  # expected base name (we'll prefer .csv, else .csv.gz)
    date_cols: List[str]


def build_table_specs() -> Dict[str, TableSpec]:
    mapping = {
        "patients": TableSpec("patients", "patients.csv", []),
        "admissions": TableSpec("admissions", "admissions.csv", DATE_CANDIDATES.get("admissions", [])),
        "transfers": TableSpec("transfers", "transfers.csv", DATE_CANDIDATES.get("transfers", [])),
        "diagnoses_icd": TableSpec("diagnoses_icd", "diagnoses_icd.csv", []),
        "procedures_icd": TableSpec("procedures_icd", "procedures_icd.csv", []),
        "pharmacy": TableSpec("pharmacy", "pharmacy.csv", DATE_CANDIDATES.get("pharmacy", [])),
        "prescriptions": TableSpec("prescriptions", "prescriptions.csv", DATE_CANDIDATES.get("prescriptions", [])),
        "labevents": TableSpec("labevents", "labevents.csv", DATE_CANDIDATES.get("labevents", [])),
        "microbiologyevents": TableSpec("microbiologyevents", "microbiologyevents.csv",
                                        DATE_CANDIDATES.get("microbiologyevents", [])),
        "d_icd_diagnoses": TableSpec("d_icd_diagnoses", "d_icd_diagnoses.csv", []),
        "d_icd_procedures": TableSpec("d_icd_procedures", "d_icd_procedures.csv", []),
        "d_labitems": TableSpec("d_labitems", "d_labitems.csv", []),
    }
    return mapping


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Ingest MIMIC-IV hosp CSVs (prefers .csv; falls back to .csv.gz) into Parquet datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--input-root", type=str,
                   default=os.environ.get("MIMIC_HOSP_DIR", "K:\\physionet.org\\files\\mimiciv\\3.1\\hosp"),
                   help="Folder containing the hosp CSV files.")
    p.add_argument("--tables", type=str, default=",".join(DEFAULT_TABLES_ORDER),
                   help="Comma list of table names to ingest (in order).")
    p.add_argument("--out-root", type=str, default="artifacts/raw_parquet/hosp",
                   help="Output root; a subfolder per table will be created.")
    p.add_argument("--chunk-rows", type=int, default=2_000_000,
                   help="Rows per chunk to stream during CSV read.")
    p.add_argument("--partition-labevents", type=str, choices=["none", "year", "month"], default="none",
                   help="Partitioning strategy for labevents (by charttime).")
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite existing table output directories.")
    p.add_argument("--dry-run", action="store_true",
                   help="List actions without writing Parquet.")
    return p.parse_args()


# ----------------------------- IO helpers ----------------------------------- #

def resolve_input_file(root: Path, base_name: str) -> Path:
    """
    Prefer .csv, else fallback to .csv.gz. Raise if neither exists.
    """
    p_csv = root / base_name
    p_gz = root / f"{base_name}.gz" if not base_name.endswith(".gz") else root / base_name
    if p_csv.exists():
        return p_csv
    if p_gz.exists():
        return p_gz
    raise FileNotFoundError(f"Neither {p_csv} nor {p_gz} exist.")


def ensure_outdir(out_dir: Path, overwrite: bool) -> None:
    if out_dir.exists():
        if overwrite:
            # remove existing files in folder (keeps folder)
            for p in out_dir.glob("**/*"):
                try:
                    p.unlink()
                except Exception:
                    pass
        else:
            # Skip re-ingest by keeping existing parts
            pass
    else:
        out_dir.mkdir(parents=True, exist_ok=True)


def to_arrow_table(df: pd.DataFrame) -> pa.Table:
    return pa.Table.from_pandas(df, preserve_index=False)


def write_part(table_dir: Path, part_idx: int, df: pd.DataFrame) -> Path:
    part_path = table_dir / f"part_{part_idx:05d}.parquet"
    pq.write_table(to_arrow_table(df), part_path)
    return part_path


def write_partitioned(dataset_dir: Path, df: pd.DataFrame, partition_cols: List[str]) -> None:
    """
    Write a pandas frame to a partitioned Parquet dataset (append mode).
    We keep it simple: write one fragment per call.
    """
    table = to_arrow_table(df)
    pq.write_to_dataset(table, root_path=str(dataset_dir), partition_cols=partition_cols)


# ----------------------------- Typing helpers -------------------------------- #

def normalize_int_cols(df: pd.DataFrame) -> pd.DataFrame:
    for c in [col for col in INT_ID_CANDIDATES if col in df.columns]:
        # Force to pandas nullable Int64 dtype
        try:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
        except Exception:
            # leave as-is if conversion fails
            pass
    return df


def parse_date_cols(df: pd.DataFrame, date_cols: Iterable[str]) -> pd.DataFrame:
    present = [c for c in date_cols if c in df.columns]
    for c in present:
        try:
            df[c] = pd.to_datetime(df[c], errors="coerce", utc=False)
        except Exception:
            pass
    return df


# ----------------------------- Labevents partitioning ------------------------ #

def add_labevents_partitions(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """
    Add partition columns based on charttime:
      - mode == 'year'  -> adds 'year'
      - mode == 'month' -> adds 'year', 'month' (YYYY, YYYY-MM)
    """
    if mode == "none" or "charttime" not in df.columns:
        return df
    ct = pd.to_datetime(df["charttime"], errors="coerce")
    df = df.copy()
    if mode == "year":
        df["year"] = ct.dt.year.astype("Int64")
    elif mode == "month":
        df["year"] = ct.dt.year.astype("Int64")
        df["month"] = ct.dt.to_period("M").astype(str)
    return df


# ----------------------------- Summaries ------------------------------------- #

@dataclass
class TableSummary:
    name: str
    rows: int
    parts_written: int
    unique_subjects: Optional[int]
    unique_hadm: Optional[int]
    date_min: Optional[str]
    date_max: Optional[str]
    out_dir: str
    input_path: str


def summarize_chunk(df: pd.DataFrame, date_cols: List[str]) -> Tuple[Optional[int], Optional[int], Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    usub = df["subject_id"].nunique() if "subject_id" in df.columns else None
    uhadm = df["hadm_id"].nunique() if "hadm_id" in df.columns else None
    dmin = dmax = None
    present = [c for c in date_cols if c in df.columns]
    if present:
        # pick the first date column heuristically
        c = present[0]
        dmin = pd.to_datetime(df[c], errors="coerce").min()
        dmax = pd.to_datetime(df[c], errors="coerce").max()
    return usub, uhadm, dmin, dmax


# ----------------------------- Ingest core ----------------------------------- #

def ingest_one_table(
    name: str,
    spec: TableSpec,
    input_root: Path,
    out_root: Path,
    chunk_rows: int,
    partition_labs: str,
    overwrite: bool,
    dry_run: bool
) -> TableSummary:
    input_path = resolve_input_file(input_root, spec.filename)
    out_dir = out_root / name
    ensure_outdir(out_dir, overwrite=overwrite)

    # Decide parsing hints
    date_cols = spec.date_cols
    compression = "infer"  # pandas will detect .gz automatically
    chunks = pd.read_csv(
        input_path,
        chunksize=chunk_rows,
        low_memory=False,
        compression=compression,
        dtype=None,           # allow pandas to infer; we fix IDs below
        na_values=["", "NA", "NaN", "NULL", "null"],
        keep_default_na=True
    )

    total_rows = 0
    parts_written = 0
    uniq_sub = 0
    uniq_hadm = 0
    global_dmin = None
    global_dmax = None

    # For labevents partitioning
    is_labs = (name == "labevents")
    part_mode = partition_labs if is_labs else "none"

    for i, df in enumerate(chunks):
        # Normalize schema
        df = normalize_int_cols(df)
        df = parse_date_cols(df, date_cols)
        if is_labs and part_mode != "none":
            df = add_labevents_partitions(df, part_mode)

        # Summaries from this chunk
        us, uh, dmin, dmax = summarize_chunk(df, date_cols)
        total_rows += len(df)
        uniq_sub += (us or 0)
        uniq_hadm += (uh or 0)
        if dmin is not None:
            global_dmin = min(global_dmin, dmin) if global_dmin is not None else dmin
        if dmax is not None:
            global_dmax = max(global_dmax, dmax) if global_dmax is not None else dmax

        if dry_run:
            continue

        # Write out
        if is_labs and part_mode != "none":
            parts_written += 1
            # Partition columns depend on mode
            pcols = ["year"] if part_mode == "year" else (["year", "month"] if part_mode == "month" else [])
            write_partitioned(out_dir, df, partition_cols=pcols)
        else:
            # Simple part file write
            part_path = write_part(out_dir, parts_written, df)
            parts_written += 1

    # Build summary
    sm = TableSummary(
        name=name,
        rows=total_rows,
        parts_written=parts_written,
        unique_subjects=uniq_sub if uniq_sub > 0 else None,
        unique_hadm=uniq_hadm if uniq_hadm > 0 else None,
        date_min=(global_dmin.isoformat() if isinstance(global_dmin, pd.Timestamp) and not pd.isna(global_dmin) else None),
        date_max=(global_dmax.isoformat() if isinstance(global_dmax, pd.Timestamp) and not pd.isna(global_dmax) else None),
        out_dir=str(out_dir),
        input_path=str(input_path),
    )
    return sm


def main() -> int:
    args = parse_args()
    input_root = Path(args.input_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    table_specs = build_table_specs()

    # Parse table list
    table_names = [t.strip() for t in args.tables.split(",") if t.strip()]
    # Ensure listed tables exist in our mapping
    bad = [t for t in table_names if t not in table_specs]
    if bad:
        raise SystemExit(f"Unknown table names: {bad}")

    # Ingest
    summaries: List[TableSummary] = []
    for tname in table_names:
        spec = table_specs[tname]
        print(f"[ingest] {tname}: reading {spec.filename} ...")
        try:
            sm = ingest_one_table(
                name=tname,
                spec=spec,
                input_root=input_root,
                out_root=out_root,
                chunk_rows=int(args.chunk_rows),
                partition_labs=args.partition_labevents,
                overwrite=bool(args.overwrite),
                dry_run=bool(args.dry_run),
            )
            summaries.append(sm)
            print(f"[ingest] {tname}: rows={sm.rows:,} parts={sm.parts_written} -> {sm.out_dir}")
        except FileNotFoundError as e:
            print(f"[ingest] SKIP {tname}: {e}")
        except Exception as e:
            print(f"[ingest] ERROR {tname}: {e}")
            raise

    # Write summary files
    manifest = {
        "generated_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "input_root": str(input_root),
        "out_root": str(out_root),
        "chunk_rows": int(args.chunk_rows),
        "partition_labevents": args.partition_labevents,
        "tables": [asdict(s) for s in summaries],
    }
    (out_root / "INGEST_SUMMARY.json").write_text(json.dumps(manifest, indent=2))

    lines = []
    lines.append(f"# MIMIC-IV hosp ingestion summary\n\nGenerated: {manifest['generated_utc']} UTC\n\n")
    for s in summaries:
        lines.append(f"## {s.name}\n")
        lines.append(f"- input: `{s.input_path}`\n- out: `{s.out_dir}`\n")
        lines.append(f"- rows: {s.rows:,}  |  parts: {s.parts_written}\n")
        if s.unique_subjects is not None:
            lines.append(f"- unique subject_id (approx per-chunk sum): {s.unique_subjects:,}\n")
        if s.unique_hadm is not None:
            lines.append(f"- unique hadm_id (approx per-chunk sum): {s.unique_hadm:,}\n")
        if s.date_min or s.date_max:
            lines.append(f"- date range: {s.date_min} .. {s.date_max}\n")
        lines.append("\n")
    (out_root / "INGEST_SUMMARY.md").write_text("".join(lines))

    print(f"[ingest] Done. Manifest -> {out_root / 'INGEST_SUMMARY.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
