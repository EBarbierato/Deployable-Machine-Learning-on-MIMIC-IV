#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python3
"""
04_train_val_test_split.py

Purpose
-------
Create leakage-safe, patient-level train/val/test splits for the comorbidity MTL pipeline.

Modes
-----
1) temporal (default)
   - Either provide explicit cut dates (--train-end, --val-end), or
     let the script derive cutpoints by index_time quantiles using --train-prop/--val-prop/--test-prop.

2) random
   - Patient-level random assignment using a fixed seed.
   - If a label column is provided (e.g., diab_incident_365d), performs *grouped stratification*
     by approximating class proportions at the patient level.

Inputs
------
- Cohort parquet from 01_extract_cohort.py with columns:
  subject_id, hadm_id, index_time  (one row per subject)
- Optional labels parquet from 02_define_labels.py for stratification.

Outputs
-------
- {outdir}/splits.parquet          # subject_id, fold, index_time, (optional stratify_label)
- {outdir}/SPLITS.md               # summary counts and date cutpoints
- {outdir}/splits_hashes.json      # integrity hashes for each fold

Examples
--------
# Temporal split using explicit cut dates
python scripts/04_train_val_test_split.py \
  --cohort-file artifacts/cohort/cohort.parquet \
  --labels-file artifacts/labels/labels.parquet \
  --mode temporal \
  --train-end 2014-12-31 \
  --val-end 2016-12-31 \
  --stratify-label any_incident_365d \
  --outdir artifacts/splits

# Temporal split using proportions (quantile cutpoints)
python scripts/04_train_val_test_split.py \
  --cohort-file artifacts/cohort/cohort.parquet \
  --mode temporal \
  --train-prop 0.7 --val-prop 0.1 --test-prop 0.2 \
  --outdir artifacts/splits

# Random split with label-aware stratification
python scripts/04_train_val_test_split.py \
  --cohort-file artifacts/cohort/cohort.parquet \
  --labels-file artifacts/labels/labels.parquet \
  --mode random \
  --stratify-label diab_incident_365d \
  --seed 42 \
  --outdir artifacts/splits
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ----------------------------- CLI ------------------------------------------ #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Create patient-level train/val/test splits (temporal or random).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--cohort-file", type=str, required=True,
                   help="Parquet from 01_extract_cohort.py (1 row per subject).")
    p.add_argument("--labels-file", type=str, default=None,
                   help="Optional labels parquet from 02_define_labels.py for stratification.")
    p.add_argument("--mode", type=str, choices=["temporal", "random"], default="temporal")

    # Temporal options
    p.add_argument("--train-end", type=str, default=None,
                   help="YYYY-MM-DD cutoff for TRAIN (index_time ≤ train-end).")
    p.add_argument("--val-end", type=str, default=None,
                   help="YYYY-MM-DD cutoff for VAL (train-end < index_time ≤ val-end). TEST is > val-end.")
    p.add_argument("--train-prop", type=float, default=0.7,
                   help="Used only if --train-end/--val-end not set; quantile split.")
    p.add_argument("--val-prop", type=float, default=0.1)
    p.add_argument("--test-prop", type=float, default=0.2)

    # Random options
    p.add_argument("--seed", type=int, default=42, help="Random seed for shuffle/random mode.")

    # Stratification
    p.add_argument("--stratify-label", type=str, default=None,
                   help="Boolean label name for stratification (e.g., diab_incident_365d). "
                        "Special value 'any_incident_365d' will be created as OR of *_incident_365d if present.")

    p.add_argument("--outdir", type=str, default="artifacts/splits")
    return p.parse_args()


# ----------------------------- UTIL ----------------------------------------- #

def _read_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)

def _to_datetime(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")

def _short_hash_of_ids(ids: pd.Series) -> str:
    # Deterministic short hash of sorted unique IDs
    vals = list(map(int, sorted(pd.Series(ids).dropna().unique())))
    # Use pandas util hash for speed, then compress to 32 chars
    return pd.util.hash_pandas_object(pd.Series(vals), index=False).astype(str).str.cat()[:32]

def _assert_unique_subjects(df: pd.DataFrame) -> None:
    if not df["subject_id"].is_unique:
        raise AssertionError("Cohort must be exactly one row per subject_id. Found duplicates.")

def _validate_props(tp: float, vp: float, sp: float) -> None:
    s = tp + vp + sp
    if not (abs(s - 1.0) < 1e-6):
        raise ValueError(f"train/val/test proportions must sum to 1.0 (got {tp}+{vp}+{sp}={s})")
    if min(tp, vp, sp) <= 0:
        raise ValueError("All proportions must be > 0.")


# ----------------------- LABEL HANDLING (optional) -------------------------- #

def attach_stratify_label(df: pd.DataFrame, labels: Optional[pd.DataFrame], name: Optional[str]) -> pd.DataFrame:
    if labels is None or name is None:
        df["__stratify__"] = np.nan
        return df

    lab = labels.copy()
    cols = set(lab.columns)

    # Special helper: 'any_incident_XXXd' -> OR of available *_incident_XXXd columns
    if name.startswith("any_incident_") and name.endswith("d"):
        horizon = name[len("any_incident_"):-1]  # e.g., "365"
        suffix = f"_incident_{horizon}d"
        cand = [c for c in lab.columns if c.endswith(suffix)]
        if not cand:
            # Fall back: if diabetes and cvd incident columns exist with any horizon, OR them all
            cand = [c for c in lab.columns if c.startswith(("diab_incident_", "cvd_incident_"))]
        if not cand:
            raise KeyError(f"Could not synthesize '{name}': no incident columns found in labels.")
        lab["__stratify__"] = lab[cand].fillna(False).astype(bool).any(axis=1)
    else:
        if name not in cols:
            raise KeyError(f"Stratify label '{name}' not found in labels file.")
        lab["__stratify__"] = lab[name].astype(bool)

    keep = lab[["subject_id", "__stratify__"]].drop_duplicates("subject_id")
    out = df.merge(keep, on="subject_id", how="left")
    return out


# --------------------------- SPLIT: TEMPORAL -------------------------------- #

@dataclass
class TemporalCutpoints:
    train_end: pd.Timestamp
    val_end: pd.Timestamp

def compute_temporal_cutpoints(df: pd.DataFrame, tp: float, vp: float) -> TemporalCutpoints:
    # df must be one row per subject with index_time as datetime
    n = len(df)
    if n < 3:
        raise ValueError("Not enough subjects to form train/val/test splits.")
    q1 = tp
    q2 = tp + vp
    # Clamp
    q1 = min(max(q1, 0.0), 0.98)
    q2 = min(max(q2, 0.02), 0.999)
    times = df["index_time"].sort_values().reset_index(drop=True)
    i1 = max(0, min(int(math.floor(q1 * n)) - 1, n - 3))
    i2 = max(i1 + 1, min(int(math.floor(q2 * n)) - 1, n - 2))
    return TemporalCutpoints(train_end=times.iloc[i1], val_end=times.iloc[i2])

def temporal_assign(df: pd.DataFrame, cp: TemporalCutpoints) -> pd.Series:
    t = df["index_time"]
    fold = pd.Series(np.where(t <= cp.train_end, "train",
                     np.where(t <= cp.val_end, "val", "test")), index=df.index, dtype="object")
    return fold


# --------------------------- SPLIT: RANDOM ---------------------------------- #

def random_assign(df: pd.DataFrame, tp: float, vp: float, sp: float, seed: int,
                  stratify: Optional[pd.Series]) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = df.index.to_numpy()

    if stratify is None or stratify.isna().all():
        # Simple shuffle
        idx_shuf = idx.copy()
        rng.shuffle(idx_shuf)
        n = len(idx_shuf)
        n_tr = int(round(tp * n))
        n_va = int(round(vp * n))
        fold = pd.Series(index=df.index, dtype="object")
        tr = idx_shuf[:n_tr]
        va = idx_shuf[n_tr:n_tr + n_va]
        te = idx_shuf[n_tr + n_va:]
        fold.loc[tr] = "train"
        fold.loc[va] = "val"
        fold.loc[te] = "test"
        return fold

    # Grouped stratification: split positives and negatives separately
    pos_idx = df[~stratify.isna() & (stratify.astype(bool))].index.to_numpy()
    neg_idx = df[~stratify.isna() & (~stratify.astype(bool))].index.to_numpy()
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)

    def slice_groups(arr: np.ndarray, tp: float, vp: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = len(arr)
        n_tr = int(round(tp * n))
        n_va = int(round(vp * n))
        return arr[:n_tr], arr[n_tr:n_tr + n_va], arr[n_tr + n_va:]

    tr_p, va_p, te_p = slice_groups(pos_idx, tp, vp)
    tr_n, va_n, te_n = slice_groups(neg_idx, tp, vp)

    fold = pd.Series(index=df.index, dtype="object")
    fold.loc[np.concatenate([tr_p, tr_n])] = "train"
    fold.loc[np.concatenate([va_p, va_n])] = "val"
    fold.loc[np.concatenate([te_p, te_n])] = "test"

    # Any unassigned (e.g., NaN stratify) go to test to be conservative
    unassigned = fold[fold.isna()].index.to_numpy()
    if unassigned.size:
        fold.loc[unassigned] = "test"
    return fold


# --------------------------- REPORTING -------------------------------------- #

def write_summaries(outdir: Path,
                    df: pd.DataFrame,
                    cp: Optional[TemporalCutpoints],
                    strat_name: Optional[str]) -> None:
    out_md = outdir / "SPLITS.md"
    lines: List[str] = []
    lines.append(f"# Splits Summary\n\nGenerated: {datetime.utcnow().isoformat(timespec='seconds')}Z\n\n")
    n = len(df)
    lines.append(f"Total patients: {n}\n\n")
    if cp is not None:
        lines.append("## Temporal cutpoints\n")
        lines.append(f"- train_end: {cp.train_end.date().isoformat()}\n")
        lines.append(f"- val_end:   {cp.val_end.date().isoformat()}\n\n")

    def fold_stats(name: str) -> str:
        sub = df[df["fold"] == name]
        k = len(sub)
        if strat_name and "__stratify__" in sub.columns and sub["__stratify__"].notna().any():
            pos = int(sub["__stratify__"].fillna(False).astype(bool).sum())
            return f"- {name}: {k} patients (positives by '{strat_name}': {pos})\n"
        return f"- {name}: {k} patients\n"

    lines.append("## Counts\n")
    lines.append(fold_stats("train"))
    lines.append(fold_stats("val"))
    lines.append(fold_stats("test"))
    out_md.write_text("".join(lines))

def write_hashes(outdir: Path, df: pd.DataFrame) -> None:
    payload: Dict[str, str] = {}
    for name in ["train", "val", "test"]:
        ids = df.loc[df["fold"] == name, "subject_id"]
        payload[f"{name}_n"] = str(len(ids))
        payload[f"{name}_subject_ids_hash32"] = _short_hash_of_ids(ids)
    (outdir / "splits_hashes.json").write_text(json.dumps(payload, indent=2))


# ----------------------------- MAIN ----------------------------------------- #

def main() -> int:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cohort = _read_parquet(Path(args.cohort_file)).copy()
    need = {"subject_id", "index_time"}
    miss = need - set(cohort.columns)
    if miss:
        raise KeyError(f"Cohort missing required columns: {miss}")
    cohort["index_time"] = _to_datetime(cohort["index_time"])
    if cohort["index_time"].isna().any():
        raise ValueError("Cohort contains NaT in index_time.")
    _assert_unique_subjects(cohort)

    # Optional labels for stratification
    labels = None
    if args.labels_file:
        labels = _read_parquet(Path(args.labels_file))

    df = cohort[["subject_id", "index_time"]].copy()
    df = attach_stratify_label(df, labels, args.stratify_label)

    # Assign folds
    fold: pd.Series
    cp: Optional[TemporalCutpoints] = None

    if args.mode == "temporal":
        if args.train_end and args.val_end:
            cp = TemporalCutpoints(train_end=pd.Timestamp(args.train_end), val_end=pd.Timestamp(args.val_end))
            if not (cp.train_end < cp.val_end):
                raise ValueError("--train-end must be strictly before --val-end.")
        else:
            _validate_props(args.train_prop, args.val_prop, args.test_prop)
            cp = compute_temporal_cutpoints(df, args.train_prop, args.val_prop)
        fold = temporal_assign(df, cp)
    else:
        _validate_props(args.train_prop, args.val_prop, args.test_prop)
        strat = df["__stratify__"] if "__stratify__" in df.columns else None
        fold = random_assign(df, args.train_prop, args.val_prop, args.test_prop, args.seed, strat)

    df["fold"] = fold.astype("string")

    # Integrity checks
    if set(df["fold"].unique()) != {"train", "val", "test"}:
        raise AssertionError("One or more folds are empty; adjust cutpoints or proportions.")

    if df["subject_id"].duplicated().any():
        raise AssertionError("Duplicate subject_id encountered after split.")

    # Persist artifacts
    out_parquet = outdir / "splits.parquet"
    df.to_parquet(out_parquet, index=False)

    write_summaries(outdir, df, cp, args.stratify_label)
    write_hashes(outdir, df)

    print(f"Wrote: {out_parquet}")
    print(f"Wrote: {outdir / 'SPLITS.md'}")
    print(f"Wrote: {outdir / 'splits_hashes.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

