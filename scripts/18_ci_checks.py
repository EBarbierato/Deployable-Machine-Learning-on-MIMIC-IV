#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python3
"""
18_ci_checks.py

Purpose
-------
Lightweight CI-style integrity checks for the numbered pipeline outputs.
- Presence: expected files for each stage found under artifacts.
- Schema: required columns/fields exist with sensible types/ranges.
- Sanity: metrics in [0,1], logloss>=0, thresholds in (0,1), AUPRC ~>= prevalence, etc.
- Cross-file alignment: y_true alignment between raw and calibrated preds, row counts, etc.
- Calibration params sanity: isotonic x sorted and y monotone; Platt params finite.

Outputs
-------
{outdir}/ci_report.json   # structured issues (ERROR/WARN/INFO)
{outdir}/SUMMARY.md       # human-readable summary
Exit code: 0 if OK (no errors), 1 if any ERROR (or WARN escalated via --strict)

Usage
-----
python scripts/18_ci_checks.py \
  --artifacts-root artifacts \
  --outdir artifacts/ci \
  --tasks diab_incident_365d,cvd_incident_365d \
  --strict
"""
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ----------------------------- CLI ------------------------------------------ #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run integrity checks over pipeline artifacts (files, schemas, metrics, thresholds, alignment).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--artifacts-root", type=str, default="artifacts",
                   help="Root folder containing baselines/, calibration/, explain/, dca/, fairness/, external_validation/, robustness/, paper_bundle/")
    p.add_argument("--outdir", type=str, default="artifacts/ci")
    p.add_argument("--tasks", type=str, default="",
                   help="Comma-separated task names for any MTL TEST/VAL preds (e.g., diab_incident_365d,cvd_incident_365d)")
    p.add_argument("--strict", action="store_true",
                   help="Treat WARN as ERROR in the exit code.")
    p.add_argument("--max-rows", type=int, default=10_000,
                   help="Max rows to read from very large parquet files for speed (use 0 for all).")
    return p.parse_args()


# ----------------------------- Issue recording ------------------------------ #

@dataclass
class Issue:
    severity: str   # ERROR | WARN | INFO
    code: str       # e.g., FILE_MISSING, BAD_RANGE
    where: str      # path or logical section
    detail: str

class Recorder:
    def __init__(self):
        self.issues: List[Issue] = []
    def add(self, severity: str, code: str, where: str, detail: str) -> None:
        self.issues.append(Issue(severity=severity, code=code, where=where, detail=detail))
    def counts(self) -> Dict[str, int]:
        c = {"ERROR": 0, "WARN": 0, "INFO": 0}
        for it in self.issues:
            c[it.severity] = c.get(it.severity, 0) + 1
        return c
    def to_json(self) -> List[dict]:
        return [asdict(i) for i in self.issues]


# ----------------------------- Helpers -------------------------------------- #

def _read_parquet_sample(path: Path, max_rows: int) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if max_rows and len(df) > max_rows:
        return df.sample(n=max_rows, random_state=0)
    return df

def _safe_read_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None

def _in_01(x: float) -> bool:
    return (x >= 0.0) and (x <= 1.0)

def _is_prob_series(s: pd.Series) -> bool:
    try:
        v = s.astype(float)
    except Exception:
        return False
    return (v.dropna().between(0.0, 1.0)).all()

def _find_dirs(root: Path, sub: str) -> List[Path]:
    p = root / sub
    if not p.exists():
        return []
    return [d for d in p.glob("**/*") if d.is_dir() and (d != p)]


# ----------------------------- Checks --------------------------------------- #

def check_baselines(root: Path, rec: Recorder, max_rows: int) -> None:
    base_root = root / "baselines"
    if not base_root.exists():
        rec.add("INFO", "SECTION_MISSING", str(base_root), "No baselines/ directory found; skipping.")
        return
    for model_dir in [d for d in base_root.glob("*/*") if d.is_dir()]:
        # Required files
        req = ["model.joblib", "metrics.json", "test_preds.parquet", "val_preds.parquet"]
        for fn in req:
            f = model_dir / fn
            if not f.exists():
                rec.add("ERROR", "FILE_MISSING", str(f), f"Missing required baseline artifact: {fn}")
        # Metrics sanity
        m = _safe_read_json(model_dir / "metrics.json")
        if m:
            for split in ["val", "test"]:
                md = m.get(split) or {}
                for k in ["auroc", "auprc"]:
                    v = md.get(k)
                    if v is None or not _in_01(float(v)):
                        rec.add("WARN", "BAD_RANGE", str(model_dir / "metrics.json"),
                                f"{split}.{k} not in [0,1]: {v}")
                for k in ["brier", "ece"]:
                    v = md.get(k)
                    if v is not None and not _in_01(float(v)):
                        rec.add("WARN", "BAD_RANGE", str(model_dir / "metrics.json"),
                                f"{split}.{k} expected in [0,1], got: {v}")
                v = md.get("logloss")
                if v is not None and float(v) < 0:
                    rec.add("WARN", "BAD_RANGE", str(model_dir / "metrics.json"),
                            f"{split}.logloss expected >= 0, got: {v}")
        # Predictions schema
        for pred_name in ["test_preds.parquet", "val_preds.parquet"]:
            p = model_dir / pred_name
            if not p.exists():
                continue
            try:
                df = _read_parquet_sample(p, max_rows=max_rows)
            except Exception as e:
                rec.add("ERROR", "PARQUET_READ_FAIL", str(p), f"Could not read: {e}")
                continue
            need = {"y_true", "y_score"}
            if not need.issubset(df.columns):
                rec.add("ERROR", "BAD_SCHEMA", str(p), f"Missing columns {need - set(df.columns)}")
                continue
            if not _is_prob_series(df["y_score"]):
                rec.add("WARN", "BAD_RANGE", str(p), "y_score values not in [0,1]")
            # prevalence vs AUPRC (basic check on TEST)
            if pred_name.startswith("test") and "metrics.json" in os.listdir(model_dir):
                prev = float(pd.to_numeric(df["y_true"], errors="coerce").mean())
                if m and m.get("test", {}).get("auprc") is not None:
                    auprc = float(m["test"]["auprc"])
                    if auprc + 1e-6 < max(0.0, prev - 0.05):
                        rec.add("WARN", "AUPRC_BELOW_PREV", str(model_dir / "metrics.json"),
                                f"TEST AUPRC ({auprc:.4f}) < prevalence-0.05 ({prev:.4f})")


def check_calibration(root: Path, rec: Recorder, max_rows: int) -> None:
    cal_root = root / "calibration"
    if not cal_root.exists():
        rec.add("INFO", "SECTION_MISSING", str(cal_root), "No calibration/ directory found; skipping.")
        return
    for d in _find_dirs(root, "calibration"):
        m = _safe_read_json(d / "metrics.json")
        th = _safe_read_json(d / "thresholds.json")
        if m is None:
            rec.add("ERROR", "FILE_MISSING", str(d / "metrics.json"), "Missing metrics.json")
        else:
            for split in ["validation", "test"]:
                sm = m.get(split, {})
                for variant, md in sm.items():
                    for k in ["auroc", "auprc", "brier", "ece"]:
                        v = md.get(k)
                        if v is not None and not _in_01(float(v)):
                            rec.add("WARN", "BAD_RANGE", str(d / "metrics.json"),
                                    f"{split}.{variant}.{k} not in [0,1]: {v}")
                    v = md.get("logloss")
                    if v is not None and float(v) < 0:
                        rec.add("WARN", "BAD_RANGE", str(d / "metrics.json"),
                                f"{split}.{variant}.logloss < 0: {v}")
        if th is None or "thresholds" not in th:
            rec.add("ERROR", "FILE_MALFORMED", str(d / "thresholds.json"), "Missing 'thresholds' key")
        else:
            for variant, rules in th["thresholds"].items():
                for rule, t in rules.items():
                    t = float(t)
                    if not (0.0 < t < 1.0):
                        rec.add("ERROR", "BAD_RANGE", str(d / "thresholds.json"),
                                f"Threshold ({variant}:{rule}) not in (0,1): {t}")

        # Calibrated preds files
        for fn in ["calibrated_val.parquet", "calibrated_test.parquet"]:
            p = d / fn
            if not p.exists():
                rec.add("ERROR", "FILE_MISSING", str(p), "Missing calibrated predictions parquet")
                continue
            try:
                df = _read_parquet_sample(p, max_rows=max_rows)
            except Exception as e:
                rec.add("ERROR", "PARQUET_READ_FAIL", str(p), f"Could not read: {e}")
                continue
            if "y_true" not in df.columns:
                rec.add("ERROR", "BAD_SCHEMA", str(p), "Missing y_true")
            score_cols = [c for c in df.columns if c.startswith("y_score")]
            if not score_cols:
                rec.add("ERROR", "BAD_SCHEMA", str(p), "No y_score_* columns present")
            for c in score_cols:
                if not _is_prob_series(df[c]):
                    rec.add("WARN", "BAD_RANGE", str(p), f"{c} values not in [0,1]")


def check_mtl_preds(root: Path, rec: Recorder, tasks: List[str], max_rows: int) -> None:
    # Look for any MTL run dirs containing TEST/VAL preds
    mtl_dirs = []
    for sub in ["mtl_shared_bottom", "mtl_cross_stitch", "mtl_mmoe"]:
        d = root / sub
        if d.exists():
            mtl_dirs.append(d)
    for d in mtl_dirs:
        for split in ["TEST", "VAL"]:
            p = d / f"{split}_preds.parquet"
            if not p.exists():
                rec.add("WARN", "FILE_MISSING", str(p), "Missing MTL predictions parquet")
                continue
            try:
                df = _read_parquet_sample(p, max_rows=max_rows)
            except Exception as e:
                rec.add("ERROR", "PARQUET_READ_FAIL", str(p), f"Could not read: {e}")
                continue
            for t in tasks:
                need = {f"{t}_y_true", f"{t}_y_score"}
                if not need.issubset(df.columns):
                    rec.add("ERROR", "BAD_SCHEMA", str(p), f"Missing MTL columns for task '{t}': {need - set(df.columns)}")
                else:
                    if not _is_prob_series(df[f"{t}_y_score"]):
                        rec.add("WARN", "BAD_RANGE", str(p), f"{t}_y_score values not in [0,1]")


def check_explain(root: Path, rec: Recorder) -> None:
    exp_root = root / "explain"
    if not exp_root.exists():
        rec.add("INFO", "SECTION_MISSING", str(exp_root), "No explain/ directory found; skipping.")
        return
    for d in _find_dirs(root, "explain"):
        p = d / "explain_global.csv"
        if not p.exists():
            rec.add("WARN", "FILE_MISSING", str(p), "Missing explain_global.csv")
            continue
        try:
            df = pd.read_csv(p)
        except Exception as e:
            rec.add("ERROR", "CSV_READ_FAIL", str(p), f"Could not read: {e}")
            continue
        if "feature" not in df.columns:
            rec.add("ERROR", "BAD_SCHEMA", str(p), "Missing 'feature' column")
        if not any(c in df.columns for c in ["shap_mean_abs", "perm_importance_mean", "value", "value_abs"]):
            rec.add("WARN", "BAD_SCHEMA", str(p), "No ranking column (shap_mean_abs/perm_importance_mean/value[_abs])")


def check_dca(root: Path, rec: Recorder) -> None:
    dca_root = root / "dca"
    if not dca_root.exists():
        rec.add("INFO", "SECTION_MISSING", str(dca_root), "No dca/ directory found; skipping.")
        return
    for d in _find_dirs(root, "dca"):
        p = d / "dca_curve.csv"
        if not p.exists():
            rec.add("WARN", "FILE_MISSING", str(p), "Missing dca_curve.csv")
            continue
        try:
            df = pd.read_csv(p)
        except Exception as e:
            rec.add("ERROR", "CSV_READ_FAIL", str(p), f"Could not read: {e}")
            continue
        need = {"threshold", "nb_model", "nb_all", "nb_none", "nri100"}
        if not need.issubset(df.columns):
            rec.add("ERROR", "BAD_SCHEMA", str(p), f"Missing columns {need - set(df.columns)}")
        if (df["threshold"].min() <= 0.0) or (df["threshold"].max() >= 1.0):
            rec.add("WARN", "BAD_RANGE", str(p), "Threshold grid should be within (0,1)")
        # Quick NB sanity: treat_none should be zero line
        if "nb_none" in df.columns and not np.allclose(df["nb_none"].fillna(0.0).values, 0.0, atol=1e-6):
            rec.add("WARN", "NB_SANITY", str(p), "nb_none not all zeros")


def check_fairness(root: Path, rec: Recorder) -> None:
    fair_root = root / "fairness"
    if not fair_root.exists():
        rec.add("INFO", "SECTION_MISSING", str(fair_root), "No fairness/ directory found; skipping.")
        return
    for d in _find_dirs(root, "fairness"):
        gm = d / "group_metrics.csv"
        pg = d / "parity_gaps.csv"
        if not gm.exists():
            rec.add("WARN", "FILE_MISSING", str(gm), "Missing group_metrics.csv")
        else:
            try:
                df = pd.read_csv(gm)
            except Exception as e:
                rec.add("ERROR", "CSV_READ_FAIL", str(gm), f"Could not read: {e}")
                df = None
            if df is not None:
                need = {"group_col", "group_val", "score_col", "auroc", "auprc", "brier", "logloss", "prevalence", "n"}
                if not need.issubset(df.columns):
                    rec.add("ERROR", "BAD_SCHEMA", str(gm), f"Missing columns {need - set(df.columns)}")
        if pg.exists():
            try:
                dfp = pd.read_csv(pg)
            except Exception as e:
                rec.add("ERROR", "CSV_READ_FAIL", str(pg), f"Could not read: {e}")


def check_external(root: Path, rec: Recorder) -> None:
    ext_root = root / "external_validation"
    if not ext_root.exists():
        rec.add("INFO", "SECTION_MISSING", str(ext_root), "No external_validation/ directory found; skipping.")
        return
    for d in _find_dirs(root, "external_validation"):
        m = _safe_read_json(d / "metrics.json")
        if m is None:
            rec.add("ERROR", "FILE_MISSING", str(d / "metrics.json"), "Missing metrics.json")
            continue
        fam = m.get("family", "baseline")
        if fam == "baseline":
            for var, md in (m.get("metrics", {}) or {}).items():
                for k in ["auroc", "auprc", "brier", "ece", "pos_rate"]:
                    v = md.get(k)
                    if v is not None and (k in {"auroc","auprc","brier","ece"} and not _in_01(float(v))):
                        rec.add("WARN", "BAD_RANGE", str(d / "metrics.json"),
                                f"{var}.{k} not in [0,1]: {v}")
                v = md.get("logloss")
                if v is not None and float(v) < 0:
                    rec.add("WARN", "BAD_RANGE", str(d / "metrics.json"), f"{var}.logloss < 0: {v}")
        elif fam == "mtl":
            for task, vmap in (m.get("metrics", {}) or {}).items():
                for var, md in vmap.items():
                    for k in ["auroc", "auprc", "brier", "ece", "pos_rate"]:
                        v = md.get(k)
                        if v is not None and (k in {"auroc","auprc","brier","ece"} and not _in_01(float(v))):
                            rec.add("WARN", "BAD_RANGE", str(d / "metrics.json"),
                                    f"{task}.{var}.{k} not in [0,1]: {v}")


def check_robustness(root: Path, rec: Recorder) -> None:
    rob_root = root / "robustness"
    if not rob_root.exists():
        rec.add("INFO", "SECTION_MISSING", str(rob_root), "No robustness/ directory found; skipping.")
        return
    for d in _find_dirs(root, "robustness"):
        p = d / "robustness_results.csv"
        if not p.exists():
            rec.add("WARN", "FILE_MISSING", str(p), "Missing robustness_results.csv")
            continue
        try:
            df = pd.read_csv(p)
        except Exception as e:
            rec.add("ERROR", "CSV_READ_FAIL", str(p), f"Could not read: {e}")
            continue
        need = {"scenario","variant","label","auroc","auprc","ece","n","pos_rate"}
        if not need.issubset(df.columns):
            rec.add("ERROR", "BAD_SCHEMA", str(p), f"Missing columns {need - set(df.columns)}")


def check_paper_bundle(root: Path, rec: Recorder) -> None:
    pb = root / "paper_bundle"
    if not pb.exists():
        rec.add("INFO", "SECTION_MISSING", str(pb), "No paper_bundle/ directory found; skipping.")
        return
    roll = pb / "tables" / "metrics_rollup.csv"
    if not roll.exists():
        rec.add("WARN", "FILE_MISSING", str(roll), "Missing metrics_rollup.csv")
    else:
        try:
            df = pd.read_csv(roll)
            need = {"label","method","test_auroc","test_auprc"}
            if not need.issubset(df.columns):
                rec.add("ERROR", "BAD_SCHEMA", str(roll), f"Missing columns {need - set(df.columns)}")
        except Exception as e:
            rec.add("ERROR", "CSV_READ_FAIL", str(roll), f"Could not read: {e}")


def check_calibration_params(root: Path, rec: Recorder) -> None:
    cal_root = root / "calibration"
    if not cal_root.exists():
        return
    for d in _find_dirs(root, "calibration"):
        p = d / "calibration_params.json"
        if not p.exists():
            continue
        meta = _safe_read_json(p)
        if not meta:
            rec.add("ERROR", "FILE_MALFORMED", str(p), "Could not parse JSON")
            continue
        prm = meta.get("params", {})
        if "isotonic" in prm:
            xs = prm["isotonic"].get("x", [])
            ys = prm["isotonic"].get("y", [])
            if not xs or not ys or len(xs) != len(ys):
                rec.add("ERROR", "BAD_CAL_PARAMS", str(p), "Isotonic params x/y missing or length mismatch")
            else:
                if any(x2 < x1 for x1, x2 in zip(xs, xs[1:])):
                    rec.add("ERROR", "BAD_CAL_PARAMS", str(p), "Isotonic x must be non-decreasing")
                if any(y2 < y1 for y1, y2 in zip(ys, ys[1:])):
                    rec.add("WARN", "CAL_NON_MONOTONE", str(p), "Isotonic y should be non-decreasing")
        if "platt" in prm:
            A = prm["platt"].get("A"); B = prm["platt"].get("B")
            if A is None or B is None or not np.isfinite([A,B]).all():
                rec.add("ERROR", "BAD_CAL_PARAMS", str(p), "Platt params A/B must be finite numbers")


def cross_checks_alignment(root: Path, rec: Recorder, max_rows: int) -> None:
    """
    Ensure calibrated_test and raw test_preds have the same y_true (and subject_id if present).
    """
    for bdir in [d for d in (root / "baselines").glob("*/*") if d.is_dir()]:
        raw = bdir / "test_preds.parquet"
        cal = (root / "calibration" / f"{bdir.name}_{bdir.parent.name}" / "calibrated_test.parquet")
        if raw.exists() and cal.exists():
            try:
                df_raw = _read_parquet_sample(raw, max_rows=max_rows)
                df_cal = _read_parquet_sample(cal, max_rows=max_rows)
            except Exception:
                continue
            if "y_true" in df_raw.columns and "y_true" in df_cal.columns:
                if not np.allclose(df_raw["y_true"].values[: min(len(df_raw), len(df_cal))],
                                   df_cal["y_true"].values[: min(len(df_raw), len(df_cal))]):
                    rec.add("WARN", "MISALIGNED_YTRUE", f"{cal}", "y_true differs between raw and calibrated TEST preds")
            if "subject_id" in df_raw.columns and "subject_id" in df_cal.columns:
                inter = set(df_raw["subject_id"]).intersection(set(df_cal["subject_id"]))
                if len(inter) < min(len(df_raw), len(df_cal)) * 0.95:
                    rec.add("WARN", "SUBJECT_MISMATCH", f"{cal}", "subject_id overlap <95% between raw and calibrated")


# ----------------------------- MAIN ----------------------------------------- #

def main() -> int:
    args = parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    root = Path(args.artifacts_root)

    rec = Recorder()
    tasks = [s.strip() for s in args.tasks.split(",") if s.strip()]

    # Run suites
    check_baselines(root, rec, max_rows=args.max_rows)
    check_calibration(root, rec, max_rows=args.max_rows)
    check_mtl_preds(root, rec, tasks, max_rows=args.max_rows)
    check_explain(root, rec)
    check_dca(root, rec)
    check_fairness(root, rec)
    check_external(root, rec)
    check_robustness(root, rec)
    check_paper_bundle(root, rec)
    check_calibration_params(root, rec)
    cross_checks_alignment(root, rec, max_rows=args.max_rows)

    counts = rec.counts()

    # Write report
    report = {
        "artifacts_root": str(root),
        "generated_utc": pd.Timestamp.utcnow().isoformat(timespec="seconds") + "Z",
        "counts": counts,
        "strict": bool(args.strict),
        "issues": rec.to_json(),
    }
    (outdir / "ci_report.json").write_text(json.dumps(report, indent=2))

    # Markdown summary
    lines: List[str] = []
    lines.append(f"# CI Checks Summary\n\n")
    lines.append(f"- Artifacts root: `{root}`\n")
    lines.append(f"- Generated: {report['generated_utc']}\n")
    lines.append(f"- Totals — ERROR: {counts.get('ERROR',0)}, WARN: {counts.get('WARN',0)}, INFO: {counts.get('INFO',0)}\n\n")
    if rec.issues:
        lines.append("## Findings (first 50)\n")
        for it in rec.issues[:50]:
            lines.append(f"- **{it.severity}** [{it.code}] — {it.where}\n  - {it.detail}\n")
    else:
        lines.append("No issues detected.\n")
    (outdir / "SUMMARY.md").write_text("".join(lines))

    # Exit code
    errors = counts.get("ERROR", 0)
    warns  = counts.get("WARN", 0)
    rc = 1 if (errors > 0 or (args.strict and warns > 0)) else 0
    print(json.dumps(report["counts"], indent=2))
    print(f"Wrote: {outdir / 'ci_report.json'}")
    print(f"Wrote: {outdir / 'SUMMARY.md'}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

