#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python3
"""
17_model_and_data_cards.py

Purpose
-------
Generate paper-friendly MODEL CARD and DATA CARD from existing artifacts:
- Model Card: overview, intended use, training/eval data, performance (VAL/TEST),
  calibration, operating thresholds, top features, fairness gaps, external validation,
  robustness highlights, and environment info.
- Data Card: cohort construction, label definitions, feature provenance, basic stats.

Inputs (provide what you have; missing pieces are skipped gracefully)
---------------------------------------------------------------------
--project-name          Short display name for the project (e.g., "Comorbidity MTL")
--cohort-file           Parquet from 01_extract_cohort.py (for cohort stats)
--labels-file           Parquet from 02_define_labels.py (for label names)
--baselines-root        Root of 06 outputs (*/model/metrics.json, preds)
--calibration-dirs      Comma-separated dirs from 09 (each with metrics.json, thresholds.json)
--explain-dirs          Comma-separated dirs from 10 (each with explain_global.csv)
--paper-bundle          Path to 11_paper_bundle.py outdir (to reuse rollup tables)
--fairness-dirs         Comma-separated dirs from 13 (each with parity_gaps.csv, group_metrics.csv)
--external-dirs         Comma-separated dirs from 14 (each with metrics.json)
--robustness-dirs       Comma-separated dirs from 15 (each with robustness_results.csv)
--env-report            Optional JSON from 00_make_env.py (library versions, seeds)
--outdir                Output directory (default docs/cards)

Outputs
-------
{outdir}/
  model_card.md
  model_card.json
  data_card.md
  data_card.json

Example
-------
python scripts/17_model_and_data_cards.py \
  --project-name "Comorbidity in DM & CVD" \
  --cohort-file artifacts/cohort/cohort.parquet \
  --labels-file artifacts/labels/labels.parquet \
  --baselines-root artifacts/baselines \
  --calibration-dirs artifacts/calibration/diab_logistic \
  --explain-dirs artifacts/explain/diab_logistic \
  --paper-bundle artifacts/paper_bundle \
  --fairness-dirs artifacts/fairness/diab_logistic_cal \
  --external-dirs artifacts/external_validation/diab_logistic \
  --robustness-dirs artifacts/robustness/diab_logistic \
  --env-report artifacts/env/env_report.json \
  --outdir docs/cards
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ----------------------------- CLI ------------------------------------------ #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate MODEL CARD and DATA CARD from pipeline artifacts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--project-name", type=str, default="Comorbidity Modeling")
    p.add_argument("--cohort-file", type=str, default=None)
    p.add_argument("--labels-file", type=str, default=None)
    p.add_argument("--baselines-root", type=str, default=None)
    p.add_argument("--calibration-dirs", type=str, default="")
    p.add_argument("--explain-dirs", type=str, default="")
    p.add_argument("--paper-bundle", type=str, default=None)
    p.add_argument("--fairness-dirs", type=str, default="")
    p.add_argument("--external-dirs", type=str, default="")
    p.add_argument("--robustness-dirs", type=str, default="")
    p.add_argument("--env-report", type=str, default=None)
    p.add_argument("--outdir", type=str, default="docs/cards")
    return p.parse_args()

# ----------------------------- IO helpers ----------------------------------- #

def _split_paths(csv_like: str) -> List[Path]:
    return [Path(s.strip()) for s in (csv_like or "").split(",") if s.strip()]

def _safe_read_json(p: Path) -> Optional[dict]:
    try:
        return json.loads(p.read_text())
    except Exception:
        return None

def _safe_read_csv_or_parquet(p: Path) -> Optional[pd.DataFrame]:
    try:
        if p.suffix.lower() == ".parquet":
            return pd.read_parquet(p)
        return pd.read_csv(p)
    except Exception:
        return None

def _exists_any(paths: List[Path]) -> bool:
    return any(p.exists() for p in paths)

# ----------------------------- Harvesters ----------------------------------- #

def harvest_baselines(root: Optional[Path]) -> pd.DataFrame:
    if not root or not root.exists():
        return pd.DataFrame()
    rows = []
    for label_dir in root.glob("*"):
        if not label_dir.is_dir():
            continue
        for model_dir in label_dir.glob("*"):
            m = _safe_read_json(model_dir / "metrics.json")
            if not m:
                continue
            rows.append({
                "label": m.get("label", label_dir.name),
                "model": m.get("model", model_dir.name),
                "val_auroc": m.get("val", {}).get("auroc"),
                "val_auprc": m.get("val", {}).get("auprc"),
                "test_auroc": m.get("test", {}).get("auroc"),
                "test_auprc": m.get("test", {}).get("auprc"),
                "dir": str(model_dir)
            })
    return pd.DataFrame(rows)

def harvest_calibration(cal_dirs: List[Path]) -> Tuple[pd.DataFrame, Dict[str, dict]]:
    rows = []
    thresholds_map: Dict[str, dict] = {}
    for d in cal_dirs:
        m = _safe_read_json(d / "metrics.json")
        th = _safe_read_json(d / "thresholds.json")
        if m:
            for split in ["validation", "test"]:
                sub = m.get(split, {})
                for variant, md in sub.items():
                    rows.append({
                        "dir": str(d), "split": split, "variant": variant,
                        "auroc": md.get("auroc"), "auprc": md.get("auprc"),
                        "brier": md.get("brier"), "logloss": md.get("logloss"), "ece": md.get("ece")
                    })
        if th:
            thresholds_map[str(d)] = th.get("thresholds", {})
    return pd.DataFrame(rows), thresholds_map

def harvest_explainability(explain_dirs: List[Path], topk: int = 10) -> pd.DataFrame:
    rows = []
    for d in explain_dirs:
        path_csv = d / "explain_global.csv"
        df = _safe_read_csv_or_parquet(path_csv)
        if df is None or df.empty or "feature" not in df.columns:
            continue
        # Ranking preference
        rank_col = None
        for c in ["shap_mean_abs", "perm_importance_mean", "value_abs", "value"]:
            if c in df.columns:
                rank_col = c; break
        if rank_col == "value" and "value_abs" not in df.columns:
            df["value_abs"] = df["value"].abs()
            rank_col = "value_abs"
        sub = df.sort_values(rank_col, ascending=False).head(topk)[["feature", rank_col]].copy()
        for _, r in sub.iterrows():
            rows.append({"dir": str(d), "feature": str(r["feature"]), "rank_value": float(r[rank_col])})
    return pd.DataFrame(rows)

def harvest_fairness(fairness_dirs: List[Path]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    gaps = []; groups = []
    for d in fairness_dirs:
        g1 = _safe_read_csv_or_parquet(d / "parity_gaps.csv")
        g2 = _safe_read_csv_or_parquet(d / "group_metrics.csv")
        if g1 is not None and not g1.empty:
            gg = g1.copy(); gg["dir"] = str(d); gaps.append(gg)
        if g2 is not None and not g2.empty:
            gm = g2.copy(); gm["dir"] = str(d); groups.append(gm)
    return (pd.concat(gaps, ignore_index=True) if gaps else pd.DataFrame(),
            pd.concat(groups, ignore_index=True) if groups else pd.DataFrame())

def harvest_external(external_dirs: List[Path]) -> pd.DataFrame:
    rows = []
    for d in external_dirs:
        m = _safe_read_json(d / "metrics.json")
        if not m:
            continue
        fam = m.get("family", "baseline")
        if fam == "baseline":
            for var, md in m.get("metrics", {}).items():
                rows.append({"dir": str(d), "family": "baseline", "variant": var, **md})
        elif fam == "mtl":
            for task, mtask in m.get("metrics", {}).items():
                for var, md in mtask.items():
                    rows.append({"dir": str(d), "family": "mtl", "task": task, "variant": var, **md})
    return pd.DataFrame(rows)

def harvest_robustness(rob_dirs: List[Path]) -> pd.DataFrame:
    rows = []
    for d in rob_dirs:
        df = _safe_read_csv_or_parquet(d / "robustness_results.csv")
        if df is not None and not df.empty:
            sub = df.copy(); sub["dir"] = str(d); rows.append(sub)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

def cohort_stats(cohort_path: Optional[Path]) -> pd.DataFrame:
    if not cohort_path or not cohort_path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(cohort_path)
    rows = []
    rows.append({"metric": "N patients", "value": int(df["subject_id"].nunique()) if "subject_id" in df.columns else int(len(df))})
    if "index_time" in df.columns:
        years = pd.to_datetime(df["index_time"], errors="coerce").dt.year
        rows.append({"metric": "Index years span", "value": f"{int(years.min())}–{int(years.max())}"})
    if "age_at_admit" in df.columns:
        a = pd.to_numeric(df["age_at_admit"], errors="coerce")
        rows.append({"metric": "Age mean (sd)", "value": f"{a.mean():.1f} ({a.std():.1f})"})
    if "gender" in df.columns:
        vc = df["gender"].astype(str).str.upper().str.strip().value_counts(dropna=False)
        for k, v in vc.items():
            rows.append({"metric": f"Gender {k}", "value": int(v)})
    return pd.DataFrame(rows)

def env_info(env_report: Optional[Path]) -> Dict[str, str]:
    info = {"generated_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z"}
    if env_report and env_report.exists():
        try:
            data = json.loads(env_report.read_text())
            info.update({k: str(v) for k, v in data.items()})
            return info
        except Exception:
            pass
    # Fallback: light introspection
    try:
        import sys, platform, importlib.metadata as md
        info.update({
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "pandas": md.version("pandas"),
            "numpy": md.version("numpy"),
        })
        for pkg in ["scikit-learn", "xgboost", "torch", "shap"]:
            try: info[pkg] = md.version(pkg)
            except Exception: pass
    except Exception:
        pass
    return info

# ----------------------------- Summaries ------------------------------------ #

def summarize_rollup(paper_bundle: Optional[Path]) -> Optional[pd.DataFrame]:
    if not paper_bundle:
        return None
    p = paper_bundle / "tables" / "metrics_rollup.csv"
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p)
        return df
    except Exception:
        return None

def summarize_thresholds(th_map: Dict[str, dict]) -> pd.DataFrame:
    rows = []
    for d, mp in th_map.items():
        for variant, rules in mp.items():
            for rule, thr in rules.items():
                rows.append({"dir": d, "variant": variant, "rule": rule, "threshold": float(thr)})
    return pd.DataFrame(rows)

def summarize_top_features(explain_df: pd.DataFrame, per_dir: int = 10) -> pd.DataFrame:
    if explain_df.empty:
        return explain_df
    # Keep top-k per model dir by rank_value
    out = []
    for d, sub in explain_df.groupby("dir"):
        ss = sub.sort_values("rank_value", ascending=False).head(per_dir)
        ss = ss.assign(model_dir=d)
        out.append(ss[["model_dir", "feature", "rank_value"]])
    return pd.concat(out, ignore_index=True)

def summarize_fairness_gaps(gaps_df: pd.DataFrame) -> pd.DataFrame:
    if gaps_df.empty:
        return gaps_df
    # For each (dir, metric), take worst (max gap) over group_col/rule
    rows = []
    for (d, met), sub in gaps_df.groupby(["dir", "metric"]):
        worst = sub.sort_values("gap_max_min", ascending=False).head(1).iloc[0]
        rows.append({
            "dir": d, "metric": met,
            "worst_gap": float(worst["gap_max_min"]),
            "worst_ratio_min_max": float(worst["ratio_min_max"]),
            "at_group_col": str(worst["group_col"]),
            "rule": str(worst["rule"])
        })
    return pd.DataFrame(rows)

def summarize_robustness(rob_df: pd.DataFrame) -> pd.DataFrame:
    if rob_df.empty:
        return rob_df
    # Compare scenarios to base per (dir, variant, label): report delta AUROC/AUPRC/ECE
    base = rob_df[rob_df["scenario"] == "base"][["dir","variant","label","auroc","auprc","ece"]].rename(
        columns={"auroc":"base_auroc","auprc":"base_auprc","ece":"base_ece"})
    merged = rob_df.merge(base, on=["dir","variant","label"], how="left")
    merged["d_auroc"] = merged["auroc"] - merged["base_auroc"]
    merged["d_auprc"] = merged["auprc"] - merged["base_auprc"]
    merged["d_ece"]   = merged["ece"]   - merged["base_ece"]
    # Keep non-base scenarios
    return merged[merged["scenario"] != "base"][["dir","label","variant","scenario","d_auroc","d_auprc","d_ece","n","pos_rate"]]

# ----------------------------- Markdown builders ---------------------------- #

def md_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df is None or df.empty:
        return "_n/a_\n"
    sub = df.head(max_rows)
    try:
        return sub.to_markdown(index=False) + "\n"
    except Exception:
        return sub.to_csv(index=False)

def build_model_card(project: str,
                     baselines: pd.DataFrame,
                     rollup: Optional[pd.DataFrame],
                     cal_metrics: pd.DataFrame,
                     th_tbl: pd.DataFrame,
                     top_feats: pd.DataFrame,
                     fair_worst: pd.DataFrame,
                     external: pd.DataFrame,
                     robustness: pd.DataFrame,
                     env: Dict[str, str]) -> Tuple[str, dict]:
    lines: List[str] = []
    lines.append(f"# Model Card — {project}\n\n")
    lines.append(f"_Generated: {env.get('generated_utc','')} UTC_\n\n")

    # Overview
    lines.append("## Overview\n")
    lines.append("- **Intended use**: Predict incident comorbid outcomes for research and hypothesis generation.\n")
    lines.append("- **Out-of-scope**: Real-time clinical decision-making without clinical validation.\n")
    lines.append("- **Targets/labels available**: from labels file; see Data Card.\n\n")

    # Performance
    lines.append("## Performance summary\n")
    if rollup is not None and not rollup.empty:
        lines.append("### TEST metrics (roll-up)\n")
        lines.append(md_table(rollup[["label","method","test_auroc","test_auprc"]].sort_values(["label","method"])))
    elif not baselines.empty:
        lines.append("### Baselines (VAL/TEST)\n")
        lines.append(md_table(baselines.sort_values(["label","model"])))
    else:
        lines.append("_No baseline metrics found._\n\n")

    # Calibration & thresholds
    if not cal_metrics.empty or not th_tbl.empty:
        lines.append("## Calibration & thresholds\n")
        if not cal_metrics.empty:
            lines.append("Calibration metrics (validation/test by variant):\n")
            lines.append(md_table(cal_metrics.sort_values(["dir","split","variant"])))
        if not th_tbl.empty:
            lines.append("Operating thresholds (selected on validation):\n")
            lines.append(md_table(th_tbl.sort_values(["dir","variant","rule"])))
        lines.append("\n")

    # Explainability
    if not top_feats.empty:
        lines.append("## Top features (global importance)\n")
        lines.append(md_table(top_feats.groupby("model_dir").head(10)))
        lines.append("\n")

    # Fairness
    if not fair_worst.empty:
        lines.append("## Fairness snapshot (worst observed gaps across groups)\n")
        lines.append(md_table(fair_worst.sort_values(["dir","metric"])))
        lines.append("\n")

    # External validation
    if not external.empty:
        lines.append("## External validation\n")
        cols = ["dir","family","task","variant","auroc","auprc","brier","logloss","ece","n","pos_rate"]
        keep = [c for c in cols if c in external.columns]
        lines.append(md_table(external[keep].sort_values(keep)))
        lines.append("\n")

    # Robustness
    if not robustness.empty:
        lines.append("## Robustness highlights (delta vs base)\n")
        lines.append(md_table(robustness.sort_values(["dir","label","variant","scenario"])))
        lines.append("\n")

    # Risks & limitations
    lines.append("## Risks, limitations, and mitigations\n")
    lines.append("- Potential dataset shift and selection bias; review external validation metrics above.\n")
    lines.append("- Calibration may degrade in new settings; consider local recalibration.\n")
    lines.append("- Feature importances reflect correlations, not causation.\n\n")

    # Environment
    lines.append("## Environment & reproducibility\n")
    env_tbl = pd.DataFrame(sorted(env.items()), columns=["key","value"])
    lines.append(md_table(env_tbl))

    payload = {
        "project": project,
        "generated_utc": env.get("generated_utc"),
        "performance": rollup.to_dict(orient="records") if rollup is not None and not rollup.empty else baselines.to_dict(orient="records"),
        "calibration_metrics": cal_metrics.to_dict(orient="records"),
        "thresholds": th_tbl.to_dict(orient="records"),
        "top_features": top_feats.to_dict(orient="records"),
        "fairness_worst": fair_worst.to_dict(orient="records"),
        "external": external.to_dict(orient="records"),
        "robustness": robustness.to_dict(orient="records"),
        "environment": env
    }
    return "".join(lines), payload

def build_data_card(project: str,
                    cohort: pd.DataFrame,
                    labels_df: Optional[pd.DataFrame]) -> Tuple[str, dict]:
    lines: List[str] = []
    lines.append(f"# Data Card — {project}\n\n")
    lines.append(f"_Generated: {datetime.utcnow().isoformat(timespec='seconds')}Z_\n\n")
    lines.append("## Summary\n")
    lines.append("Longitudinal EHR-derived cohort for comorbidity risk modeling; index defined per 01_extract_cohort.\n\n")
    if not cohort.empty:
        lines.append("## Cohort summary\n")
        lines.append(md_table(cohort))
    else:
        lines.append("_Cohort summary not available._\n\n")
    if labels_df is not None and not labels_df.empty:
        lines.append("## Labels present (columns)\n")
        cols = [c for c in labels_df.columns if c != "subject_id"]
        head = pd.DataFrame({"label_column": cols})
        lines.append(md_table(head))
    lines.append("## Provenance & preprocessing\n")
    lines.append("- See numbered scripts (00–16) for extraction, cleaning, imputation, and splits.\n")
    lines.append("- Protected health information removed or pseudonymized; use governed by IRB/data use agreements where applicable.\n\n")
    payload = {
        "project": project,
        "cohort_summary": cohort.to_dict(orient="records") if not cohort.empty else [],
        "label_columns": [c for c in labels_df.columns if c != "subject_id"] if labels_df is not None else []
    }
    return "".join(lines), payload

# ----------------------------- MAIN ----------------------------------------- #

def main() -> int:
    args = parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # Collect artifacts
    baselines = harvest_baselines(Path(args.baselines_root)) if args.baselines_root else pd.DataFrame()
    cal_metrics, th_map = harvest_calibration(_split_paths(args.calibration_dirs)) if args.calibration_dirs else (pd.DataFrame(), {})
    explain = harvest_explainability(_split_paths(args.explain_dirs)) if args.explain_dirs else pd.DataFrame()
    fairness_gaps, fairness_groups = harvest_fairness(_split_paths(args.fairness_dirs)) if args.fairness_dirs else (pd.DataFrame(), pd.DataFrame())
    external = harvest_external(_split_paths(args.external_dirs)) if args.external_dirs else pd.DataFrame()
    robustness = harvest_robustness(_split_paths(args.robustness_dirs)) if args.robustness_dirs else pd.DataFrame()
    rollup = summarize_rollup(Path(args.paper-bundle)) if args.paper_bundle else None  # type: ignore[attr-defined]

    # Summaries for cards
    th_tbl = summarize_thresholds(th_map) if th_map else pd.DataFrame()
    top_feats = summarize_top_features(explain, per_dir=10) if not explain.empty else pd.DataFrame()
    fair_worst = summarize_fairness_gaps(fairness_gaps) if not fairness_gaps.empty else pd.DataFrame()

    # Environment
    env = env_info(Path(args.env_report) if args.env_report else None)

    # Data card inputs
    cohort_tbl = cohort_stats(Path(args.cohort_file)) if args.cohort_file else pd.DataFrame()
    labels_tbl = None
    if args.labels_file and Path(args.labels_file).exists():
        try:
            labels_tbl = pd.read_parquet(args.labels_file)
        except Exception:
            try:
                labels_tbl = pd.read_csv(args.labels_file)
            except Exception:
                labels_tbl = None

    # Build cards
    model_md, model_json = build_model_card(
        project=args.project_name,
        baselines=baselines,
        rollup=rollup,
        cal_metrics=cal_metrics,
        th_tbl=th_tbl,
        top_feats=top_feats,
        fair_worst=fair_worst,
        external=external,
        robustness=summarize_robustness(robustness) if not robustness.empty else pd.DataFrame(),
        env=env
    )
    data_md, data_json = build_data_card(
        project=args.project_name,
        cohort=cohort_tbl,
        labels_df=labels_tbl
    )

    # Write outputs
    (outdir / "model_card.md").write_text(model_md)
    (outdir / "model_card.json").write_text(json.dumps(model_json, indent=2))
    (outdir / "data_card.md").write_text(data_md)
    (outdir / "data_card.json").write_text(json.dumps(data_json, indent=2))

    print(f"Wrote model & data cards to: {outdir}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

