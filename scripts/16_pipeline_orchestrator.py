#!/usr/bin/env python3
"""
16_pipeline_orchestrator.py  —  unified runner for the numbered pipeline

Purpose
-------
Run the full research pipeline via a single, reproducible entry point.
Stages (toggle via config and/or --skip):
  0) (optional) Ingest (01)
  1) Train baselines (06)
  2) Calibrate & thresholds (09)
  3) Explainability (10)
  4) Paper bundle (11)
  5) Decision curves (12)
  6) Fairness audit (13)
  7) External validation (14)
  8) Robustness checks (15)
  9) (optional) MTL downstream helpers (07/08 produced elsewhere)

Config (YAML or JSON)
---------------------
See the example block inside this file for all supported keys.
Only the sections you enable will run.

Usage
-----
python scripts/16_pipeline_orchestrator.py --config configs/pipeline.yaml
  [--force] [--dry-run] [--python-bin python] [--skip paper,external]

Notes
-----
- Missing scripts are detected and the stage is skipped with a notice.
- Downstream stages prefer calibrated predictions when available.
- Works with both relative and absolute paths; all outputs are created if needed.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

# -----------------------------------------------------------------------------
# Config helpers
# -----------------------------------------------------------------------------

EXAMPLE_CFG = """
project:
  name: "comorbidity_mimiciv"
  out_root: "/home_local/corte/comorbidity_artifacts"

data:
  cohort: "/home_local/corte/comorbidity_artifacts/cohort/cohort.parquet"
  labels: "/home_local/corte/comorbidity_artifacts/labels/labels.parquet"
  imputed:
    train: "/home_local/corte/comorbidity_artifacts/imputed/X_train.parquet"
    val:   "/home_local/corte/comorbidity_artifacts/imputed/X_val.parquet"
    test:  "/home_local/corte/comorbidity_artifacts/imputed/X_test.parquet"

labels:
  primary: "diab_incident_365d"
  secondary: ""   # optional

baselines:
  enable: true
  models: ["logistic", "random_forest"]

calibration:
  enable: true
  methods: ["platt", "isotonic"]

explain:
  enable: true
  metric: "auprc"
  top_k: 30
  shap: true
  shap_n: 2000
  shap_bg: 200

decision_curves:
  enable: true
  grid: "0.01:0.99:0.01"
  bootstrap: 0

fairness:
  enable: false
  group_cols: ["sex"]
  age_bins: 5
  numeric_bins: 0
  use_calibrated: true

external:
  enable: false
  x_ext: "data_ext/X_external.parquet"
  labels_ext: "data_ext/labels_external.parquet"

robustness:
  enable: false
  ablate_from_explain: ""
  ablate_topk: 0
  ablate_strategy: "mean"
  noise_levels: []
  time_meta: "/home_local/corte/comorbidity_artifacts/cohort/cohort.parquet"
  time_col: "index_time"
  slice_by: null  # year | quarter | month

mtl:
  enable: false
  run_dirs: []
  task_cols: []
"""

def load_config(path: Path) -> dict:
    text = path.read_text()
    try:
        import yaml  # type: ignore
        return yaml.safe_load(text)
    except Exception:
        return json.loads(text)

def as_list(x, allow_str: bool = False) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(v) for v in x]
    if allow_str and isinstance(x, str) and x:
        return [x]
    return [str(x)]

# -----------------------------------------------------------------------------
# Orchestrator core
# -----------------------------------------------------------------------------

@dataclass
class Ctx:
    cfg: dict
    root: Path
    py: str
    dry: bool
    force: bool
    skip: List[str]

def script_exists(name: str) -> bool:
    return Path(f"scripts/{name}").exists()

def run_cmd(ctx: Ctx, cmd: List[str], must_exist: Optional[Path] = None,
            creates: Optional[List[Path]] = None) -> None:
    if must_exist and not must_exist.exists():
        print(f"[orchestrator] WARN: input not found, skipping stage: {must_exist}")
        return
    if creates and (not ctx.force):
        if all(p.exists() for p in creates):
            print(f"[orchestrator] SKIP (exists): {' '.join(shlex.quote(c) for c in cmd)}")
            return
    print(f"[orchestrator] RUN: {' '.join(shlex.quote(c) for c in cmd)}")
    if ctx.dry:
        return
    subprocess.run(cmd, check=True)

# Paths layout helpers
def baseline_dir(root: Path, label: str, model: str) -> Path:
    return root / "baselines" / label / model

def cal_dir_for(bdir: Path) -> Path:
    return bdir.parent.parent / "baselines" / "calibration" / f"{bdir.name}_{bdir.parent.name}"

def explain_dir_for(bdir: Path) -> Path:
    return bdir.parent.parent / "explain" / f"{bdir.parent.name}_{bdir.name}"

def dca_dir_for(bdir: Path, calibrated: bool) -> Path:
    tag = "cal" if calibrated else "raw"
    return bdir.parent.parent / "dca" / f"{bdir.parent.name}_{bdir.name}_{tag}"

def fairness_dir_for(bdir: Path, calibrated: bool) -> Path:
    tag = "cal" if calibrated else "raw"
    return bdir.parent.parent / "fairness" / f"{bdir.parent.name}_{bdir.name}_{tag}"

# -----------------------------------------------------------------------------
# Stage wrappers
# -----------------------------------------------------------------------------

def stage_ingest(ctx: Ctx, ingest_cfg: dict) -> None:
    if not ingest_cfg.get("enable", False) or ("ingest" in ctx.skip):
        return
    if not script_exists("01_ingest_mimic_hosp.py"):
        print("[orchestrator] NOTE: 01_ingest_mimic_hosp.py not found, skipping ingest.")
        return
    out_root = Path(ingest_cfg.get("out_root", "artifacts/raw_parquet/hosp")).resolve()
    expected = [out_root / "INGEST_SUMMARY.json"]
    cmd = [
        ctx.py, "scripts/01_ingest_mimic_hosp.py",
        "--input-root", ingest_cfg["input_root"],
        "--tables", ingest_cfg.get("tables", ""),
        "--out-root", str(out_root),
        "--chunk-rows", str(int(ingest_cfg.get("chunk_rows", 2_000_000))),
        "--partition-labevents", ingest_cfg.get("partition_labevents", "none"),
    ]
    if bool(ingest_cfg.get("overwrite", False)):
        cmd += ["--overwrite"]
    run_cmd(ctx, cmd, creates=expected)

def stage_baseline(ctx: Ctx, label: str, model: str, X: dict, labels_path: Path) -> Path:
    if ("baselines" in ctx.skip) or (not ctx.cfg.get("baselines", {}).get("enable", True)):
        return baseline_dir(ctx.root, label, model)
    if not script_exists("06_train_baselines.py"):
        print("[orchestrator] NOTE: 06_train_baselines.py not found, skipping baselines.")
        return baseline_dir(ctx.root, label, model)

    outdir = baseline_dir(ctx.root, label, model)
    outdir.mkdir(parents=True, exist_ok=True)
    expected = [
        outdir / "model.joblib",
        outdir / "metrics.json",
        outdir / "val_preds.parquet",
        outdir / "test_preds.parquet",
    ]
    cmd = [
        ctx.py, "scripts/06_train_baselines.py",
        "--x-train", X["train"],
        "--x-val",   X["val"],
        "--x-test",  X["test"],
        "--labels", str(labels_path),
        "--label-col", label,
        "--model", model,
        "--outdir", str(outdir),
    ]
    run_cmd(ctx, cmd, must_exist=Path(X["train"]), creates=expected)
    return outdir

def stage_calibration(ctx: Ctx, bdir: Path, methods: List[str]) -> Path:
    if ("calibration" in ctx.skip) or (not ctx.cfg.get("calibration", {}).get("enable", True)):
        return cal_dir_for(bdir)
    if not script_exists("09_calibrate_and_thresholds.py"):
        print("[orchestrator] NOTE: 09_calibrate_and_thresholds.py not found, skipping calibration.")
        return cal_dir_for(bdir)

    outdir = cal_dir_for(bdir)
    outdir.mkdir(parents=True, exist_ok=True)
    expected = [
        outdir / "calibrated_val.parquet",
        outdir / "calibrated_test.parquet",
        outdir / "metrics.json",
        outdir / "thresholds.json",
        outdir / "calibration_params.json",
    ]
    cmd = [
        ctx.py, "scripts/09_calibrate_and_thresholds.py",
        "--val-preds", str(bdir / "val_preds.parquet"),
        "--test-preds", str(bdir / "test_preds.parquet"),
        "--methods", ",".join(methods or ["platt", "isotonic"]),
        "--outdir", str(outdir),
    ]
    run_cmd(ctx, cmd, must_exist=bdir / "test_preds.parquet", creates=expected)
    return outdir

def stage_explain(ctx: Ctx, bdir: Path, label: str, model: str, X_val: Path, labels_path: Path) -> Path:
    if ("explain" in ctx.skip) or (not ctx.cfg.get("explain", {}).get("enable", True)):
        return explain_dir_for(bdir)
    if not script_exists("10_explainability.py"):
        print("[orchestrator] NOTE: 10_explainability.py not found, skipping explainability.")
        return explain_dir_for(bdir)

    ex_cfg = ctx.cfg.get("explain", {})
    outdir = explain_dir_for(bdir)
    outdir.mkdir(parents=True, exist_ok=True)
    expected = [outdir / "explain_global.csv", outdir / "EXPLAIN.md"]
    cmd = [
        ctx.py, "scripts/10_explainability.py",
        "--x-val", str(X_val),
        "--labels", str(labels_path),
        "--label-col", label,
        "--model-type", model if model in {"logistic", "random_forest", "xgboost"} else "logistic",
        "--model-path", str(bdir / "model.joblib"),
        "--metric", ex_cfg.get("metric", "auprc"),
        "--top-k", str(ex_cfg.get("top_k", 30)),
        "--shap", str(ex_cfg.get("shap", True)).lower(),
        "--shap-n-sample", str(ex_cfg.get("shap_n", 2000)),
        "--shap-bg", str(ex_cfg.get("shap_bg", 200)),
        "--outdir", str(outdir),
    ]
    run_cmd(ctx, cmd, must_exist=bdir / "model.joblib", creates=expected)
    return outdir

def stage_paper_bundle(ctx: Ctx, label: str, model: str) -> None:
    if ("paper" in ctx.skip):
        return
    if not script_exists("11_paper_bundle.py"):
        print("[orchestrator] NOTE: 11_paper_bundle.py not found, skipping paper bundle.")
        return

    outdir = ctx.root / "paper_bundle"
    outdir.mkdir(parents=True, exist_ok=True)
    bdir = baseline_dir(ctx.root, label, model)
    cmd = [
        ctx.py, "scripts/11_paper_bundle.py",
        "--cohort-file", str(Path(ctx.cfg["data"].get("cohort", ctx.cfg["data"]["labels"])).resolve()),
        "--labels-file", str(Path(ctx.cfg["data"]["labels"]).resolve()),
        "--baselines-root", str((ctx.root / "baselines").resolve()),
        "--calibration-dirs", str(cal_dir_for(bdir)),
        "--explain-dirs", str(explain_dir_for(bdir)),
        "--tasks", ",".join([label] + as_list(ctx.cfg.get("labels", {}).get("secondary", None), allow_str=True)),
        "--outdir", str(outdir),
    ]
    run_cmd(ctx, cmd, creates=[outdir / "tables" / "metrics_rollup.csv"])

def stage_dca(ctx: Ctx, preds_path: Path, outdir: Path, grid: str, bootstrap: int, task_col: Optional[str] = None) -> Path:
    if ("decision_curves" in ctx.skip) or (not ctx.cfg.get("decision_curves", {}).get("enable", True)):
        return outdir
    if not script_exists("12_decision_curves.py"):
        print("[orchestrator] NOTE: 12_decision_curves.py not found, skipping DCA.")
        return outdir

    outdir.mkdir(parents=True, exist_ok=True)
    expected = [outdir / "dca_curve.csv", outdir / "decision_curve.png"]
    cmd = [
        ctx.py, "scripts/12_decision_curves.py",
        "--preds", str(preds_path),
        "--grid", grid,
        "--bootstrap", str(int(bootstrap)),
        "--outdir", str(outdir),
    ]
    if task_col:
        cmd += ["--task-col", task_col]
    run_cmd(ctx, cmd, must_exist=preds_path, creates=expected)
    return outdir

def stage_fairness(ctx: Ctx, preds_path: Path, meta_path: Path, group_cols: List[str],
                   age_bins: int, numeric_bins: int, outdir: Path, thresholds_json: Optional[Path] = None,
                   task_col: Optional[str] = None) -> Path:
    if ("fairness" in ctx.skip) or (not ctx.cfg.get("fairness", {}).get("enable", False)):
        return outdir
    if not script_exists("13_fairness_audit.py"):
        print("[orchestrator] NOTE: 13_fairness_audit.py not found, skipping fairness.")
        return outdir

    outdir.mkdir(parents=True, exist_ok=True)
    expected = [outdir / "group_metrics.csv", outdir / "SUMMARY.md"]
    cmd = [
        ctx.py, "scripts/13_fairness_audit.py",
        "--preds", str(preds_path),
        "--meta-file", str(meta_path),
        "--group-cols", ",".join(group_cols),
        "--age-bins", str(int(age_bins)),
        "--numeric-bins", str(int(numeric_bins)),
        "--outdir", str(outdir),
    ]
    if thresholds_json and thresholds_json.exists():
        cmd += ["--thresholds-json", str(thresholds_json)]
    if task_col:
        cmd += ["--task-col", task_col]
    run_cmd(ctx, cmd, must_exist=preds_path, creates=expected)
    return outdir

def stage_external(ctx: Ctx, b_or_mtl: str, outdir: Path, **kwargs) -> None:
    if ("external" in ctx.skip) or (not ctx.cfg.get("external", {}).get("enable", False)):
        return
    if not script_exists("14_external_validation.py"):
        print("[orchestrator] NOTE: 14_external_validation.py not found, skipping external validation.")
        return

    outdir.mkdir(parents=True, exist_ok=True)
    expected = [outdir / "metrics.json", outdir / "METRICS.md"]

    base = [
        ctx.py, "scripts/14_external_validation.py",
        "--x-ext", ctx.cfg["external"]["x_ext"],
        "--labels-ext", ctx.cfg["external"]["labels_ext"],
        "--outdir", str(outdir),
    ]
    if b_or_mtl == "baseline":
        bdir: Path = kwargs["bdir"]
        label: str = kwargs["label"]
        cmd = base + [
            "--baseline-model-path", str(bdir / "model.joblib"),
            "--label-col", label,
        ]
        cal_json = cal_dir_for(bdir) / "calibration_params.json"
        thr_json = cal_dir_for(bdir) / "thresholds.json"
        if cal_json.exists():
            cmd += ["--calibration-params", str(cal_json)]
        if thr_json.exists():
            cmd += ["--thresholds-json", str(thr_json)]
    else:
        run_dir: Path = kwargs["run_dir"]
        task_cols: List[str] = kwargs["task_cols"]
        cmd = base + [
            "--mtl-run-dir", str(run_dir),
            "--task-cols", ",".join(task_cols),
        ]
    run_cmd(ctx, cmd, creates=expected)

def stage_robustness(ctx: Ctx, bdir: Path, label: str, X_test: Path, labels_path: Path) -> None:
    if ("robustness" in ctx.skip) or (not ctx.cfg.get("robustness", {}).get("enable", False)):
        return
    if not script_exists("15_robustness_checks.py"):
        print("[orchestrator] NOTE: 15_robustness_checks.py not found, skipping robustness.")
        return

    outdir = ctx.root / "robustness" / f"{label}_{bdir.name}"
    outdir.mkdir(parents=True, exist_ok=True)
    expected = [outdir / "robustness_results.csv", outdir / "SUMMARY.md"]

    cmd = [
        ctx.py, "scripts/15_robustness_checks.py",
        "--x-test", str(X_test),
        "--labels", str(labels_path),
        "--baseline-model-path", str(bdir / "model.joblib"),
        "--label-col", label,
        "--outdir", str(outdir),
    ]

    # Optional extras
    rob = ctx.cfg.get("robustness", {})
    cal_json = cal_dir_for(bdir) / "calibration_params.json"
    thr_json = cal_dir_for(bdir) / "thresholds.json"
    if cal_json.exists():
        cmd += ["--calibration-params", str(cal_json)]
    if thr_json.exists():
        cmd += ["--thresholds-json", str(thr_json)]
    if rob.get("ablate_from_explain"):
        cmd += ["--ablate-from-explain", str(rob["ablate_from_explain"])]
        if int(rob.get("ablate_topk", 0)) > 0:
            cmd += ["--ablate-topk", str(int(rob["ablate_topk"])),
                    "--ablate-strategy", str(rob.get("ablate_strategy", "mean"))]
    if rob.get("noise_levels"):
        cmd += ["--noise-levels", ",".join(str(x) for x in rob["noise_levels"])]
    if rob.get("time_meta") and rob.get("time_col") and rob.get("slice_by"):
        cmd += ["--meta-file", str(rob["time_meta"]),
                "--time-col", str(rob["time_col"]),
                "--slice-by", str(rob["slice_by"])]

    run_cmd(ctx, cmd, creates=expected)

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Run the numbered pipeline with one command.")
    ap.add_argument("--config", type=str, required=True, help="YAML or JSON config path.")
    ap.add_argument("--python-bin", type=str, default=sys.executable, help="Python interpreter to use.")
    ap.add_argument("--force", action="store_true", help="Overwrite/ignore existing outputs.")
    ap.add_argument("--dry-run", action="store_true", help="Print commands without executing.")
    ap.add_argument("--skip", type=str, default="", help="Comma list of stages to skip (e.g., 'paper,external').")
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    root = Path(cfg.get("project", {}).get("out_root", "artifacts")).resolve()
    root.mkdir(parents=True, exist_ok=True)

    ctx = Ctx(
        cfg=cfg,
        root=root,
        py=args.python_bin,
        dry=args.dry_run,
        force=args.force,
        skip=[s.strip().lower() for s in args.skip.split(",") if s.strip()]
    )

    # Optional ingest
    if cfg.get("ingest", {}).get("enable", False):
        stage_ingest(ctx, cfg["ingest"])

    # Core paths
    data = cfg["data"]
    X = {
        "train": str(Path(data["imputed"]["train"]).resolve()),
        "val":   str(Path(data["imputed"]["val"]).resolve()),
        "test":  str(Path(data["imputed"]["test"]).resolve()),
    }
    labels_path = Path(data["labels"]).resolve()
    cohort_meta = Path(data.get("cohort", labels_path)).resolve()

    label_primary = cfg["labels"]["primary"]
    label_secondary = cfg["labels"].get("secondary")

    # Baselines loop
    models = as_list(cfg.get("baselines", {}).get("models", ["logistic", "random_forest"]))
    for model in models:
        bdir = stage_baseline(ctx, label_primary, model, X, labels_path)

        # Calibration
        cal_methods = as_list(cfg.get("calibration", {}).get("methods", ["platt", "isotonic"]), allow_str=True)
        cal_dir = stage_calibration(ctx, bdir, methods=cal_methods)

        # Explainability
        stage_explain(ctx, bdir=bdir, label=label_primary, model=model,
                      X_val=Path(X["val"]), labels_path=labels_path)

        # Paper bundle (collects everything that exists)
        stage_paper_bundle(ctx, label_primary, model)

        # Preds for downstream (prefer calibrated if exists)
        cal_test = cal_dir / "calibrated_test.parquet"
        preds_for_downstream = cal_test if cal_test.exists() else bdir / "test_preds.parquet"

        # Decision curves
        dc = cfg.get("decision_curves", {})
        stage_dca(ctx, preds_path=preds_for_downstream,
                  outdir=dca_dir_for(bdir, calibrated=cal_test.exists()),
                  grid=dc.get("grid", "0.01:0.99:0.01"),
                  bootstrap=int(dc.get("bootstrap", 0)))

        # Fairness
        fair = cfg.get("fairness", {})
        if fair.get("enable", False):
            stage_fairness(
                ctx,
                preds_path=preds_for_downstream,
                meta_path=cohort_meta,
                group_cols=as_list(fair.get("group_cols", [])),
                age_bins=int(fair.get("age_bins", 0)),
                numeric_bins=int(fair.get("numeric_bins", 0)),
                outdir=fairness_dir_for(bdir, calibrated=cal_test.exists()),
                thresholds_json=(cal_dir / "thresholds.json") if cal_test.exists() else None,
            )

        # External validation (baseline)
        ext = cfg.get("external", {})
        if ext.get("enable", False):
            stage_external(ctx, "baseline",
                           outdir=ctx.root / "external_validation" / f"{label_primary}_{bdir.name}",
                           bdir=bdir, label=label_primary)

        # Robustness
        stage_robustness(ctx, bdir=bdir, label=label_primary,
                         X_test=Path(X["test"]), labels_path=labels_path)

    # Optional MTL helpers
    if cfg.get("mtl", {}).get("enable", False) and ("mtl" not in ctx.skip):
        run_dirs = [Path(p) for p in as_list(cfg["mtl"].get("run_dirs", []))]
        task_cols = as_list(cfg["mtl"].get("task_cols", []))
        dc = cfg.get("decision_curves", {})
        fair = cfg.get("fairness", {})

        for rd in run_dirs:
            # DCA
            if cfg.get("decision_curves", {}).get("enable", True):
                for t in task_cols:
                    stage_dca(ctx,
                              preds_path=rd / "TEST_preds.parquet",
                              outdir=ctx.root / "dca" / f"mtl_{rd.name}_{t}",
                              grid=dc.get("grid", "0.01:0.99:0.01"),
                              bootstrap=int(dc.get("bootstrap", 0)),
                              task_col=t)
            # Fairness
            if fair.get("enable", False):
                for t in task_cols:
                    stage_fairness(ctx,
                                   preds_path=rd / "TEST_preds.parquet",
                                   meta_path=cohort_meta,
                                   group_cols=as_list(fair.get("group_cols", [])),
                                   age_bins=int(fair.get("age_bins", 0)),
                                   numeric_bins=int(fair.get("numeric_bins", 0)),
                                   outdir=ctx.root / "fairness" / f"mtl_{rd.name}_{t}",
                                   thresholds_json=None,
                                   task_col=t)
            # External validation (MTL)
            ext = cfg.get("external", {})
            if ext.get("enable", False):
                stage_external(ctx, "mtl",
                               outdir=ctx.root / "external_validation" / f"mtl_{rd.name}",
                               run_dir=rd, task_cols=task_cols)

    print("\n[orchestrator] Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
