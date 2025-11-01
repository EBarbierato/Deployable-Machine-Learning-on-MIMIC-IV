#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python3
"""
19_release_packager.py

Purpose
-------
Assemble a clean, reproducible release bundle for journal submission or archive:
- Gathers selected artifacts (tables/figures/bundles/cards/CI reports/configs).
- Writes MANIFEST (paths, sizes, SHA-256, category) + checksums file.
- Copies files into a deterministic staging dir, then creates .zip / .tar.gz.
- Optional gating: fail if CI (script 18) reported errors (or warnings in strict).
- Optional inclusion of numbered scripts and config(s) for reproducibility.

Default layout
--------------
dist/
  <project>-<version>/
    README_RELEASE.md
    MANIFEST.json
    checksums.sha256
    LICENSE            # if provided
    cards/             # model & data cards
    paper/             # from 11_paper_bundle.py
    tables/            # CSV tables (metrics_rollup, group metrics, etc.)
    figures/           # PNG/SVG (decision curves, etc.)
    ci/                # CI summary/report from 18
    configs/           # pipeline.yaml, etc. (optional)
    scripts/           # numbered scripts (optional; excludes large ckpts)

Inputs
------
--project-name         Short slug, used in archive name (default "comorbidity")
--version              Semver-like or date stamp (default auto: YYYYMMDD)
--artifacts-root       Root "artifacts" directory (default artifacts)
--cards-dir            Path to cards (from 17), default docs/cards
--paper-bundle         Path to 11_paper_bundle.py outdir (default artifacts/paper_bundle)
--include-configs      Comma globs of configs to include (e.g., configs/*.yaml)
--include-scripts      If set, copy numbered scripts/ (excludes *.ckpt, *.pt, large files)
--license-file         Optional LICENSE path to include
--require-clean-ci     If set, read artifacts/ci/ci_report.json and abort on errors (or warns with --strict-ci)
--strict-ci            Escalate WARN to failure if --require-clean-ci is set
--formats              Archive formats to write (zip,tar.gz) (default zip,tar.gz)
--max-file-mb          Skip files larger than this size (default 50)
--extra-include        Extra comma globs to include (e.g., "artifacts/fairness/**/*.csv")
--outdir               Output directory for bundles (default dist)

Examples
--------
python scripts/19_release_packager.py \
  --project-name comorbidity_mtl \
  --version 1.0.0 \
  --include-configs configs/pipeline.yaml \
  --include-scripts \
  --require-clean-ci \
  --outdir dist
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tarfile
import zipfile
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# ----------------------------- CLI ------------------------------------------ #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Package a clean release bundle with manifest and checksums.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--project-name", type=str, default="comorbidity")
    p.add_argument("--version", type=str, default=datetime.utcnow().strftime("%Y%m%d"))
    p.add_argument("--artifacts-root", type=str, default="artifacts")
    p.add_argument("--cards-dir", type=str, default="docs/cards")
    p.add_argument("--paper-bundle", type=str, default="artifacts/paper_bundle")
    p.add_argument("--include-configs", type=str, default="", help="Comma globs (e.g., configs/*.yaml,*.json)")
    p.add_argument("--include-scripts", action="store_true", help="Include numbered scripts/ directory")
    p.add_argument("--license-file", type=str, default=None)
    p.add_argument("--require-clean-ci", action="store_true", help="Abort if CI report has errors (or warns with --strict-ci)")
    p.add_argument("--strict-ci", action="store_true", help="With --require-clean-ci, treat WARN as failure too")
    p.add_argument("--formats", type=str, default="zip,tar.gz", help="Comma list: zip, tar.gz")
    p.add_argument("--max-file-mb", type=int, default=50, help="Skip files larger than this size")
    p.add_argument("--extra-include", type=str, default="", help="Comma globs to include additional files")
    p.add_argument("--outdir", type=str, default="dist")
    return p.parse_args()

# ----------------------------- Data classes --------------------------------- #

@dataclass
class Entry:
    relpath: str
    bytes: int
    sha256: str
    category: str

# ----------------------------- Helpers -------------------------------------- #

def sha256_of(path: Path, bufsize: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(bufsize)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def gather_globs(globs: Iterable[str]) -> List[Path]:
    out: List[Path] = []
    for g in globs:
        for p in Path().glob(g):
            if p.is_file():
                out.append(p.resolve())
    return out

def within_size(p: Path, max_mb: int) -> bool:
    try:
        return (p.stat().st_size <= max_mb * 1024 * 1024)
    except FileNotFoundError:
        return False

def copy_into(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)

def write_zip(src_dir: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(src_dir.rglob("*")):
            if p.is_file():
                zf.write(p, arcname=str(p.relative_to(src_dir)))

def write_targz(src_dir: Path, tgz_path: Path) -> None:
    with tarfile.open(tgz_path, "w:gz") as tf:
        tf.add(src_dir, arcname=src_dir.name)

def load_ci_counts(ci_json: Path) -> Optional[Dict[str, int]]:
    if not ci_json.exists():
        return None
    try:
        meta = json.loads(ci_json.read_text())
        return meta.get("counts", None)
    except Exception:
        return None

# ----------------------------- Core logic ----------------------------------- #

def default_includes(artifacts_root: Path, cards_dir: Path, paper_bundle: Path) -> Dict[str, List[Path]]:
    cats: Dict[str, List[Path]] = {k: [] for k in ["cards", "paper", "tables", "figures", "ci"]}
    # Cards
    for fn in ["model_card.md", "data_card.md", "model_card.json", "data_card.json"]:
        p = cards_dir / fn
        if p.exists():
            cats["cards"].append(p.resolve())
    # Paper bundle (tables/figures)
    if paper_bundle.exists():
        # tables
        for p in (paper_bundle / "tables").glob("*"):
            if p.is_file():
                cats["tables"].append(p.resolve())
        # figures
        for p in (paper_bundle / "figures").glob("*"):
            if p.is_file():
                cats["figures"].append(p.resolve())
        # the PDF/TeX if present
        for g in ["*paper*.pdf", "*.tex", "*.bib"]:
            for p in paper_bundle.glob(g):
                if p.is_file():
                    cats["paper"].append(p.resolve())
    # Decision curves / Fairness extras (csv/png)
    for sub in ["dca", "fairness", "external_validation", "robustness"]:
        d = artifacts_root / sub
        if not d.exists():
            continue
        for p in d.glob("**/*"):
            if p.is_file() and (p.suffix.lower() in {".csv", ".png", ".svg", ".json", ".md"}):
                cats["figures" if p.suffix.lower() in {".png", ".svg"} else "tables"].append(p.resolve())
    # CI reports
    ci = artifacts_root / "ci"
    for fn in ["ci_report.json", "SUMMARY.md"]:
        p = ci / fn
        if p.exists():
            cats["ci"].append(p.resolve())
    return cats

def scripts_to_include(max_file_mb: int) -> List[Path]:
    out: List[Path] = []
    sdir = Path("scripts")
    if not sdir.exists():
        return out
    deny_ext = {".ckpt", ".pt", ".pth", ".onnx", ".bin"}
    for p in sdir.glob("**/*"):
        if not p.is_file():
            continue
        if p.suffix.lower() in deny_ext:
            continue
        if not within_size(p, max_file_mb):
            continue
        out.append(p.resolve())
    return out

def configs_to_include(globs_csv: str, max_file_mb: int) -> List[Path]:
    if not globs_csv:
        return []
    globs = [g.strip() for g in globs_csv.split(",") if g.strip()]
    files = gather_globs(globs)
    return [p for p in files if within_size(p, max_file_mb)]

def filter_large(files: List[Path], max_file_mb: int) -> List[Path]:
    return [p for p in files if within_size(p, max_file_mb)]

def stage_bundle(
    name: str,
    version: str,
    cats: Dict[str, List[Path]],
    cfgs: List[Path],
    scripts: List[Path],
    license_path: Optional[Path],
    max_file_mb: int,
    outdir: Path
) -> Tuple[Path, List[Entry]]:
    stage_root = outdir / f"{name}-{version}"
    if stage_root.exists():
        shutil.rmtree(stage_root)
    stage_root.mkdir(parents=True, exist_ok=True)

    manifest: List[Entry] = []

    # Category folders
    submap = {
        "cards": "cards",
        "paper": "paper",
        "tables": "tables",
        "figures": "figures",
        "ci": "ci",
    }

    for cat, files in cats.items():
        subdir = stage_root / submap[cat]
        for src in filter_large(files, max_file_mb):
            rel = Path(submap[cat]) / src.name
            dst = stage_root / rel
            copy_into(src, dst)
            manifest.append(Entry(relpath=str(rel), bytes=dst.stat().st_size, sha256=sha256_of(dst), category=cat))

    # Configs
    if cfgs:
        cdir = stage_root / "configs"
        for src in cfgs:
            rel = Path("configs") / src.name
            dst = stage_root / rel
            copy_into(src, dst)
            manifest.append(Entry(relpath=str(rel), bytes=dst.stat().st_size, sha256=sha256_of(dst), category="configs"))

    # Scripts
    if scripts:
        sroot = stage_root / "scripts"
        for src in scripts:
            rel = Path("scripts") / src.relative_to(Path("scripts"))
            dst = stage_root / rel
            copy_into(src, dst)
            manifest.append(Entry(relpath=str(rel), bytes=dst.stat().st_size, sha256=sha256_of(dst), category="scripts"))

    # LICENSE
    if license_path:
        lp = Path(license_path)
        if lp.exists() and lp.is_file():
            rel = Path("LICENSE")
            dst = stage_root / rel
            copy_into(lp, dst)
            manifest.append(Entry(relpath=str(rel), bytes=dst.stat().st_size, sha256=sha256_of(dst), category="legal"))

    # README_RELEASE.md
    readme = stage_root / "README_RELEASE.md"
    readme.write_text(
        f"# Release: {name} v{version}\n\n"
        f"Generated: {datetime.utcnow().isoformat(timespec='seconds')}Z\n\n"
        "This bundle contains:\n"
        "- Cards (model/data),\n"
        "- Paper bundle (tables/figures),\n"
        "- Key CSV/JSON/PNG artifacts (DCA, fairness, external, robustness),\n"
        "- CI report, and optionally configs & scripts.\n\n"
        "## Reproduction (high-level)\n"
        "1. Create environment with `scripts/00_make_env.py` (or see cards' env section).\n"
        "2. Use `scripts/16_pipeline_orchestrator.py` with your `configs/pipeline.yaml`.\n"
        "3. Verify outputs with `scripts/18_ci_checks.py`.\n\n"
        "## Notes\n"
        "- Large or sensitive raw data are NOT included.\n"
        "- See MANIFEST.json and checksums.sha256 for integrity.\n"
    )

    # MANIFEST.json
    manifest_json = [asdict(m) for m in manifest]
    (stage_root / "MANIFEST.json").write_text(json.dumps(manifest_json, indent=2))

    # checksums.sha256
    ck = stage_root / "checksums.sha256"
    with ck.open("w") as f:
        for m in manifest:
            f.write(f"{m.sha256}  {m.relpath}\n")

    return stage_root, manifest

def make_archives(stage_root: Path, formats: List[str]) -> List[Path]:
    outs: List[Path] = []
    base = stage_root.parent / stage_root.name
    if "zip" in formats:
        zp = base.with_suffix(".zip")
        if zp.exists():
            zp.unlink()
        write_zip(stage_root, zp)
        outs.append(zp)
    if "tar.gz" in formats or "tgz" in formats:
        tgz = base.with_suffix(".tar.gz")
        if tgz.exists():
            tgz.unlink()
        write_targz(stage_root, tgz)
        outs.append(tgz)
    return outs

# ----------------------------- MAIN ----------------------------------------- #

def main() -> int:
    args = parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    artifacts_root = Path(args.artifacts_root).resolve()
    cards_dir = Path(args.cards_dir).resolve()
    paper_bundle = Path(args.paper_bundle).resolve()

    # Optional CI gate
    if args.require_clean_ci:
        ci_counts = load_ci_counts(artifacts_root / "ci" / "ci_report.json")
        if ci_counts is None:
            print("[packager] ERROR: --require-clean-ci set but no CI report found at artifacts/ci/ci_report.json")
            return 2
        errs = int(ci_counts.get("ERROR", 0))
        warns = int(ci_counts.get("WARN", 0))
        if errs > 0 or (args.strict_ci and warns > 0):
            print(f"[packager] ERROR: CI not clean (ERROR={errs}, WARN={warns}, strict={args.strict_ci})")
            return 3  # noqa: E999 (hyphen used above intentionally in f-string)
        print(f"[packager] CI OK (ERROR={errs}, WARN={warns}, strict={args.strict_ci})")

    # Collect defaults
    cats = default_includes(artifacts_root, cards_dir, paper_bundle)

    # Extras
    extra_files = []
    if args.extra_include:
        extra_files = gather_globs([g.strip() for g in args.extra_include.split(",") if g.strip()])
        if extra_files:
            cats.setdefault("tables", [])
            for p in extra_files:
                cats["tables"].append(p.resolve())

    cfgs = configs_to_include(args.include_configs, args.max_file_mb)
    scr = scripts_to_include(args.max_file_mb) if args.include_scripts else []

    # LICENSE
    lic = Path(args.license_file).resolve() if args.license_file else None

    # Stage bundle
    stage_root, manifest = stage_bundle(
        name=args.project_name,
        version=args.version,
        cats=cats,
        cfgs=cfgs,
        scripts=scr,
        license_path=lic,
        max_file_mb=int(args.max_file_mb),
        outdir=outdir
    )

    # Archives
    formats = [s.strip().lower() for s in args.formats.split(",") if s.strip()]
    outs = make_archives(stage_root, formats)

    print(f"[packager] Staged at: {stage_root}")
    for p in outs:
        print(f"[packager] Wrote: {p}  ({p.stat().st_size/1e6:.2f} MB)")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

