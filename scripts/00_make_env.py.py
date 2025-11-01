#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python3
"""


Purpose
-------
Establish and record a deterministic, auditable runtime environment for the
comorbidity MTL project. This script:
  - Verifies Python version and key library availability (optional).
  - Sets global determinism (Python, NumPy, PyTorch, TensorFlow when present).
  - Captures environment facts (OS, CPU/GPU, library versions, pip freeze).
  - Runs a small reproducibility smoke test.
  - Writes JSON and Markdown reports plus optional pip-freeze snapshot.

Usage
-----
python scripts/00_make_env.py \
  --outdir artifacts/env \
  --seed 42 \
  --expect-gpu false \
  --freeze true

This script does NOT install packages; it records and validates what exists.
"""

from __future__ import annotations
import argparse
import hashlib
import importlib
import json
import os
import platform
import random
import subprocess
import sys
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ------------------------- CLI & CONFIG ------------------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Initialize and record deterministic environment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--outdir", type=str, default="artifacts/env",
                   help="Directory to write reports and logs.")
    p.add_argument("--seed", type=int, default=42, help="Global RNG seed.")
    p.add_argument("--expect-gpu", type=str2bool, default=False,
                   help="If true, fail if no GPU is detected.")
    p.add_argument("--freeze", type=str2bool, default=True,
                   help="Write pip-freeze snapshot.")
    p.add_argument("--fail-on-warnings", type=str2bool, default=False,
                   help="Exit nonzero if compatibility warnings are raised.")
    p.add_argument("--extra-packages", type=str, nargs="*", default=[],
                   help="Additional packages to record versions for.")
    return p.parse_args()

def str2bool(v: str) -> bool:
    return str(v).lower() in {"1", "true", "t", "yes", "y"}

# ------------------------- UTILITIES ---------------------------------------- #

def safe_import_version(pkg: str) -> Optional[str]:
    try:
        mod = importlib.import_module(pkg)
    except Exception:
        return None
    return getattr(mod, "__version__", "unknown")

def run_cmd(cmd: List[str], timeout: int = 15) -> Tuple[int, str, str]:
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return res.returncode, res.stdout.strip(), res.stderr.strip()
    except Exception as e:
        return 1, "", f"{type(e).__name__}: {e}"

def sha256_blob(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

# ------------------------- ENV COLLECTION ----------------------------------- #

def collect_system_facts() -> Dict[str, Any]:
    uname = platform.uname()
    py_impl = platform.python_implementation()
    facts = {
        "timestamp_utc": now_iso(),
        "python": {
            "version": platform.python_version(),
            "implementation": py_impl,
            "executable": sys.executable,
        },
        "os": {
            "system": uname.system,
            "release": uname.release,
            "version": uname.version,
            "machine": uname.machine,
            "processor": uname.processor,
            "platform": platform.platform(),
        },
        "env": {
            "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "TF_DETERMINISTIC_OPS": os.environ.get("TF_DETERMINISTIC_OPS"),
            "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        },
    }
    return facts

def collect_gpu_info() -> Dict[str, Any]:
    g: Dict[str, Any] = {"nvidia_smi": None, "torch": None, "tensorflow": None}

    # nvidia-smi (if available)
    rc, out, _ = run_cmd([
        "nvidia-smi", "--query-gpu=name,memory.total,driver_version",
        "--format=csv,noheader,nounits"
    ])
    if rc == 0 and out:
        lines = [l for l in out.splitlines() if l.strip()]
        devices = []
        for l in lines:
            parts = [p.strip() for p in l.split(",")]
            if len(parts) >= 3:
                devices.append({"name": parts[0], "memory_total_mb": parts[1], "driver": parts[2]})
        g["nvidia_smi"] = {"detected": True, "devices": devices}
    else:
        g["nvidia_smi"] = {"detected": False, "devices": []}

    # torch view
    try:
        import torch  # type: ignore
        torch_info = {
            "installed": True,
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if hasattr(torch, "version") else None,
            "num_devices": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "devices": [],
        }
        if torch_info["cuda_available"]:
            for i in range(torch.cuda.device_count()):
                torch_info["devices"].append({
                    "index": i,
                    "name": torch.cuda.get_device_name(i),
                    "capability": torch.cuda.get_device_capability(i),
                })
        g["torch"] = torch_info
    except Exception:
        g["torch"] = {"installed": False}

    # tensorflow view
    try:
        import tensorflow as tf  # type: ignore
        gpus = tf.config.list_physical_devices("GPU")
        g["tensorflow"] = {
            "installed": True,
            "version": tf.__version__,
            "gpus_detected": len(gpus),
            "gpu_names": [d.name for d in gpus],
        }
    except Exception:
        g["tensorflow"] = {"installed": False}

    return g

def collect_pkg_versions(extra: List[str]) -> Dict[str, Optional[str]]:
    common = [
        "numpy", "pandas", "scipy", "scikit-learn", "xgboost",
        "lightgbm", "catboost", "matplotlib", "seaborn",
        "torch", "tensorflow", "shap", "joblib", "pyyaml"
    ]
    pkgs = sorted(set(common + extra))
    return {p: safe_import_version(p) for p in pkgs}

def get_pip_freeze() -> Tuple[str, str]:
    rc, out, err = run_cmd([sys.executable, "-m", "pip", "freeze"])
    if rc != 0:
        out = f"# pip freeze unavailable\n# stderr:\n{err}\n"
    return out, sha256_blob(out)

# ------------------------- DETERMINISM -------------------------------------- #

def set_global_determinism(seed: int, env_report: Dict[str, Any]) -> List[str]:
    warnings: List[str] = []

    # Python & NumPy
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np  # type: ignore
        np.random.seed(seed)
    except Exception:
        warnings.append("NumPy not available; cannot set NumPy RNG seed.")

    # PyTorch (if installed)
    try:
        import torch  # type: ignore
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Deterministic ops (available on most recent versions)
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            warnings.append("torch.use_deterministic_algorithms not supported; best-effort only.")
        try:
            import torch.backends.cudnn as cudnn  # type: ignore
            cudnn.deterministic = True
            cudnn.benchmark = False
        except Exception:
            pass
        # Optional cuBLAS config for strict determinism on some GPUs
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    except Exception:
        # Torch not present; fine.
        pass

    # TensorFlow (if installed)
    try:
        import tensorflow as tf  # type: ignore
        try:
            tf.random.set_seed(seed)
        except Exception:
            warnings.append("Failed to set TensorFlow random seed.")
        # Prefer API when available (TF 2.13+); else env var
        try:
            from tensorflow.python.framework import config as tfcfg  # type: ignore
            try:
                tfcfg.enable_op_determinism()
            except Exception:
                os.environ["TF_DETERMINISTIC_OPS"] = "1"
        except Exception:
            os.environ["TF_DETERMINISTIC_OPS"] = "1"
    except Exception:
        # TF not present; fine.
        pass

    # Record effective env vars after setting
    env_report["env"].update({
        "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
        "TF_DETERMINISTIC_OPS": os.environ.get("TF_DETERMINISTIC_OPS"),
        "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    })
    return warnings

def reproducibility_smoketest(seed: int) -> Dict[str, Any]:
    """
    Produces fixed random samples across Python, NumPy, and (if present) Torch/TF.
    Returns values and their SHA-256 to verify determinism between runs.
    """
    data: Dict[str, Any] = {"seed": seed, "samples": {}, "hashes": {}}

    random.seed(seed)
    py_vals = [random.random() for _ in range(5)]
    data["samples"]["python_random"] = py_vals

    try:
        import numpy as np  # type: ignore
        np.random.seed(seed)
        np_vals = np.random.RandomState(seed).randn(5).tolist()
        data["samples"]["numpy_random"] = np_vals
    except Exception:
        data["samples"]["numpy_random"] = None

    try:
        import torch  # type: ignore
        torch.manual_seed(seed)
        t = torch.randn(5)
        data["samples"]["torch_random"] = [float(x) for x in t]
    except Exception:
        data["samples"]["torch_random"] = None

    try:
        import tensorflow as tf  # type: ignore
        tf.random.set_seed(seed)
        v = tf.random.normal([5]).numpy().tolist()
        data["samples"]["tf_random"] = v
    except Exception:
        data["samples"]["tf_random"] = None

    # Hashes
    for k, v in data["samples"].items():
        data["hashes"][k] = sha256_blob(json.dumps(v, sort_keys=True))
    data["hashes"]["combined"] = sha256_blob(json.dumps(data["samples"], sort_keys=True))
    return data

# ------------------------- COMPAT WARNINGS ---------------------------------- #

def compatibility_warnings(sys_facts: Dict[str, Any],
                           pkgs: Dict[str, Optional[str]],
                           gpu: Dict[str, Any],
                           expect_gpu: bool) -> List[str]:
    warns: List[str] = []

    # Python version (>=3.10 recommended)
    py_ver = sys_facts["python"]["version"]
    major, minor, *_ = [int(x) for x in py_ver.split(".")]
    if (major, minor) < (3, 10):
        warns.append(f"Python {py_ver} detected; >= 3.10 recommended.")

    # NumPy 2.x with older SHAP sometimes problematic
    npv = pkgs.get("numpy")
    shapv = pkgs.get("shap")
    if npv and shapv:
        try:
            if npv.split(".")[0] == "2":
                # Very rough heuristic; SHAP >=0.45 generally supports NumPy 2
                major_shap = int(shapv.split(".")[0]) if shapv.split(".")[0].isdigit() else 0
                if major_shap < 0:  # placeholder branch; keep logic simple
                    warns.append("Potential NumPy 2.x / SHAP incompatibility.")
        except Exception:
            pass

    # GPU expectations
    if expect_gpu:
        torch_installed = bool(gpu.get("torch", {}).get("installed"))
        tf_installed = bool(gpu.get("tensorflow", {}).get("installed"))
        any_gpu = False
        if torch_installed and gpu["torch"].get("cuda_available"):
            any_gpu = True
        if tf_installed and (gpu["tensorflow"].get("gpus_detected", 0) > 0):
            any_gpu = True
        if not any_gpu:
            warns.append("GPU expected but none detected via Torch/TF/nvidia-smi.")

    # nvidia-smi presence when CUDA claimed
    if gpu.get("torch", {}).get("cuda_available") and not gpu.get("nvidia_smi", {}).get("detected"):
        warns.append("Torch reports CUDA available but nvidia-smi not found; driver tools missing?")

    return warns

# ------------------------- WRITE ARTIFACTS ---------------------------------- #

def write_json(p: Path, obj: Dict[str, Any]) -> None:
    p.write_text(json.dumps(obj, indent=2, sort_keys=True))

def write_markdown(p: Path, env_report: Dict[str, Any], warnings: List[str]) -> None:
    sysf = env_report["system"]
    pkgs = env_report["packages"]
    gpu = env_report["gpu"]
    freeze_hash = env_report.get("pip_freeze", {}).get("sha256")

    lines = []
    lines.append(f"# Environment Report\n\nGenerated: {env_report['system']['timestamp_utc']}\n")
    lines.append("## System\n")
    lines.append(f"- Python: `{sysf['python']['version']} ({sysf['python']['implementation']})`  \n"
                 f"- Executable: `{sysf['python']['executable']}`  \n"
                 f"- OS: `{sysf['os']['platform']}`\n")
    lines.append("## GPU\n")
    lines.append(f"- nvidia-smi: `{gpu['nvidia_smi']['detected']}`  \n"
                 f"- Torch CUDA available: `{gpu['torch'].get('cuda_available', False)}`  \n"
                 f"- TF GPUs detected: `{gpu['tensorflow'].get('gpus_detected', 0)}`\n")
    if gpu["torch"].get("devices"):
        for d in gpu["torch"]["devices"]:
            lines.append(f"  - Torch GPU {d['index']}: {d['name']} (cap {d['capability']})\n")
    if gpu["nvidia_smi"]["devices"]:
        for d in gpu["nvidia_smi"]["devices"]:
            lines.append(f"  - nvidia-smi: {d['name']} ({d['memory_total_mb']} MB), driver {d['driver']}\n")
    lines.append("## Packages\n")
    for k in sorted(pkgs.keys()):
        lines.append(f"- {k}: `{pkgs[k]}`\n")
    if freeze_hash:
        lines.append(f"\n`pip freeze` SHA-256: `{freeze_hash}`\n")
    if warnings:
        lines.append("\n## Warnings\n")
        for w in warnings:
            lines.append(f"- {w}\n")
    p.write_text("".join(lines))

# ------------------------- MAIN -------------------------------------------- #

def main() -> int:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Collect system facts first
    system = collect_system_facts()

    # Versions of common packages (and any extras requested)
    packages = collect_pkg_versions(args.extra_packages)

    # GPU inventory
    gpu = collect_gpu_info()

    # Set determinism (also updates env vars in report)
    determinism_warnings = set_global_determinism(args.seed, system)

    # Reproducibility smoke test
    smoke = reproducibility_smoketest(args.seed)

    # pip freeze snapshot (optional)
    freeze_payload: Dict[str, Any] = {}
    if args.freeze:
        freeze_txt, freeze_hash = get_pip_freeze()
        (outdir / "pip-freeze.txt").write_text(freeze_txt)
        freeze_payload = {"sha256": freeze_hash, "file": "pip-freeze.txt"}

    # Aggregate report
    report: Dict[str, Any] = {
        "system": system,
        "packages": packages,
        "gpu": gpu,
        "seed": args.seed,
        "smoketest": smoke,
        "pip_freeze": freeze_payload,
        "expect_gpu": args.expect_gpu,
    }

    # Compatibility warnings
    warns = determinism_warnings + compatibility_warnings(system, packages, gpu, args.expect_gpu)
    report["warnings"] = warns

    # Write artifacts
    write_json(outdir / "environment_report.json", report)
    write_markdown(outdir / "ENVIRONMENT.md", report, warns)

    # Console summary
    print(textwrap.dedent(f"""
     Environment recorded to: {outdir}
       - environment_report.json
       - ENVIRONMENT.md
       - pip-freeze.txt ({'written' if args.freeze else 'skipped'})
       - seed: {args.seed}
       - GPU expected: {args.expect_gpu}
       - warnings: {len(warns)}
    """).strip())

    if warns and args.fail_on_warnings:
        print("Exiting with non-zero status due to warnings (see ENVIRONMENT.md).", file=sys.stderr)
        return 2
    return 0

if __name__ == "__main__":
    sys.exit(main())


# In[ ]:




