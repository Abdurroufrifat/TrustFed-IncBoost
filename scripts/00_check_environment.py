from __future__ import annotations

import importlib
import platform
import shutil
import subprocess
import sys
from pathlib import Path


REQUIRED = {
    "numpy": "numpy",
    "pandas": "pandas",
    "sklearn": "scikit-learn",
    "xgboost": "xgboost",
    "yaml": "PyYAML",
    "joblib": "joblib",
    "psutil": "psutil",
}


def main() -> int:
    print("TrustFed-IncBoost environment check")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Executable: {sys.executable}")
    print(f"Platform: {platform.platform()}")
    print(f"Working directory: {Path.cwd()}")
    free_gb = shutil.disk_usage(Path.cwd()).free / (1024**3)
    print(f"Free disk: {free_gb:.1f} GB")
    missing = []
    for module_name, package_name in REQUIRED.items():
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, "__version__", "installed")
            print(f"[OK] {package_name}: {version}")
        except Exception as exc:
            missing.append(package_name)
            print(f"[MISSING] {package_name}: {type(exc).__name__}")
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            stderr=subprocess.STDOUT,
            text=True,
            timeout=10,
        ).strip()
        print(f"GPU: {output}")
    except Exception:
        print("GPU: NVIDIA GPU not detected (CPU execution is supported).")
    if sys.version_info[:2] != (3, 12):
        print("[FAIL] Use Python 3.12 for this project.")
        return 1
    if free_gb < 10:
        print("[FAIL] At least 10 GB free disk is required for the first stage.")
        return 1
    if missing:
        print(f"[FAIL] Install missing packages: {', '.join(missing)}")
        return 1
    print("status: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

