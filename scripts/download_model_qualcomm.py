"""
Download YOLOv8 detection from Qualcomm AI Hub and save as models/yolov8_det.onnx.

Uses Qualcomm AI Hub Models (qai_hub_models) to download the v8 detection model
and export to ONNX for use with the Drone/PHANTOM backend (NPU or CPU).

Install first (if needed):
    pip install "qai_hub_models[yolov8]"

Optional: sign in at https://app.aihub.qualcomm.com/ and set API token for cloud compile:
    qai-hub configure --api_token YOUR_TOKEN

If Qualcomm AI Hub is not available, falls back to Ultralytics YOLOv8n export.
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_ONNX = MODELS_DIR / "yolov8_det.onnx"

MODELS_DIR.mkdir(parents=True, exist_ok=True)


def run_qualcomm_export() -> bool:
    """Run Qualcomm AI Hub YOLOv8 detection export to ONNX. Returns True if ONNX was produced."""
    try:
        from qai_hub_models.models.yolov8_det import Model as YOLOv8Model
        from qai_hub_models import TargetRuntime
    except ImportError:
        print("Installing qai_hub_models with YOLOv8 support...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "qai-hub-models", "qai-hub-models[yolov8]", "-q"
        ], cwd=str(PROJECT_ROOT))
        from qai_hub_models.models.yolov8_det import Model as YOLOv8Model
        from qai_hub_models import TargetRuntime

    out_dir = MODELS_DIR / "qualcomm_export"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading YOLOv8 detection from Qualcomm AI Hub...")
    print("(First run may download ~12MB from Hugging Face / Qualcomm)")

    # Run the official export module: ONNX runtime, skip cloud profiling/inference for speed
    cmd = [
        sys.executable, "-m", "qai_hub_models.models.yolov8_det.export",
        "--target-runtime", "onnx",
        "--skip-profiling",
        "--skip-inferencing",
        "--skip-summary",
        "--output-dir", str(out_dir),
    ]
    try:
        subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True, timeout=300)
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
        print(f"Qualcomm export failed: {e}")
        return False

    # Find the exported ONNX file (may be in out_dir or PROJECT_ROOT/export_assets)
    onnx_files = list(out_dir.rglob("*.onnx"))
    if not onnx_files:
        export_assets = PROJECT_ROOT / "export_assets"
        if export_assets.exists():
            onnx_files = list(export_assets.rglob("*.onnx"))
    if not onnx_files:
        onnx_files = list(PROJECT_ROOT.rglob("*.onnx"))
        onnx_files = [p for p in onnx_files if "yolov8" in p.name.lower() or "det" in p.name.lower()]
    if onnx_files:
        src = onnx_files[0]
        shutil.copy2(src, OUTPUT_ONNX)
        print(f"Copied {src.name} -> {OUTPUT_ONNX}")
        return True
    return False


def run_ultralytics_fallback() -> bool:
    """Fallback: Ultralytics YOLOv8n export (CPU-friendly ONNX)."""
    print("Falling back to Ultralytics YOLOv8n export...")
    script = SCRIPT_DIR / "download_model.py"
    if not script.exists():
        print("download_model.py not found.")
        return False
    r = subprocess.run([sys.executable, str(script)], cwd=str(PROJECT_ROOT), timeout=120)
    return r.returncode == 0 and OUTPUT_ONNX.is_file()


def main():
    print("YOLOv8 detection — Qualcomm AI Hub download")
    print(f"Output: {OUTPUT_ONNX}\n")

    if run_qualcomm_export():
        print(f"\nDone. Model saved to {OUTPUT_ONNX}")
        return 0

    if run_ultralytics_fallback():
        print(f"\nDone (Ultralytics fallback). Model saved to {OUTPUT_ONNX}")
        return 0

    print("\nAll methods failed. Install: pip install 'qai-hub-models[yolov8]' or pip install ultralytics")
    return 1


if __name__ == "__main__":
    sys.exit(main())
