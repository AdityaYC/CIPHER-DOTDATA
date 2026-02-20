"""
Download YOLOv8n and export to ONNX for PHANTOM CODE.
Saves to phantom_code/models/yolov8_det.onnx (CPU-friendly; replace with Qualcomm AI Hub export for NPU).
"""

import os
import sys

# Run from phantom_code so paths are correct
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
OUTPUT_PATH = os.path.join(MODELS_DIR, "yolov8_det.onnx")

os.makedirs(MODELS_DIR, exist_ok=True)


def main():
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Installing ultralytics...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics", "-q"])
        from ultralytics import YOLO

    print("Loading YOLOv8n (downloads ~6MB if needed)...")
    model = YOLO("yolov8n.pt")
    print("Exporting to ONNX (640x640, CPU)...")
    os.chdir(PROJECT_ROOT)
    exported_path = model.export(format="onnx", imgsz=640, dynamic=False, half=False)
    import shutil
    if isinstance(exported_path, str) and os.path.isfile(exported_path):
        if os.path.abspath(exported_path) != os.path.abspath(OUTPUT_PATH):
            shutil.move(exported_path, OUTPUT_PATH)
    else:
        # export may write yolov8n.onnx in cwd
        default_name = os.path.join(PROJECT_ROOT, "yolov8n.onnx")
        if os.path.isfile(default_name):
            shutil.move(default_name, OUTPUT_PATH)
        else:
            raise FileNotFoundError("Export succeeded but could not find exported ONNX file.")
    print(f"Model saved to {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
