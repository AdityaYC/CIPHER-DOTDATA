"""
Download and cache the Depth Anything V2 model (HuggingFace).
Used for optional depth on the tactical map / minimap.
Cache goes to ~/.cache/huggingface/hub/. Run from repo root.
"""

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Small variant for lower RAM; use Depth-Anything-V2-Base-hf or Depth-Anything-V2-Large-hf for better quality
DEPTH_MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"


def main():
    print("Downloading Depth Anything V2 (Small) from HuggingFace...")
    print("  Model:", DEPTH_MODEL_ID)
    print("  Cache: ~/.cache/huggingface/hub/")
    try:
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    except ImportError:
        print("Installing transformers...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers", "torch", "accelerate", "-q"])
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    try:
        AutoImageProcessor.from_pretrained(DEPTH_MODEL_ID)
        print("  Image processor cached.")
        AutoModelForDepthEstimation.from_pretrained(DEPTH_MODEL_ID)
        print("  Depth model cached.")
    except Exception as e:
        print(f"  Error: {e}")
        return 1
    print("Done. Depth model is ready for optional use (e.g. minimap depth).")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
