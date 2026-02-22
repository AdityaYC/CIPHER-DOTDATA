"""
Export Depth Anything V2 (ViT-S) via Qualcomm AI Hub for NPU inference.

Requirements:
    pip install qai-hub "qai-hub-models[depth_anything_v2_small_hf]"

    You also need a Qualcomm AI Hub account + API token:
        qai-hub configure --api_token <YOUR_TOKEN>
        (get token at https://aihub.qualcomm.com/  → Account → API Token)

Output:
    CIPHER/models/depth_anything_v2_vits.onnx   ← load with onnxruntime-qnn
"""

import sys
from pathlib import Path

_HERE   = Path(__file__).resolve().parent
_MODELS = _HERE.parent.parent / "models"
_OUT    = _MODELS / "depth_anything_v2_vits.onnx"
_MODELS.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("Depth Anything V2 — Qualcomm AI Hub ONNX Export")
print("=" * 70)
print(f"\nTarget: {_OUT}\n")

if _OUT.exists():
    print(f"Already exists: {_OUT}")
    print("Delete and re-run to re-export.")
    sys.exit(0)

# ── 1. Load the model via qai_hub_models ─────────────────────────────────────
try:
    from qai_hub_models.models.depth_anything_v2 import Model
except ImportError:
    print("ERROR: qai_hub_models not installed.")
    print('Run:  pip install "qai-hub-models[depth-anything-v2]"')
    sys.exit(1)

print("Loading Depth Anything V2 Small from Qualcomm AI Hub Models …")
try:
    model = Model.from_pretrained()
    model.eval()
except Exception as e:
    print(f"ERROR loading model: {e}")
    sys.exit(1)

# ── 2. Compile & export via qai_hub (cloud compile → local ONNX) ─────────────
try:
    import qai_hub
    import torch

    print("Submitting compile job to Qualcomm AI Hub (requires API token) …")

    dummy = torch.zeros(1, 3, 518, 518)
    traced = torch.jit.trace(model, dummy)

    compile_job = qai_hub.submit_compile_job(
        model=traced,
        device=qai_hub.Device("Snapdragon X Elite CRD"),
        input_specs={"image": ((1, 3, 518, 518), "float32")},
        options="--target_runtime onnx",
    )
    print(f"Compile job: {compile_job.job_id}  (waiting for cloud compile …)")
    assert compile_job.wait() == qai_hub.JobStatus.PASSED, "Compile job failed"

    print("Downloading compiled ONNX …")
    compile_job.download_target_model(str(_OUT))
    print(f"\n✅  Saved NPU-optimised ONNX: {_OUT}")

except ImportError:
    # qai_hub not installed — fall back to local torch.onnx.export
    # (standard ONNX, still runs on NPU via onnxruntime-qnn at load time)
    print("qai_hub not installed — exporting standard ONNX locally.")
    print("(onnxruntime-qnn will compile to Hexagon NPU at runtime)\n")
    import torch

    dummy = torch.zeros(1, 3, 518, 518)
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            str(_OUT),
            opset_version=17,
            input_names=["image"],
            output_names=["depth"],
            dynamic_axes={"image": {0: "batch"}, "depth": {0: "batch"}},
        )
    print(f"\n✅  Saved standard ONNX: {_OUT}")

except Exception as e:
    print(f"ERROR during export: {e}")
    if _OUT.exists():
        _OUT.unlink()
    sys.exit(1)

size_mb = _OUT.stat().st_size / 1_048_576
print(f"Model size: {size_mb:.1f} MB")
print("\nNext: restart the backend.")
print("Watch for:  '  Depth: Depth Anything V2 (QNN NPU)'")
