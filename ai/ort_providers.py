"""
Build ONNX Runtime execution provider lists for NPU + GPU acceleration.

- NPU (QNN) first when available — HTP backend so Task Manager shows NPU Compute (not just Shared Memory).
- GPU (CUDA / DirectML) next when not preferring NPU.
- CPU as fallback.

Use PREFER_NPU_OVER_GPU so YOLO/Depth run on NPU. Requires: pip install onnxruntime-qnn
"""

from pathlib import Path
from typing import Any, Dict, List

# QNN provider options to force HTP (NPU) compute so Task Manager shows "NPU Compute" not just "Shared Memory"
# backend_type 'htp' = offload to NPU; htp_performance_mode = drive NPU; enable_htp_shared_memory_allocator '0' = use NPU compute path
QNN_HTP_OPTIONS: Dict[str, str] = {
    "backend_type": "htp",  # HTP = Hexagon Tensor Processor (NPU)
    "htp_performance_mode": "high_performance",  # Use NPU actively so NPU Compute shows in Task Manager
    "qnn_context_priority": "high",  # Prefer NPU context
    "enable_htp_shared_memory_allocator": "0",  # 0 = default, use NPU compute; 1 = shared memory allocator
}


def get_available_providers():
    """Return set of available ONNX Runtime execution provider names."""
    try:
        import onnxruntime as ort
        return set(ort.get_available_providers())
    except Exception:
        return set()


def resolve_qnn_backend_path(config_path: str | None) -> str | None:
    """
    Resolve path to QnnHtp.dll for NPU. On Qualcomm laptops we need this for NPU usage.
    Tries: config path, Qualcomm AIStack, onnxruntime-qnn package (capi/), then rglob.
    """
    if config_path and Path(config_path).exists():
        return config_path
    try:
        import os
        # 1) Qualcomm AIStack — standard on Snapdragon X Elite / Qualcomm dev machines
        for qairt in (
            os.environ.get("QAIRT_PATH"),
            r"C:\Qualcomm\AIStack\QAIRT",
            os.path.expandvars(r"%ProgramFiles%\Qualcomm\AIStack\QAIRT"),
        ):
            if not qairt:
                continue
            base = Path(qairt)
            if not base.exists():
                continue
            # Try versioned subdirs (e.g. 2.22.0) then lib
            for libdir in ("lib/arm64x-windows-msvc", "lib/x86_64-windows-msvc", "lib"):
                p = base / libdir / "QnnHtp.dll"
                if p.exists():
                    return str(p)
            for sub in base.iterdir():
                if sub.is_dir():
                    for libdir in ("lib/arm64x-windows-msvc", "lib/x86_64-windows-msvc", "lib"):
                        p = sub / libdir / "QnnHtp.dll"
                        if p.exists():
                            return str(p)
        # 2) onnxruntime-qnn package (capi/QnnHtp.dll)
        import onnxruntime as ort
        ort_root = Path(ort.__file__).resolve().parent if getattr(ort, "__file__", None) else None
        if ort_root is None:
            for _p in getattr(ort, "__path__", []) or []:
                ort_root = Path(_p)
                break
        if ort_root and ort_root.exists():
            for subdir in ("capi", "lib", "."):
                for name in ("QnnHtp.dll", "QnnHtp.so", "libQnnHtp.so"):
                    p = (ort_root / subdir / name) if subdir != "." else (ort_root / name)
                    if p.exists():
                        return str(p)
            for dll in ort_root.rglob("QnnHtp.dll"):
                return str(dll)
            for so in ort_root.rglob("libQnnHtp.so"):
                return str(so)
    except Exception:
        pass
    return None


def build_providers(
    available: set,
    qnn_dll_path: str | None,
    use_gpu: bool,
    prefer_npu: bool = True,
) -> List[Any]:
    """
    Build provider list: NPU (QNN) and/or GPU (CUDA, DirectML), then CPU.
    When prefer_npu and QNN path exists: [QNN, CPU] so NPU is used (no GPU).
    Otherwise: [QNN?, CUDA?, DML?, CPU].
    """
    resolved_qnn = resolve_qnn_backend_path(qnn_dll_path)
    qnn = []
    if "QNNExecutionProvider" in available:
        qnn.append(("QNNExecutionProvider", _qnn_provider_options(resolved_qnn)))
    gpu = []
    if use_gpu and not prefer_npu:
        if "CUDAExecutionProvider" in available:
            gpu.append("CUDAExecutionProvider")
        if "DmlExecutionProvider" in available:
            gpu.append("DmlExecutionProvider")
    # Prefer NPU: use only [QNN, CPU] so inference runs on NPU (Task Manager shows NPU)
    if prefer_npu and qnn:
        return qnn + ["CPUExecutionProvider"]
    if prefer_npu:
        return gpu + ["CPUExecutionProvider"]
    return qnn + gpu + ["CPUExecutionProvider"]


def _qnn_provider_options(backend_path: str | None) -> Dict[str, str]:
    """Build QNN EP options. Prefer minimal options for max compatibility on Qualcomm NPU."""
    if backend_path:
        # Explicit path: use it; avoid conflicting options (backend_path vs backend_type)
        return {"backend_path": backend_path, "htp_performance_mode": "high_performance"}
    return dict(QNN_HTP_OPTIONS)


def yolo_providers(
    available: set,
    qnn_dll_path: str | None,
    use_gpu: bool,
    split_npu_gpu: bool,
    prefer_npu_over_gpu: bool = True,
) -> List[Any]:
    """
    Providers for YOLO. When prefer_npu_over_gpu: try NPU first (by DLL path), then CPU.
    On Qualcomm laptops we ALWAYS try QNN when QnnHtp.dll is found — do not gate on get_available_providers().
    """
    if prefer_npu_over_gpu:
        resolved = resolve_qnn_backend_path(qnn_dll_path)
        if resolved:
            # Force NPU: we have the DLL, use it. Session creation will fall back to CPU if it fails.
            opts = _qnn_provider_options(resolved)
            return [("QNNExecutionProvider", opts), "CPUExecutionProvider"]
        if "QNNExecutionProvider" in available:
            opts = _qnn_provider_options(resolved)
            return [("QNNExecutionProvider", opts), "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]
    if split_npu_gpu and use_gpu:
        resolved = resolve_qnn_backend_path(qnn_dll_path)
        prov = []
        if resolved or "QNNExecutionProvider" in available:
            prov.append(("QNNExecutionProvider", _qnn_provider_options(resolved)))
        prov.append("CPUExecutionProvider")
        return prov
    return build_providers(available, qnn_dll_path, use_gpu, prefer_npu=True)


def depth_providers(
    available: set,
    qnn_dll_path: str | None,
    use_gpu: bool,
    split_npu_gpu: bool,
    prefer_npu_over_gpu: bool = True,
) -> List[Any]:
    """
    Providers for Depth. When prefer_npu_over_gpu: try NPU first (by DLL path), then CPU.
    On Qualcomm laptops we ALWAYS try QNN when QnnHtp.dll is found.
    """
    if prefer_npu_over_gpu:
        resolved = resolve_qnn_backend_path(qnn_dll_path)
        if resolved:
            opts = _qnn_provider_options(resolved)
            return [("QNNExecutionProvider", opts), "CPUExecutionProvider"]
        if "QNNExecutionProvider" in available:
            resolved = resolve_qnn_backend_path(qnn_dll_path)
            opts = _qnn_provider_options(resolved)
            return [("QNNExecutionProvider", opts), "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]
    if split_npu_gpu and use_gpu:
        prov = []
        if "CUDAExecutionProvider" in available:
            prov.append("CUDAExecutionProvider")
        if "DmlExecutionProvider" in available:
            prov.append("DmlExecutionProvider")
        resolved = resolve_qnn_backend_path(qnn_dll_path)
        if "QNNExecutionProvider" in available:
            prov.append(("QNNExecutionProvider", _qnn_provider_options(resolved)))
        prov.append("CPUExecutionProvider")
        return prov
    return build_providers(available, qnn_dll_path, use_gpu, prefer_npu=True)
