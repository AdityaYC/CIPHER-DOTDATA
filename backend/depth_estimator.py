"""
Depth Anything V2 — monocular depth for tactical map.

HuggingFace path uses the METRIC variant which outputs real metres directly.
No hardcoded 0.2–20 m mapping — the model is calibrated for physical distance.

  Indoor metric model: 0–20 m range (good for room-scale demos)
  Outdoor metric model: 0–80 m range (better for drone/outdoor use)

Qualcomm ONNX path (relative model) still maps 0–1 → 0.2–20 m as a fallback.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Metric model — outputs depth in metres directly (no fake scale mapping)
MODEL_NAME = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf"
# Hard clamp for sanity (metric model is calibrated but can still produce outliers)
METRIC_MIN_M = 0.1
METRIC_MAX_M = 50.0

# Qualcomm ONNX fallback (relative model — still needs scale mapping)
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
QUALCOMM_ONNX_PATH = _REPO_ROOT / "models" / "depth_anything_v2_indoor_small_onnx_mnl2019oq.onnx"
ONNX_INPUT_SIZE = 518
ONNX_DEPTH_MIN_M = 0.2
ONNX_DEPTH_MAX_M = 20.0
# ImageNet normalization
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class DepthEstimator:
    """Depth Anything V2: tries Qualcomm AI Hub ONNX first, then HuggingFace metric."""

    def __init__(self):
        self._backend = None   # "qualcomm_onnx" | "huggingface_metric"
        self._session  = None  # onnxruntime session (ONNX path)
        self._model    = None
        self._processor = None
        self._device   = "cpu"
        self._load()

    def _load(self) -> None:
        if QUALCOMM_ONNX_PATH.is_file():
            if self._load_qualcomm_onnx():
                return
        self._load_huggingface()

    # ------------------------------------------------------------------
    def _load_qualcomm_onnx(self) -> bool:
        try:
            import onnxruntime as ort
        except ImportError:
            logger.debug("onnxruntime not installed — skip Qualcomm depth ONNX")
            return False
        try:
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            # NPU + GPU: prefer NPU so Task Manager shows NPU utilisation (needs onnxruntime-qnn)
            try:
                from backend.ort_providers import get_available_providers, depth_providers
                from backend import config
                qnn_path = getattr(config, "QNN_DLL_PATH", None)
                available = get_available_providers()
                use_gpu = getattr(config, "USE_GPU", True)
                split_npu_gpu = getattr(config, "SPLIT_NPU_GPU", False)
                prefer_npu = getattr(config, "PREFER_NPU_OVER_GPU", True)
                prov_list = depth_providers(
                    available, qnn_path, use_gpu, split_npu_gpu, prefer_npu
                )
            except Exception:
                qnn_dll = Path(ort.__file__).parent / "capi" / "QnnHtp.dll"
                available = set(ort.get_available_providers())
                prov_list = []
                if "QNNExecutionProvider" in available and qnn_dll.exists():
                    prov_list.append(("QNNExecutionProvider", {"backend_path": str(qnn_dll)}))
                prov_list.append("CPUExecutionProvider")
            self._session = ort.InferenceSession(str(QUALCOMM_ONNX_PATH), opts, providers=prov_list)
            self._backend = "qualcomm_onnx"
            active = self._session.get_providers()
            primary = active[0] if active else "?"
            if "QNNExecutionProvider" in active:
                print(f"  Depth: Depth Anything V2 ONNX on NPU (QNN)")
            else:
                print(f"  Depth: Depth Anything V2 ONNX on {primary}")
            return True
        except Exception as e:
            logger.warning(f"  Depth: Qualcomm ONNX load failed — {e}")
            return False

    def _load_huggingface(self) -> None:
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
            print(f"  Depth: loading {MODEL_NAME} …")
            self._processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
            self._model = AutoModelForDepthEstimation.from_pretrained(MODEL_NAME)
            self._model.eval()
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model = self._model.to(self._device)
            self._backend = "huggingface_metric"
            print(f"  Depth: Depth Anything V2 Metric ready on {self._device.upper()} — outputs real metres")
        except Exception as e:
            logger.warning(f"  Depth: failed to load — {e}")
            self._model = None

    # ------------------------------------------------------------------
    @property
    def loaded(self) -> bool:
        return self._session is not None or self._model is not None

    @property
    def provider(self) -> str:
        if self._backend == "qualcomm_onnx":
            return "Qualcomm AI Hub"
        if self._model is not None:
            return self._device.upper()
        return "not loaded"

    # ------------------------------------------------------------------
    def infer(self, bgr_frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Run depth estimation on a BGR frame.
        Returns float32 (H, W) depth map in METRES.
        HuggingFace metric path: physically calibrated metres.
        ONNX path: relative depth mapped to 0.2–20 m.
        """
        if self._backend == "qualcomm_onnx" and self._session is not None:
            return self._infer_onnx(bgr_frame)
        if self._model is not None:
            return self._infer_huggingface(bgr_frame)
        return None

    def _infer_onnx(self, bgr_frame: np.ndarray) -> Optional[np.ndarray]:
        """Qualcomm AI Hub ONNX — outputs metric depth in metres, clip to valid indoor range."""
        try:
            import cv2
            h_orig, w_orig = bgr_frame.shape[:2]
            rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (ONNX_INPUT_SIZE, ONNX_INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
            x = (resized.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
            x = np.expand_dims(x.transpose(2, 0, 1), 0).astype(np.float32)
            out = self._session.run(None, {self._session.get_inputs()[0].name: x})[0]
            depth = out.squeeze().astype(np.float32)
            if depth.shape[0] != h_orig or depth.shape[1] != w_orig:
                depth = cv2.resize(depth, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
            # Model outputs disparity (higher = closer) — empirical calibration: k=0.643
            depth = np.clip(0.643 / np.maximum(depth, 0.01), ONNX_DEPTH_MIN_M, ONNX_DEPTH_MAX_M)
            return depth
        except Exception as e:
            logger.debug(f"Depth ONNX inference error: {e}")
            return None

    def _infer_huggingface(self, bgr_frame: np.ndarray) -> Optional[np.ndarray]:
        """Metric model — predicted_depth is already in metres."""
        try:
            import cv2
            import torch
            from PIL import Image
            h_orig, w_orig = bgr_frame.shape[:2]
            pil_img = Image.fromarray(cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB))
            inputs = {k: v.to(self._device) for k, v in self._processor(images=pil_img, return_tensors="pt").items()}
            with torch.no_grad():
                outputs = self._model(**inputs)
            # predicted_depth: (1, H', W') in metres — upsample to original size
            depth = torch.nn.functional.interpolate(
                outputs.predicted_depth.unsqueeze(1),
                size=(h_orig, w_orig),
                mode="bicubic",
                align_corners=False,
            ).squeeze().cpu().numpy().astype(np.float32)
            # Clamp to sane range
            depth = np.clip(depth, METRIC_MIN_M, METRIC_MAX_M)
            return depth
        except Exception as e:
            logger.debug(f"Depth metric inference error: {e}")
            return None

    # ------------------------------------------------------------------
    def depth_at_bbox(self, depth_map: np.ndarray, bbox, frame_w: int, frame_h: int) -> float:
        """
        Return median depth in metres inside a YOLO bounding box.
        depth_map is already in metres (from infer()).
        """
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame_w - 1, x2), min(frame_h - 1, y2)
        if x2 <= x1 or y2 <= y1:
            return METRIC_MAX_M

        # Sample inner 50% of bbox to avoid boundary noise
        pad_x = (x2 - x1) // 4
        pad_y = (y2 - y1) // 4
        region = depth_map[y1 + pad_y : y2 - pad_y, x1 + pad_x : x2 - pad_x]
        if region.size == 0:
            region = depth_map[y1:y2, x1:x2]
        if region.size == 0:
            return METRIC_MAX_M

        # Median is more robust than mean for depth (avoids outlier pixels)
        dist_m = float(np.median(region))
        return round(np.clip(dist_m, METRIC_MIN_M, METRIC_MAX_M), 2)
