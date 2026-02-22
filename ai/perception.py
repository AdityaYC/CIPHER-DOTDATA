"""
PHANTOM CODE — YOLOv8 inference on NPU via ONNX Runtime.
"""

import numpy as np
import cv2
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# COCO 80 class names (standard order)
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet",
    "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]


class YOLODetector:
    """Run YOLOv8 object detection via ONNX Runtime (NPU + GPU + CPU)."""

    def __init__(
        self,
        model_path: str,
        qnn_dll_path: str | None = None,
        confidence_threshold: float = 0.45,
        input_size: int = 640,
        use_gpu: bool = True,
        split_npu_gpu: bool = True,
        prefer_npu_over_gpu: bool = True,
    ):
        import onnxruntime as ort
        from ai.ort_providers import get_available_providers, yolo_providers

        self.model_path = model_path
        self._last_latency_ms: float = 0.0
        self._input_size = int(input_size)
        self._conf_threshold = confidence_threshold

        # Resolve model path (may be relative to project root)
        path = Path(model_path)
        if not path.is_absolute():
            for base in [Path.cwd(), Path.cwd().parent]:
                candidate = base / model_path
                if candidate.exists():
                    model_path = str(candidate)
                    break

        # Session options: enable graph optimizations for faster inference
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_opts.intra_op_num_threads = 2  # avoid CPU oversubscription when NPU/GPU used

        available = get_available_providers()
        prov_list = yolo_providers(
            available, qnn_dll_path, use_gpu, split_npu_gpu, prefer_npu_over_gpu
        )
        if "QNNExecutionProvider" in str(prov_list):
            from ai.ort_providers import resolve_qnn_backend_path
            _path = resolve_qnn_backend_path(qnn_dll_path)
            logger.info("YOLO: attempting NPU (QNN) backend_path=%s", _path or "(default)")
        # Retry NPU session creation — Qualcomm NPU can need a moment to init
        last_err = None
        for attempt in range(4):
            try:
                self._session = ort.InferenceSession(
                    model_path, sess_options=sess_opts, providers=prov_list
                )
                last_err = None
                break
            except Exception as e:
                last_err = e
                if attempt < 3 and "QNNExecutionProvider" in str(prov_list):
                    import time
                    time.sleep(1.0 * (attempt + 1))  # give NPU time to init
                    continue
                logger.warning(
                    "NPU/GPU failed (attempt %d): %s — using CPU. Check QnnHtp.dll path in config.",
                    attempt + 1, e
                )
                self._session = ort.InferenceSession(
                    model_path, sess_options=sess_opts, providers=["CPUExecutionProvider"]
                )
                break

        active = self._session.get_providers()
        if "QNNExecutionProvider" in active:
            logger.info(f"YOLO: NPU (QNN) — providers {active}")
        elif "CUDAExecutionProvider" in active or "DmlExecutionProvider" in active:
            logger.info(f"YOLO: GPU — providers {active}")
        else:
            logger.warning(
                "\033[93m YOLO on CPU. For NPU: pip install onnxruntime-qnn (Windows Snapdragon).\033[0m"
            )
        self._input_name = self._session.get_inputs()[0].name

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Resize frame to 640x640, BGR->RGB, normalize, NCHW."""
        img = cv2.resize(frame, (self._input_size, self._input_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        # NCHW
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0)
        return img

    def detect(self, frame: np.ndarray) -> list[dict]:
        """
        Run inference and return list of detections.
        Each: {"class": str, "confidence": float, "bbox": [x1,y1,x2,y2], "center": [cx,cy]}
        bbox and center in original frame pixel coordinates.
        """
        import time
        h_orig, w_orig = frame.shape[:2]
        inp = self.preprocess(frame)
        start = time.perf_counter()
        # Retry once on failure (NPU/HTP can glitch or restart)
        try:
            out = self._session.run(None, {self._input_name: inp})
        except Exception as e1:
            try:
                out = self._session.run(None, {self._input_name: inp})
            except Exception:
                raise e1
        self._last_latency_ms = (time.perf_counter() - start) * 1000

        # YOLOv8 ONNX output: (1, 84, 8400) or (1, 8400, 84) depending on export
        raw = out[0]
        if raw.ndim == 3:
            raw = raw[0]
        # Ensure shape (8400, 84): 8400 boxes, 84 = 4 (cx,cy,w,h) + 80 classes
        if raw.shape[0] == 84 and raw.shape[1] == 8400:
            raw = np.transpose(raw, (1, 0))  # -> (8400, 84)
        elif raw.shape[0] == 8400 and raw.shape[1] == 84:
            pass  # already (8400, 84)
        else:
            raw = np.transpose(raw, (1, 0))  # try transpose for other layouts
        scale_x = w_orig / self._input_size
        scale_y = h_orig / self._input_size

        detections = []
        for i in range(raw.shape[0]):
            cx, cy, w, h = raw[i, :4]
            probs = raw[i, 4:84]
            class_id = int(np.argmax(probs))
            confidence = float(probs[class_id])
            if confidence < self._conf_threshold:
                continue
            # scale to original frame
            cx_s = cx * scale_x
            cy_s = cy * scale_y
            w_s = w * scale_x
            h_s = h * scale_y
            x1 = cx_s - w_s / 2
            y1 = cy_s - h_s / 2
            x2 = cx_s + w_s / 2
            y2 = cy_s + h_s / 2
            detections.append({
                "class": COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else "object",
                "confidence": confidence,
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "center": [float(cx_s), float(cy_s)],
            })

        # NMS
        detections = self._nms(detections, iou_threshold=0.5)
        return detections

    def _nms(self, detections: list[dict], iou_threshold: float = 0.45) -> list[dict]:
        """Non-max suppression by bbox IoU."""
        if not detections:
            return []
        boxes = np.array([d["bbox"] for d in detections])
        scores = np.array([d["confidence"] for d in detections])
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        area = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            if order.size == 1:
                break
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
            iou = inter / (area[i] + area[order[1:]] - inter)
            order = order[1:][iou <= iou_threshold]
        return [detections[j] for j in keep]

    def get_provider(self) -> str:
        """Return the primary execution provider in use."""
        return self._session.get_providers()[0]

    def get_last_latency(self) -> float:
        """Return last inference time in milliseconds."""
        return self._last_latency_ms
