"""Local FastAPI backend - replaces Modal.

Serves: /getImage, /stream_agents (Drone), plus Drone2 features when run from repo root:
  /api/status, /api/detections, /api/advisory, /api/mission, /api/feed, /live_detections (laptop camera).

Run from Drone2 repo root (so Drone2 backend is on PYTHONPATH):
    py -m uvicorn Drone.local_backend.app:app --host 0.0.0.0 --port 8000
Or from this folder:
    pip install -r requirements.txt
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload
"""

import asyncio
import base64
import bisect
import collections as _collections
import csv
import json
import math
import os
import sys
import time
import threading as _threading
import uuid
from pathlib import Path
from typing import Dict, List, Optional
from io import BytesIO

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from PIL import Image

# Repo root (parent of backend/)
_HERE = Path(__file__).resolve().parent
_DRONE2_ROOT = _HERE.parent
if str(_DRONE2_ROOT) not in sys.path:
    sys.path.insert(0, str(_DRONE2_ROOT))
# So "from world_graph import WorldGraph" works when cwd is repo root
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# Optional Drone2 backend (camera, YOLO ONNX, advisory)
_phantom = None
try:
    import cv2
    import numpy as np
    from backend import config as phantom_config
    # So backend modules that "import config" resolve to backend.config
    sys.modules["config"] = phantom_config
    from ai.camera_manager import CameraManager as PhantomCameraManager
    from ai.perception import YOLODetector as PhantomYOLODetector
    from backend import detection_mapper as phantom_detection_mapper
    from ai.llm_advisory import get_advisory as phantom_get_advisory
    _phantom = {
        "config": phantom_config,
        "cv2": cv2,
        "np": np,
        "CameraManager": PhantomCameraManager,
        "YOLODetector": PhantomYOLODetector,
        "detection_mapper": phantom_detection_mapper,
        "get_advisory": phantom_get_advisory,
    }
except Exception as e:
    print(f"Drone2 backend not loaded (run from repo root with backend on path): {e}")

app = FastAPI(title="Cipher — tactical AI backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_PROJECT = _HERE.parent
_DATA = _PROJECT / "data"
_DRONE_FRONTEND_DIST = _PROJECT / "frontend" / "dist"
_EXPORTS_DIR = _DRONE2_ROOT / "exports"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FRAME_INTERVAL = 0.1
STEP_SIZE = 0.1
ALLOWED_THRESHOLD = 0.15
MAX_STEPS = 15

DIRECTION_OFFSETS = {
    0:   {"forward": (1, 0),  "backward": (-1, 0),  "left": (0, -1), "right": (0, 1)},
    90:  {"forward": (0, 1),  "backward": (0, -1),  "left": (1, 0),  "right": (-1, 0)},
    180: {"forward": (-1, 0), "backward": (1, 0),   "left": (0, 1),  "right": (0, -1)},
    270: {"forward": (0, -1), "backward": (0, 1),   "left": (-1, 0), "right": (1, 0)},
}

BOUNDS = {
    "x": (-200, 200),
    "y": (-200, 200),
    "z": (-100, 100),
}

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
class ImageDatabase:
    """In-memory database of trajectory frames."""
    
    def __init__(self):
        self.db: List[Dict] = []
        self.wp_ts = []
        self.wp_xs = []
        self.wp_ys = []
        self.wp_zs = []
        self.gyro_ts = []
        self.gyro_yaws = []
        self.gyro_pitches = []
        self.gyro_rolls = []
        
    def load(self):
        """Load waypoints, gyro, and frames."""
        print("Loading trajectory data...")
        self._load_waypoints()
        self._load_gyro()
        self._load_frames()
        print(f"Database ready: {len(self.db)} frames")
        
    def _load_waypoints(self):
        csv_path = _DATA / "timestamp_coordinates.csv"
        if not csv_path.exists():
            print(f"Warning: {csv_path} not found")
            return
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                self.wp_ts.append(float(row["timestamp"]))
                self.wp_xs.append(float(row["x"]))
                self.wp_ys.append(float(row["y"]))
                self.wp_zs.append(float(row["z"]))
    
    def _load_gyro(self):
        csv_path = _DATA / "gyro.csv"
        if not csv_path.exists():
            print(f"Warning: {csv_path} not found")
            return
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                self.gyro_ts.append(float(row["timestamp_s"]))
                self.gyro_yaws.append(float(row["yaw_deg"]))
                self.gyro_pitches.append(float(row["pitch_deg"]))
                self.gyro_rolls.append(float(row["roll_deg"]))
    
    def _load_frames(self):
        frame_dir = _DATA / "image_samples"
        if not frame_dir.exists():
            print(f"Warning: {frame_dir} not found")
            return
        
        frame_files = sorted(frame_dir.glob("frame_*.jpg"))
        max_t = self.wp_ts[-1] if self.wp_ts else 0
        
        for i, fpath in enumerate(frame_files):
            t = i * FRAME_INTERVAL
            if t > max_t:
                break
            x, y, z = self._interpolate_position(t)
            yaw, pitch, roll = self._closest_gyro(t)
            
            self.db.append({
                "t": t,
                "x": x,
                "y": y,
                "z": z,
                "yaw": yaw,
                "pitch": pitch,
                "roll": roll,
                "filename": fpath.name,
                "path": fpath,
            })
    
    def _interpolate_position(self, t: float):
        if not self.wp_ts:
            return 0.0, 0.0, 0.0
        i = bisect.bisect_right(self.wp_ts, t) - 1
        i = max(0, min(i, len(self.wp_ts) - 2))
        t0, t1 = self.wp_ts[i], self.wp_ts[i + 1]
        dt = t1 - t0
        alpha = (t - t0) / dt if dt > 0 else 0.0
        alpha = max(0.0, min(1.0, alpha))
        x = self.wp_xs[i] + alpha * (self.wp_xs[i + 1] - self.wp_xs[i])
        y = self.wp_ys[i] + alpha * (self.wp_ys[i + 1] - self.wp_ys[i])
        z = self.wp_zs[i] + alpha * (self.wp_zs[i + 1] - self.wp_zs[i])
        return x, y, z
    
    def _closest_gyro(self, t: float):
        if not self.gyro_ts:
            return 0.0, 0.0, 0.0
        i = bisect.bisect_right(self.gyro_ts, t) - 1
        i = max(0, min(i, len(self.gyro_ts) - 1))
        if i + 1 < len(self.gyro_ts) and abs(self.gyro_ts[i + 1] - t) < abs(self.gyro_ts[i] - t):
            i += 1
        return self.gyro_yaws[i], self.gyro_pitches[i], self.gyro_rolls[i]
    
    def find_best(self, x: float, y: float, z: float, yaw: float) -> int:
        """Find closest matching frame."""
        best_score, best_idx = float("inf"), 0
        for i, e in enumerate(self.db):
            dx = e["x"] - x
            dy = e["y"] - y
            dz = e["z"] - z
            pos_dist = math.sqrt(dx**2 + dy**2 + dz**2)
            dyaw = abs(self._angle_diff(e["yaw"], yaw))
            score = pos_dist + 0.05 * dyaw
            if score < best_score:
                best_score = score
                best_idx = i
        return best_idx
    
    def check_allowed(self, x: float, y: float, z: float, yaw: float) -> Dict:
        """Check which directions are navigable."""
        yaw_key = int(round(yaw)) % 360
        offsets = DIRECTION_OFFSETS.get(yaw_key, DIRECTION_OFFSETS[0])
        
        allowed = {}
        for direction, (dx, dy) in offsets.items():
            tx = x + dx * STEP_SIZE
            ty = y + dy * STEP_SIZE
            found = False
            for e in self.db:
                dist = math.sqrt((e["x"] - tx)**2 + (e["y"] - ty)**2 + (e["z"] - z)**2)
                if dist < ALLOWED_THRESHOLD:
                    yaw_diff = abs(self._angle_diff(e["yaw"], yaw_key))
                    if yaw_diff < 45:
                        found = True
                        break
            allowed[direction] = found
        
        for turn_name, turn_yaw in [("turnLeft", (yaw_key - 90) % 360),
                                     ("turnRight", (yaw_key + 90) % 360)]:
            found = False
            for e in self.db:
                pos_dist = math.sqrt((e["x"] - x)**2 + (e["y"] - y)**2 + (e["z"] - z)**2)
                if pos_dist < ALLOWED_THRESHOLD:
                    yaw_diff = abs(self._angle_diff(e["yaw"], turn_yaw))
                    if yaw_diff < 45:
                        found = True
                        break
            allowed[turn_name] = found
        
        return allowed
    
    @staticmethod
    def _angle_diff(a: float, b: float) -> float:
        d = (a - b) % 360
        return d - 360 if d > 180 else d


# Global database
image_db = ImageDatabase()

# ---------------------------------------------------------------------------
# Models (lazy loaded)
# ---------------------------------------------------------------------------
class ModelManager:
    def __init__(self):
        self.llama_vision = None
        self.llama_processor = None
        self.yolo = None
    
    def load_llama_vision(self):
        """Load Qwen2.5-VL-3B via mlx-vlm (fast on Apple Silicon MPS, ~2GB)."""
        if self.llama_vision is not None:
            return
        
        print("Loading vision model via mlx-vlm (Apple Silicon optimized)...")
        from mlx_vlm import load
        
        self._vlm_model_id = "mlx-community/Qwen2.5-VL-3B-Instruct-4bit"
        self.llama_vision, self.llama_processor = load(self._vlm_model_id)
        print("✅ Vision model loaded (Qwen2.5-VL-3B 4-bit)")
    
    _yolo_load_failed: bool = False  # class-level so we only print once when import fails

    def load_yolo(self):
        """Load YOLO model (once). If ultralytics missing, skip and don't spam log."""
        if self.yolo is not None:
            return
        if getattr(ModelManager, "_yolo_load_failed", False):
            return
        try:
            print("Loading YOLO model...")
            from ultralytics import YOLO
            self.yolo = YOLO("yolov8n.pt")
            print("YOLO loaded")
        except Exception as e:
            ModelManager._yolo_load_failed = True
            print(f"YOLO not available ({e}). Install: python -m pip install ultralytics")
    
    def infer_llama(self, image: Image.Image, prompt: str) -> str:
        """Run vision inference via mlx-vlm (~1-3s on Apple Silicon)."""
        self.load_llama_vision()
        
        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template
        from mlx_vlm.utils import load_config
        import tempfile, os
        
        # Resize to 336x336 — reduces prefill tokens from ~2700 to ~500, 5x faster
        img_small = image.copy()
        img_small.thumbnail((336, 336), Image.LANCZOS)

        # Save PIL image to temp file (mlx-vlm needs a file path)
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            img_small.save(tmp.name, format="JPEG", quality=85)
            tmp_path = tmp.name
        
        try:
            config = load_config(self._vlm_model_id)
            formatted_prompt = apply_chat_template(
                self.llama_processor, config, prompt, num_images=1
            )
            output = generate(
                self.llama_vision,
                self.llama_processor,
                image=tmp_path,
                prompt=formatted_prompt,
                max_tokens=100,
                verbose=False,
            )
            # GenerationResult object — extract text
            if hasattr(output, "text"):
                return output.text.strip()
            if hasattr(output, "__iter__"):
                return "".join(output).strip()
            return str(output).strip()
        finally:
            os.unlink(tmp_path)
    
    def detect_objects(self, image: Image.Image) -> List[Dict]:
        """Run YOLO object detection (conf=0.15 to get more detections)."""
        self.load_yolo()
        if self.yolo is None:
            return []
        # conf=0.15 so person/objects are detected more readily
        results = self.yolo(image, conf=0.15, verbose=False)
        detections = []
        for r in results:
            for box in r.boxes:
                detections.append({
                    "class": r.names[int(box.cls[0])],
                    "confidence": float(box.conf[0]),
                    "bbox": box.xyxy[0].tolist(),
                })
        return detections


models = ModelManager()

# ---------------------------------------------------------------------------
# API Models
# ---------------------------------------------------------------------------
class AgentRequest(BaseModel):
    query: str
    start_x: float = 0.0
    start_y: float = 0.0
    start_z: float = 0.0
    start_yaw: float = 0.0
    num_agents: int = 2


# ---------------------------------------------------------------------------
# World graph for tactical map (same as Drone2 graph_api)
# ---------------------------------------------------------------------------
_world_graph = None


def _get_world_graph():
    """Return world graph, creating it if needed so import/3D map always have one."""
    global _world_graph
    if _world_graph is not None:
        return _world_graph
    try:
        from world_graph import WorldGraph
        _world_graph = WorldGraph()
        return _world_graph
    except Exception:
        try:
            from .world_graph import WorldGraph
            _world_graph = WorldGraph()
            return _world_graph
        except Exception as e:
            print(f"World graph not loaded: {e}")
            return None
    return None


# Ensure world graph exists at load (so ingest loop etc. can use it)
_world_graph = _get_world_graph()

# ---------------------------------------------------------------------------
# Drone2 tactical state (laptop camera, YOLO, advisory)
# ---------------------------------------------------------------------------
phantom_state = {
    "detections": {},
    "raw_detections": [],
    "feeds": {},
    "processed_frames": {},
    "advisory": {"text": "", "mission": "search_rescue", "timestamp": ""},
    "npu_provider": "CPUExecutionProvider",
    "yolo_latency_ms": 0.0,
    "depth_enabled": False,      # True once depth_anything model loaded
    "depth_latency_ms": 0.0,     # per-frame depth inference time
    "depth_provider": "not loaded",
    "current_mission": "search_rescue",
    "last_llm_time": 0.0,
    "yolo_error": None,  # set if Drone YOLO load or run fails
    "yolo_enabled": False,  # OFF by default — frontend START AI enables it
    "recording": False,     # True while START AI is on — clip is written and auto-imported on STOP AI
    "agent_response": {"answer": "", "node_ids": [], "ts": 0.0},  # tactical query (voice/text) for Agent tab
}

# Combined agent (spatial + knowledge): 3D Map AGENT mode
agent_state = {
    "ready": False,
    "initializing": True,
    "last_result": None,  # OrchestratorResult for UI highlight/path
    "running": False,
}


async def _agent_init_background():
    """Pre-load CLIP, Chroma + manuals, Genie check. Sets agent_state.ready."""
    global agent_state
    try:
        from clip_navigator import load_clip
        load_clip()
    except Exception as e:
        print(f"  Agent (CLIP): skip ({e})")
    try:
        from knowledge_agent import load_vector_db
        wg = _get_world_graph()
        load_vector_db(wg)
    except Exception as e:
        print(f"  Agent (vector DB): skip ({e})")
    try:
        from ai.genie_runner import is_available
        if is_available():
            print("  Agent (Genie): ready")
    except Exception:
        pass
    agent_state["initializing"] = False
    agent_state["ready"] = True


phantom_camera_manager = None
phantom_yolo_detector = None
# Consecutive NPU detect() failures — after N we try to reload YOLO on NPU instead of staying on CPU
_npu_fail_count = 0
# Prefer 320 model when available (faster NPU/CPU); fallback to 640
_phantom_model_path = _DRONE2_ROOT / "models" / "yolov8_det.onnx"
_phantom_model_path_fast = _DRONE2_ROOT / "models" / "yolov8_det_320.onnx"

# Simple laptop camera when Drone2 stack is not loaded (no PYTHONPATH or model missing)
_simple_capture = None
_simple_camera_frame = None  # BGR numpy array, updated by background task
_simple_camera_jpeg = None   # bytes for /api/feed/.../processed
_placeholder_jpeg = None     # "Camera starting..." placeholder to avoid 503 flicker

# Live YOLO frame buffer for Agent gallery
_LIVE_FRAME_MAX = 40
_STORE_INTERVAL_S = 1.0
_live_frame_buffer: "collections.deque" = _collections.deque(maxlen=_LIVE_FRAME_MAX)  # type: ignore[type-arg]
_live_frame_counter = 0

def _store_live_frame(bgr_frame, detections: list) -> None:
    """Store a thumbnail + detections in the live frame buffer (throttled to 1fps)."""
    global _live_frame_counter
    if not detections:
        return
    now = time.time()
    if _live_frame_buffer and now - _live_frame_buffer[-1]["ts"] < _STORE_INTERVAL_S:
        return
    try:
        import cv2 as _cv2
        thumb = _cv2.resize(bgr_frame, (160, 120))
        _, jpeg = _cv2.imencode(".jpg", thumb, [_cv2.IMWRITE_JPEG_QUALITY, 70])
        _live_frame_counter += 1
        _live_frame_buffer.append({
            "id": f"lf{_live_frame_counter}",
            "ts": round(now, 2),
            "image_b64": base64.b64encode(jpeg.tobytes()).decode("ascii"),
            "detections": [
                {"class": d.get("class", ""), "confidence": round(float(d.get("confidence", 0)), 2)}
                for d in detections
            ],
        })
    except Exception:
        pass

# Recording clip for Agent: START AI starts, STOP AI ends and auto-imports
_recording_writer = None
_recording_path: Optional[str] = None

# Depth Anything V2 estimator (loaded at startup if model file exists)
_depth_estimator = None


def _make_placeholder_frame(cv2_module):
    """Return a 640x480 BGR image with 'Camera starting...' text (no file dependency)."""
    try:
        img = cv2_module.zeros((480, 640, 3), dtype="uint8")
        img[:] = (40, 40, 40)
        cv2_module.putText(
            img, "Camera starting...", (120, 250),
            cv2_module.FONT_HERSHEY_SIMPLEX, 1.2, (180, 180, 180), 2,
        )
        cv2_module.putText(
            img, "Webcam & YOLO will appear here", (100, 300),
            cv2_module.FONT_HERSHEY_SIMPLEX, 0.7, (120, 120, 120), 1,
        )
        return img
    except Exception:
        return None


# Tactical map zone (same as Drone2 backend config) for mapping detections when _phantom is None
TACTICAL_ZONE_DRONE1 = {"x_min": 50, "y_min": 50, "x_max": 370, "y_max": 550}
TACTICAL_ZONE_DRONE2 = {"x_min": 430, "y_min": 50, "x_max": 750, "y_max": 550}


def _bbox_to_center(bbox: List[float]) -> List[float]:
    """Return [cx, cy] from bbox [x1, y1, x2, y2]."""
    if len(bbox) < 4:
        return [0.0, 0.0]
    return [(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0]


def _map_detections_to_zone(detections: List[Dict], zone: Dict, frame_width: int, frame_height: int) -> List[Dict]:
    """Map detections with bbox to zone coords (map_x, map_y). Same logic as Drone2 detection_mapper."""
    if not detections or frame_width <= 0 or frame_height <= 0:
        return []
    x_min, y_min = zone["x_min"], zone["y_min"]
    x_max, y_max = zone["x_max"], zone["y_max"]
    zone_w, zone_h = x_max - x_min, y_max - y_min
    out = []
    for d in detections:
        bbox = d.get("bbox", [0, 0, 0, 0])
        cx, cy = _bbox_to_center(bbox)
        norm_x = max(0, min(1, cx / frame_width))
        norm_y = max(0, min(1, cy / frame_height))
        map_x = x_min + norm_x * zone_w
        map_y = y_min + norm_y * zone_h
        out.append({**d, "map_x": round(map_x, 1), "map_y": round(map_y, 1)})
    return out


def _eager_load_drone_yolo() -> None:
    """Try to load Drone YOLO once; set phantom_state['yolo_error'] and print if it fails."""
    phantom_state["yolo_error"] = None
    try:
        models.load_yolo()
    except Exception as e:
        phantom_state["yolo_error"] = str(e)
        print(f"  YOLO: Drone CPU failed to load — {e}")
        print("  Install with: pip install ultralytics  (yolov8n.pt will download on first run)")


def _eager_load_depth() -> None:
    """Load Depth Anything V2 via HuggingFace transformers (auto-downloads model)."""
    global _depth_estimator
    try:
        from depth_estimator import DepthEstimator
    except ImportError:
        try:
            sys.path.insert(0, str(_HERE))
            from depth_estimator import DepthEstimator
        except ImportError:
            print("  Depth: depth_estimator.py not found — skipping")
            return

    _depth_estimator = DepthEstimator()
    if _depth_estimator.loaded:
        phantom_state["depth_enabled"] = True
        phantom_state["depth_provider"] = _depth_estimator.provider
    else:
        print("  Depth: failed to load — check that transformers and torch are installed")


def _run_drone_yolo_on_frame(bgr_frame) -> List[Dict]:
    """Run Drone's models.detect_objects on a BGR frame. Returns list of {class, confidence, bbox, center}."""
    if bgr_frame is None:
        return []
    try:
        import cv2
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        dets = models.detect_objects(image)
        result = []
        for d in dets:
            bbox = list(d.get("bbox", [0, 0, 0, 0]))
            center = _bbox_to_center(bbox)
            result.append({
                "class": d.get("class", "?"),
                "confidence": float(d.get("confidence", 0)),
                "bbox": bbox,
                "center": center,
            })
        return result
    except Exception as e:
        phantom_state["yolo_error"] = str(e)
        return []


def _draw_yolo_and_depth_on_frame(cv2_module, frame, detections: List[Dict]):
    """Draw YOLO boxes and labels (with depth in cm when present) on frame for real-time webcam view."""
    if frame is None or not detections:
        return frame
    try:
        for d in detections:
            bbox = d.get("bbox", [0, 0, 0, 0])
            if len(bbox) < 4:
                continue
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            label = d.get("class", "?")
            if d.get("distance_meters") is not None:
                label += f" {int(d['distance_meters'] * 100 / 25)}"
            cv2_module.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            (tw, th), _ = cv2_module.getTextSize(label, cv2_module.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2_module.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), (0, 0, 255), -1)
            cv2_module.putText(frame, label, (x1 + 2, y1 - 4), cv2_module.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    except Exception:
        pass
    return frame


async def _simple_camera_loop():
    """Grab frames from laptop webcam (device 0) when Drone2 camera is not available."""
    global _simple_camera_frame, _simple_camera_jpeg, _simple_capture
    try:
        import cv2
    except ImportError:
        return
    if _simple_capture is None or not _simple_capture.isOpened():
        return
    loop = asyncio.get_event_loop()
    while True:
        try:
            # Run blocking cap.read() in a thread so it doesn't stall the event loop
            ret, frame = await loop.run_in_executor(None, _simple_capture.read)
            if ret and frame is not None:
                frame = cv2.resize(frame, (640, 480))
                _simple_camera_frame = frame
                _, jpeg = cv2.imencode(".jpg", frame)
                _simple_camera_jpeg = jpeg.tobytes()
        except Exception:
            pass
        await asyncio.sleep(0.033)


async def _display_loop():
    """Output feed at a steady 30 FPS (smooth like Zoom). Uses latest camera frame + latest YOLO/depth detections. When recording, writes vis to clip for Agent import."""
    global _recording_writer, _recording_path
    import cv2
    while True:
        await asyncio.sleep(0.033)
        if _simple_camera_frame is None:
            continue
        try:
            frame = _simple_camera_frame.copy()
            detections = phantom_state.get("raw_detections", [])
            vis = _draw_yolo_and_depth_on_frame(cv2, frame, detections)
            _, jpeg = cv2.imencode(".jpg", vis)
            jpeg_bytes = jpeg.tobytes()
            phantom_state["processed_frames"]["Drone-1"] = jpeg_bytes
            phantom_state["processed_frames"]["Drone-2"] = jpeg_bytes
            if phantom_state.get("recording") and _recording_path:
                if _recording_writer is None:
                    h, w = vis.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    _recording_writer = cv2.VideoWriter(_recording_path, fourcc, 30.0, (w, h))
                if _recording_writer is not None:
                    _recording_writer.write(vis)
        except Exception:
            pass


def _recreate_phantom_yolo():
    """Try to recreate YOLO detector on NPU (e.g. after HTP subsystem restart). Returns True if NPU session created."""
    global phantom_yolo_detector
    if not _phantom:
        return False
    p = _phantom
    cfg = p["config"]
    _yolo_path = _phantom_model_path
    _yolo_input_size = 640
    if getattr(cfg, "YOLO_USE_FAST", False) and _phantom_model_path_fast.exists():
        _yolo_path = _phantom_model_path_fast
        _yolo_input_size = 320
    if not _yolo_path.exists():
        return False
    try:
        phantom_yolo_detector = p["YOLODetector"](
            str(_yolo_path),
            qnn_dll_path=getattr(cfg, "QNN_DLL_PATH", None),
            confidence_threshold=getattr(cfg, "YOLO_CONFIDENCE_THRESHOLD", 0.45),
            input_size=_yolo_input_size,
            use_gpu=getattr(cfg, "USE_GPU", True),
            split_npu_gpu=getattr(cfg, "SPLIT_NPU_GPU", False),
            prefer_npu_over_gpu=getattr(cfg, "PREFER_NPU_OVER_GPU", True),
        )
        if "QNNExecutionProvider" in phantom_yolo_detector.get_provider():
            return True
    except Exception:
        phantom_yolo_detector = None
    return False


async def phantom_background_loop():
    """Run YOLO (and optional advisory) on camera frames. Uses phantom_camera_manager or simple laptop frame."""
    global phantom_camera_manager, phantom_yolo_detector, _npu_fail_count
    if not _phantom:
        return
    p = _phantom
    while True:
        try:
            if phantom_camera_manager is not None:
                frames = phantom_camera_manager.grab_all_frames()
            elif _simple_camera_frame is not None:
                frames = {"Drone-1": _simple_camera_frame.copy(), "Drone-2": _simple_camera_frame.copy()}
            else:
                await asyncio.sleep(0.1)
                continue
            phantom_state["detections"] = {}
            for drone_id, frame in frames.items():
                phantom_state["feeds"][drone_id] = frame is not None
                if frame is None:
                    phantom_state["detections"][drone_id] = []
                    continue
                try:
                    h, w = frame.shape[:2]
                    loop = asyncio.get_event_loop()
                    # Run YOLO and Depth in parallel on NPU to maximize utilization
                    if phantom_yolo_detector is not None and _depth_estimator is not None and _depth_estimator.loaded:
                        def _run_yolo():
                            try:
                                return phantom_yolo_detector.detect(frame), None
                            except Exception as e1:
                                return [], e1
                        def _run_depth():
                            try:
                                t0 = time.time()
                                out = _depth_estimator.infer(frame)
                                return out, (time.time() - t0) * 1000
                            except Exception:
                                return None, 0.0
                        (detections, yolo_err), (depth_map, depth_latency_ms) = await asyncio.gather(
                            loop.run_in_executor(None, _run_yolo),
                            loop.run_in_executor(None, _run_depth),
                        )
                        phantom_state["depth_latency_ms"] = depth_latency_ms
                        if yolo_err and phantom_yolo_detector is not None:
                            try:
                                detections = phantom_yolo_detector.detect(frame)
                                _npu_fail_count = 0
                            except Exception:
                                _npu_fail_count = phantom_state.get("_npu_fail_count", 0) + 1
                                phantom_state["_npu_fail_count"] = _npu_fail_count
                                detections = []
                        else:
                            _npu_fail_count = 0
                        if phantom_yolo_detector is not None and detections is not None:
                            phantom_state["npu_provider"] = phantom_yolo_detector.get_provider()
                            phantom_state["yolo_latency_ms"] = phantom_yolo_detector.get_last_latency()
                        if depth_map is not None:
                            for d in detections:
                                d["distance_meters"] = _depth_estimator.depth_at_bbox(
                                    depth_map, d.get("bbox", [0, 0, 1, 1]), w, h
                                )
                    elif phantom_yolo_detector is not None:
                        try:
                            detections = phantom_yolo_detector.detect(frame)
                            _npu_fail_count = 0
                        except Exception as e1:
                            try:
                                detections = phantom_yolo_detector.detect(frame)
                                _npu_fail_count = 0
                            except Exception:
                                _npu_fail_count = phantom_state.get("_npu_fail_count", 0) + 1
                                phantom_state["_npu_fail_count"] = _npu_fail_count
                                detections = []
                                if _npu_fail_count >= 5:
                                    if _recreate_phantom_yolo():
                                        phantom_state["_npu_fail_count"] = 0
                                        phantom_state["npu_provider"] = phantom_yolo_detector.get_provider()
                                        print("  YOLO: NPU session recreated (recovery after HTP glitch)")
                                if phantom_state.get("_last_phantom_log_time", 0) < time.time() - 10:
                                    print(f"  Phantom YOLO frame: {e1}")
                                    phantom_state["_last_phantom_log_time"] = time.time()
                        if phantom_yolo_detector is not None and detections is not None:
                            phantom_state["npu_provider"] = phantom_yolo_detector.get_provider()
                            phantom_state["yolo_latency_ms"] = phantom_yolo_detector.get_last_latency()
                        if _depth_estimator is not None and _depth_estimator.loaded:
                            try:
                                t_d = time.time()
                                depth_map = _depth_estimator.infer(frame)
                                phantom_state["depth_latency_ms"] = (time.time() - t_d) * 1000
                                if depth_map is not None:
                                    for d in detections:
                                        d["distance_meters"] = _depth_estimator.depth_at_bbox(
                                            depth_map, d.get("bbox", [0, 0, 1, 1]), w, h
                                        )
                            except Exception:
                                pass
                    else:
                        detections = _run_drone_yolo_on_frame(frame)
                        if _depth_estimator is not None and _depth_estimator.loaded:
                            try:
                                depth_map = _depth_estimator.infer(frame)
                                if depth_map is not None:
                                    for d in detections:
                                        d["distance_meters"] = _depth_estimator.depth_at_bbox(
                                            depth_map, d.get("bbox", [0, 0, 1, 1]), w, h
                                        )
                            except Exception:
                                pass
                    # Clear error when we successfully get detections
                    phantom_state["yolo_error"] = None
                    mapped = p["detection_mapper"].map_detections(drone_id, detections, w, h)
                    phantom_state["detections"][drone_id] = mapped
                    if drone_id == "Drone-1":
                        phantom_state["raw_detections"] = list(detections)
                    # When using laptop camera, _display_loop writes processed_frames at 30 FPS for smooth video
                    if phantom_camera_manager is not None:
                        vis = _draw_yolo_and_depth_on_frame(p["cv2"], frame.copy(), detections)
                        _, jpeg = p["cv2"].imencode(".jpg", vis)
                        phantom_state["processed_frames"][drone_id] = jpeg.tobytes()
                except Exception as e:
                    phantom_state["detections"][drone_id] = []
                    if phantom_state.get("_last_phantom_log_time", 0) < time.time() - 10:
                        print(f"  Phantom YOLO frame: {e}")
                        phantom_state["_last_phantom_log_time"] = time.time()
                    if phantom_camera_manager is not None:
                        try:
                            vis = _draw_yolo_and_depth_on_frame(p["cv2"], frame.copy(), [])
                            _, jpeg = p["cv2"].imencode(".jpg", vis)
                            phantom_state["processed_frames"][drone_id] = jpeg.tobytes()
                        except Exception:
                            pass
            now = time.time()
            if now - phantom_state["last_llm_time"] >= getattr(p["config"], "LLM_UPDATE_INTERVAL", 12):
                phantom_state["last_llm_time"] = now
                try:
                    summary = p["detection_mapper"].get_detection_summary(phantom_state["detections"])
                    mission = phantom_state["current_mission"]
                    loop = asyncio.get_event_loop()
                    text = await loop.run_in_executor(None, p["get_advisory"], summary, mission)
                    phantom_state["advisory"] = {
                        "text": text,
                        "mission": mission,
                        "timestamp": time.strftime("%H:%M:%S", time.localtime()),
                    }
                except Exception:
                    pass
        except Exception:
            pass
        await asyncio.sleep(0.033)


def _blocking_yolo(frame):
    """Run YOLO inference only (called in thread executor)."""
    try:
        detections = _run_drone_yolo_on_frame(frame)
        return detections, None
    except Exception as e:
        return [], str(e)


def _blocking_depth(frame):
    """Run depth inference only (called in thread executor)."""
    if _depth_estimator is None or not _depth_estimator.loaded:
        return None, 0.0
    try:
        t_d = time.time()
        depth_map = _depth_estimator.infer(frame)
        return depth_map, (time.time() - t_d) * 1000
    except Exception:
        return None, 0.0


# Cached detections from the last completed inference — drawn on every live frame
_cached_detections: list = []
_inference_running = False


async def _inference_background_loop():
    """Run YOLO and depth in parallel threads; updates cached detections when both finish."""
    global _cached_detections, _inference_running
    import asyncio as _asyncio
    loop = _asyncio.get_event_loop()
    while True:
        try:
            if not phantom_state.get("yolo_enabled", False) or _simple_camera_frame is None:
                _cached_detections = []
                await _asyncio.sleep(0.1)
                continue
            frame = _simple_camera_frame.copy()
            _inference_running = True
            # Run YOLO (CPU/NPU) and Depth (HTP/NPU) concurrently in separate threads
            (detections, yolo_error), (depth_map, depth_latency) = await _asyncio.gather(
                loop.run_in_executor(None, _blocking_yolo, frame),
                loop.run_in_executor(None, _blocking_depth, frame),
            )
            _inference_running = False
            if phantom_yolo_detector is not None:
                phantom_state["npu_provider"] = phantom_yolo_detector.get_provider()
                phantom_state["yolo_latency_ms"] = phantom_yolo_detector.get_last_latency()
            h, w = frame.shape[:2]
            if depth_map is not None:
                for d in detections:
                    d["distance_meters"] = _depth_estimator.depth_at_bbox(
                        depth_map, d.get("bbox", [0, 0, 1, 1]), w, h
                    )
            _cached_detections = detections
            phantom_state["yolo_error"] = yolo_error
            phantom_state["depth_latency_ms"] = depth_latency
            phantom_state["raw_detections"] = list(detections)
            phantom_state["detections"]["Drone-1"] = _map_detections_to_zone(detections, TACTICAL_ZONE_DRONE1, w, h)
            phantom_state["detections"]["Drone-2"] = _map_detections_to_zone(detections, TACTICAL_ZONE_DRONE2, w, h)
        except Exception as e:
            _inference_running = False
            if phantom_state.get("_last_yolo_log_time", 0) < time.time() - 10:
                print(f"  Inference loop: {e}")
                phantom_state["_last_yolo_log_time"] = time.time()
        # Throttle to ~30 FPS so frame and overlay stay in sync (reduces glitching)
        await _asyncio.sleep(0.033)


async def _simple_yolo_loop():
    """Frame loop: draw YOLO + depth on live camera frame for real-time webcam-like stream."""
    import cv2
    while True:
        try:
            if _simple_camera_frame is None:
                await asyncio.sleep(0.05)
                continue
            frame = _simple_camera_frame.copy()

            if not phantom_state.get("yolo_enabled", False):
                phantom_state["raw_detections"] = []
                phantom_state["detections"]["Drone-1"] = []
                phantom_state["detections"]["Drone-2"] = []
                vis = frame.copy()
            else:
                detections = phantom_state.get("raw_detections", [])
                vis = _draw_yolo_and_depth_on_frame(cv2, frame.copy(), detections)

            _, jpeg = cv2.imencode(".jpg", vis)
            phantom_state["processed_frames"]["Drone-1"] = jpeg.tobytes()
            phantom_state["processed_frames"]["Drone-2"] = jpeg.tobytes()
        except Exception as e:
            if phantom_state.get("_last_yolo_log_time", 0) < time.time() - 10:
                print(f"  YOLO loop: {e}")
                phantom_state["_last_yolo_log_time"] = time.time()
        await asyncio.sleep(0.033)


# Synthetic position for world graph when no real GPS (laptop webcam)
_world_graph_counter = 0
BASE_LAT, BASE_LON = 37.7, -122.4


async def _world_graph_ingest_loop():
    """Periodically add current frame + detections to world graph so the map has nodes."""
    global _world_graph_counter
    import cv2
    import base64
    if _world_graph is None:
        return
    while True:
        await asyncio.sleep(2.5)
        try:
            raw = phantom_state.get("raw_detections", [])
            frame = _simple_camera_frame
            if frame is None:
                continue
            # Synthetic position so we get a new node every time (MIN_DISTANCE ~0.5m)
            _world_graph_counter += 1
            lat = BASE_LAT + _world_graph_counter * 0.00005
            lon = BASE_LON + _world_graph_counter * 0.00005
            yolo_detections = [{"class": d.get("class", "?"), "confidence": float(d.get("confidence", 0)), "bbox": list(d.get("bbox", [0, 0, 0, 0]))} for d in raw]
            thumb = cv2.resize(frame, (160, 120))
            _, jpeg = cv2.imencode(".jpg", thumb)
            image_b64 = base64.b64encode(jpeg.tobytes()).decode("ascii")
            _world_graph.add_node(lat=lat, lon=lon, alt=10.0, yaw=0.0, yolo_detections=yolo_detections, image_b64=image_b64)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup_event():
    """Load data on startup. Always try laptop webcam first (as in original Drone setup)."""
    global phantom_camera_manager, phantom_yolo_detector, _simple_capture, _placeholder_jpeg
    try:
        image_db.load()
    except Exception as e:
        print(f"  Image DB: skip ({e})")

    # Placeholder image so /api/feed never 503-flickers before first frame
    try:
        import cv2
        img = _make_placeholder_frame(cv2)
        if img is not None:
            _, jpeg = cv2.imencode(".jpg", img)
            _placeholder_jpeg = jpeg.tobytes()
    except Exception:
        pass

    # 1) Laptop webcam: try index 0, 1; on Windows try DSHOW then MSMF
    try:
        import cv2
        def _open_cam(idx, api=cv2.CAP_ANY):
            cap = cv2.VideoCapture(idx, api) if api != cv2.CAP_ANY else cv2.VideoCapture(idx)
            if cap.isOpened():
                # Warmup: some webcams need a few reads before giving valid frames
                for _ in range(5):
                    cap.read()
                return cap
            return None
        for idx in [0, 1]:
            cap = _open_cam(idx)
            if cap is not None:
                _simple_capture = cap
                asyncio.create_task(_simple_camera_loop())
                phantom_state["feeds"]["Drone-1"] = True
                phantom_state["feeds"]["Drone-2"] = True
                print(f"  Camera: laptop webcam (index {idx})")
                break
        if _simple_capture is None and os.name == "nt":
            for api, name in [(cv2.CAP_DSHOW, "DSHOW"), (cv2.CAP_MSMF, "MSMF")]:
                cap = _open_cam(0, api)
                if cap is not None:
                    _simple_capture = cap
                    asyncio.create_task(_simple_camera_loop())
                    phantom_state["feeds"]["Drone-1"] = True
                    phantom_state["feeds"]["Drone-2"] = True
                    print(f"  Camera: laptop webcam (index 0, {name})")
                    break
        if _simple_capture is None:
            print("  Camera: no device opened (tried 0, 1, DSHOW, MSMF). Grant app camera access or close other apps using the webcam.")
    except Exception as e:
        print(f"  Camera: {e}")

    # 2) YOLO: Drone2 ONNX if available (prefer 320 for speed), else Drone CPU YOLO on laptop feed
    _yolo_path = _phantom_model_path
    _yolo_input_size = 640
    if _phantom and getattr(_phantom.get("config"), "YOLO_USE_FAST", False):
        if _phantom_model_path_fast.exists():
            _yolo_path = _phantom_model_path_fast
            _yolo_input_size = 320
    if _phantom and _yolo_path.exists():
        p = _phantom
        cfg = p["config"]
        qnn_path = getattr(cfg, "QNN_DLL_PATH", None)
        if not qnn_path:
            try:
                from ai.ort_providers import resolve_qnn_backend_path
                qnn_path = resolve_qnn_backend_path(None)
                if qnn_path:
                    print(f"  NPU: using QnnHtp.dll at {qnn_path[:60]}...")
            except Exception:
                pass
        try:
            phantom_yolo_detector = p["YOLODetector"](
                str(_yolo_path),
                qnn_dll_path=qnn_path,
                confidence_threshold=getattr(cfg, "YOLO_CONFIDENCE_THRESHOLD", 0.45),
                input_size=_yolo_input_size,
                use_gpu=getattr(cfg, "USE_GPU", True),
                split_npu_gpu=getattr(cfg, "SPLIT_NPU_GPU", False),
                prefer_npu_over_gpu=getattr(cfg, "PREFER_NPU_OVER_GPU", True),
            )
            phantom_state["npu_provider"] = phantom_yolo_detector.get_provider()
            phantom_state["yolo_error"] = None
            asyncio.create_task(phantom_background_loop())
            prov = phantom_yolo_detector.get_provider()
            if "QNN" in str(prov):
                print(f"  Drone2 YOLO: loaded on NPU ({_yolo_input_size}x{_yolo_input_size})")
            else:
                print(f"  Drone2 YOLO: loaded {_yolo_input_size}x{_yolo_input_size} (on laptop feed)")
        except Exception as e:
            print(f"  Drone2 YOLO: not loaded ({e})")
            phantom_yolo_detector = None
            if _simple_capture is not None and _simple_capture.isOpened():
                _eager_load_drone_yolo()
                asyncio.create_task(phantom_background_loop())
                print("  YOLO: using Drone CPU fallback (on laptop feed)")
    elif _simple_capture is not None and _simple_capture.isOpened():
        _eager_load_drone_yolo()
        asyncio.create_task(_simple_yolo_loop())
        asyncio.create_task(_inference_background_loop())
        print("  YOLO: Drone CPU (on laptop feed)")

    # 2b) Depth Anything V2 — load alongside YOLO for real distance estimation
    _eager_load_depth()

    # 2c) Smooth 30 FPS display loop (Zoom-like) when using laptop camera
    if _simple_capture is not None and _simple_capture.isOpened():
        asyncio.create_task(_display_loop())

    # 3) World graph for tactical map: ingest laptop feed so map has nodes
    if _world_graph is not None and _simple_capture is not None and _simple_capture.isOpened():
        asyncio.create_task(_world_graph_ingest_loop())
        print("  World graph: ingesting (map will populate)")

    # 4) Agent tab: load emergency manuals into vector DB (data/ and data/emergency_manuals/)
    try:
        from emergency_manuals import ensure_all_manuals
        ensure_all_manuals()
        from ai.vector_db import load_manuals_from_data_dir
        n = load_manuals_from_data_dir()
        manuals_dir = str(_DRONE2_ROOT / "data" / "emergency_manuals")
        if Path(manuals_dir).is_dir():
            n2 = load_manuals_from_data_dir(manuals_dir)
            n += n2
        if n > 0:
            print(f"  Agent (vector DB): loaded {n} manual chunks")
    except Exception as e:
        print(f"  Agent (vector DB): skip ({e})")

    # 5) Combined agent (spatial + knowledge): pre-load in background
    asyncio.create_task(_agent_init_background())

    # Summary: webcam, YOLO, and NPU status
    cam_ok = _simple_capture is not None and _simple_capture.isOpened()
    yolo_ok = phantom_yolo_detector is not None or getattr(models, "yolo", None) is not None
    npu_active = False
    try:
        prov = phantom_state.get("npu_provider") or (phantom_yolo_detector.get_provider() if phantom_yolo_detector is not None else None)
        if prov and "QNN" in str(prov):
            npu_active = True
    except Exception:
        pass
    try:
        from ai.ort_providers import get_available_providers
        avail = get_available_providers()
        if npu_active:
            print("  >>> NPU (QNN): ACTIVE — YOLO/Depth using Snapdragon NPU")
        elif "QNNExecutionProvider" in avail:
            print("  >>> NPU (QNN): available but YOLO loaded on CPU/GPU — check QnnHtp.dll path in config")
        else:
            print("  >>> NPU: NOT IN USE — run .\\scripts\\install_onnxruntime_qnn.ps1 then restart backend. Providers:", sorted(avail))
    except Exception:
        pass
    print(f"  >>> Backend ready. Webcam: {'OK' if cam_ok else 'NOT OPEN'}. YOLO: {'OK' if yolo_ok else 'NOT LOADED (pip install ultralytics?)'}")


@app.get("/getImage")
async def get_image(x: float, y: float, z: float, yaw: float):
    """Get image and navigation metadata for a position."""
    if not image_db.db:
        raise HTTPException(status_code=503, detail="Database not loaded")
    
    idx = image_db.find_best(x, y, z, yaw)
    src = image_db.db[idx]
    allowed = image_db.check_allowed(src["x"], src["y"], src["z"], yaw)
    
    with open(src["path"], "rb") as f:
        jpeg_bytes = f.read()
    
    b64 = base64.b64encode(jpeg_bytes).decode("ascii")
    
    return {
        "image": b64,
        "x": src["x"],
        "y": src["y"],
        "z": src["z"],
        "yaw": src["yaw"],
        "filename": src["filename"],
        "allowed": allowed,
    }


@app.post("/stream_agents")
async def stream_agents(request: AgentRequest):
    """Stream agent exploration events (SSE). Works with or without trajectory data / Llama."""
    from agent_runner import AgentRunner
    import asyncio

    async def event_generator():
        winner_agent_id = None

        # Send agent_started events
        for agent_id in range(request.num_agents):
            agent_yaw = (request.start_yaw + (agent_id % 2) * 180) % 360
            event = {
                "type": "agent_started",
                "agent_id": agent_id,
                "start_pose": {
                    "x": request.start_x,
                    "y": request.start_y,
                    "z": request.start_z,
                    "yaw": agent_yaw,
                },
            }
            yield f"data: {json.dumps(event)}\n\n"

        # If no trajectory frames, send agent_done for each and complete (so UI still works)
        if not getattr(image_db, "db", None) or len(image_db.db) == 0:
            for agent_id in range(request.num_agents):
                yield f"data: {json.dumps({'type': 'agent_done', 'agent_id': agent_id, 'found': False, 'steps': 0, 'trajectory': []})}\n\n"
            yield f"data: {json.dumps({'type': 'session_complete', 'winner_agent_id': None, 'description': 'No trajectory data loaded. Add frames to Drone/data or use Manual tab to capture.'})}\n\n"
            return

        runner = AgentRunner(models, image_db)
        for agent_id in range(request.num_agents):
            agent_yaw = (request.start_yaw + (agent_id % 2) * 180) % 360
            try:
                for event in runner.run_agent(
                    query=request.query,
                    start_x=request.start_x,
                    start_y=request.start_y,
                    start_z=request.start_z,
                    start_yaw=agent_yaw,
                    agent_id=agent_id,
                    max_steps=MAX_STEPS,
                ):
                    yield f"data: {json.dumps(event)}\n\n"
                    if event.get("type") == "agent_found" and winner_agent_id is None:
                        winner_agent_id = agent_id
                        break
                    await asyncio.sleep(0.1)
                if winner_agent_id is not None:
                    break
            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'agent_id': agent_id, 'message': str(e)})}\n\n"

        yield f"data: {json.dumps({'type': 'session_complete', 'winner_agent_id': winner_agent_id, 'description': 'Target found' if winner_agent_id is not None else 'No target found'})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "Access-Control-Allow-Origin": "*"},
    )


@app.get("/health")
async def health():
    return {"status": "ok", "frames": len(image_db.db)}


# ---------------------------------------------------------------------------
# Drone2 API (tactical map, advisory, mission, feed)
# ---------------------------------------------------------------------------
class MissionBody(BaseModel):
    mission: str


@app.get("/graph")
def get_graph():
    """World graph for tactical map (nodes with gps_lat, gps_lon, detections). Same as Drone2 graph_api."""
    if _world_graph is None:
        return {"nodes": [], "node_count": 0, "last_updated": time.time()}
    return _world_graph.get_graph()


@app.get("/api/graph_3d")
def get_graph_3d():
    """3D Map tab: point cloud, path, nodes with poses, and stats."""
    if _world_graph is None:
        return {
            "pointcloud": [],
            "path": [],
            "nodes": [],
            "stats": {
                "node_count": 0,
                "point_count": 0,
                "area_m2": 0.0,
                "survivors": 0,
                "hazards": 0,
                "exits": 0,
                "structural": 0,
            },
        }
    try:
        from pointcloud_builder import build_pointcloud
        points = build_pointcloud(_world_graph)
    except Exception:
        points = _world_graph.to_3d_pointcloud()
    ordered = sorted(_world_graph.nodes.keys())
    path = []
    nodes_payload = []
    for node_id in ordered:
        pos = _world_graph.get_pose_at_node(node_id)
        if pos is not None:
            path.append({"x": pos[0], "y": pos[1], "z": pos[2]})
        n = _world_graph.nodes[node_id]
        d = n.to_dict()
        pos = _world_graph.get_pose_at_node(node_id)
        d["pose"] = [float(pos[0]), float(pos[1]), float(pos[2])] if pos else None
        d["structural_risk_score"] = 0.0  # placeholder
        nodes_payload.append(d)
    st = _world_graph.get_stats()
    cc = st.get("category_counts", {})
    return {
        "pointcloud": [{"x": p[0], "y": p[1], "z": p[2], "r": p[3], "g": p[4], "b": p[5]} for p in points],
        "path": path,
        "nodes": nodes_payload,
        "stats": {
            "node_count": st.get("node_count", 0),
            "point_count": len(points),
            "area_m2": round(st.get("coverage_m2", 0), 2),
            "survivors": cc.get("survivor", 0),
            "hazards": cc.get("hazard", 0),
            "exits": cc.get("exit", 0),
            "structural": cc.get("structural", 0),
        },
    }


@app.get("/api/graph_3d/neighbor")
def get_graph_3d_neighbor(node_id: str, direction: str):
    """Get neighbor node. direction: forward, back, left, right (spatial) or next, prev (video order)."""
    wg = _get_world_graph()
    if wg is None:
        raise HTTPException(status_code=404, detail="No world graph")
    if direction not in ("forward", "back", "left", "right", "next", "prev"):
        raise HTTPException(status_code=400, detail="direction must be forward, back, left, right, next, or prev")
    if direction in ("next", "prev"):
        neighbor_id = wg.get_neighbor_by_order(node_id, direction)
    else:
        neighbor_id = wg.get_neighbor_direction(node_id, direction)
    if neighbor_id is None:
        raise HTTPException(status_code=404, detail="No neighbor in that direction")
    n = wg.nodes[neighbor_id]
    d = n.to_dict()
    pos = wg.get_pose_at_node(neighbor_id)
    d["pose"] = [float(pos[0]), float(pos[1]), float(pos[2])] if pos else None
    d["structural_risk_score"] = 0.0
    return {"node_id": neighbor_id, "node": d}


@app.post("/api/import_video")
@app.post("/api/import_video/")  # allow trailing slash so proxy/redirect doesn't turn POST into GET
def import_video(file: Optional[UploadFile] = File(None, alias="file")):
    """Upload video (MP4/AVI/MOV); process in background. Use multipart form with key 'file'."""
    wg = _get_world_graph()
    if wg is None:
        raise HTTPException(status_code=400, detail="No world graph")
    if file is None:
        raise HTTPException(status_code=400, detail="No file in request. Send multipart form with field 'file'.")
    filename = getattr(file, "filename", None) or ""
    if not filename.strip():
        raise HTTPException(status_code=400, detail="No file uploaded (empty filename)")
    ext = (Path(filename).suffix or "").lower()
    if ext not in (".mp4", ".avi", ".mov", ".mkv", ".webm"):
        raise HTTPException(status_code=400, detail=f"Use MP4, AVI, MOV, MKV, or WEBM (e.g. YouTube downloads). Got: {ext or 'no extension'}")
    import tempfile
    from video_import import run_import_async, get_import_status
    if get_import_status().get("status") == "running":
        raise HTTPException(status_code=409, detail="Import already in progress")
    try:
        contents = file.file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read file: {e}")
    if not contents:
        raise HTTPException(status_code=400, detail="File is empty")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    tmp.write(contents)
    tmp.close()
    run_import_async(tmp.name, wg)
    return {"success": True, "message": "Import started. Poll /api/import_video/status for progress."}


@app.get("/api/import_video/status")
def import_video_status():
    """Progress of video import: status, current, total, message."""
    try:
        from video_import import get_import_status
        return get_import_status()
    except Exception:
        return {"status": "idle", "current": 0, "total": 0, "message": ""}


# ---------------------------------------------------------------------------
# Video upload → YOLO/depth analysis → playback with detections + PDF report
# ---------------------------------------------------------------------------

@app.post("/api/video/analyze")
def api_video_analyze(file: Optional[UploadFile] = File(None, alias="file"), use_depth: bool = False):
    """Upload video; run YOLO (and optional depth) per frame. Returns job_id. Poll GET /api/video/analysis/{job_id} for status and result."""
    if file is None:
        raise HTTPException(status_code=400, detail="Send multipart form with field 'file'.")
    filename = getattr(file, "filename", None) or ""
    if not filename.strip():
        raise HTTPException(status_code=400, detail="No file uploaded")
    ext = (Path(filename).suffix or "").lower()
    if ext not in (".mp4", ".avi", ".mov", ".mkv", ".webm"):
        raise HTTPException(status_code=400, detail="Use MP4, AVI, MOV, MKV, or WEBM")
    try:
        contents = file.file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not contents:
        raise HTTPException(status_code=400, detail="File is empty")
    import tempfile
    from video_analyze import create_job, run_analyze_async, get_analyze_dir
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    tmp.write(contents)
    tmp.close()
    job_id = create_job()
    run_analyze_async(job_id, tmp.name, use_depth=use_depth)
    return {"job_id": job_id, "message": "Analysis started. Poll GET /api/video/analysis/{job_id} for status and playback URL."}


@app.get("/api/video/analysis/{job_id}")
def api_video_analysis_status(job_id: str):
    """Status and result of video analysis. When status=complete: video_url, detections_by_frame, fps, total_frames, summary."""
    from video_analyze import get_job
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job_id")
    out = {
        "job_id": job_id,
        "status": job["status"],
        "current": job["current"],
        "total": job["total"],
        "message": job["message"],
        "error": job.get("error"),
    }
    if job["status"] == "complete":
        out["video_url"] = job.get("video_url")
        out["fps"] = job.get("fps")
        out["total_frames"] = job.get("total_frames")
        out["summary"] = job.get("summary", {})
        out["detections_by_frame"] = job.get("detections_by_frame", [])
    return out


@app.get("/api/video/analysis/{job_id}/report.pdf")
def api_video_analysis_report_pdf(job_id: str):
    """Generate and download PDF report: objects found + plan."""
    from video_analyze import get_job, generate_report_pdf, get_analyze_dir
    job = get_job(job_id)
    if job is None or job["status"] != "complete":
        raise HTTPException(status_code=404, detail="Job not found or not complete")
    out_dir = get_analyze_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"{job_id}_report.pdf"
    if not generate_report_pdf(job_id, pdf_path):
        raise HTTPException(status_code=500, detail="PDF generation failed")
    return FileResponse(str(pdf_path), media_type="application/pdf", filename=f"cipher_video_report_{job_id}.pdf")


@app.get("/api/video/analysis/{job_id}/video")
def api_video_analysis_video(job_id: str):
    """Serve the annotated output video for a completed analysis job."""
    from video_analyze import get_job, get_job_video_path
    job = get_job(job_id)
    if job is None or job["status"] != "complete":
        raise HTTPException(status_code=404, detail="Job not found or not complete")
    video_file = get_job_video_path(job_id)
    if not video_file or not Path(video_file).exists():
        raise HTTPException(status_code=404, detail="Video file not found")
    media_type = "video/mp4"
    return FileResponse(video_file, media_type=media_type)


@app.post("/api/export_vr")
def export_vr():
    """Export PLY + offline VR viewer HTML to exports/; copy three.min.js if needed. Returns URL to open."""
    if _world_graph is None:
        raise HTTPException(status_code=400, detail="No world graph")
    try:
        from vr_exporter import run_export
    except Exception as e:
        raise HTTPException(status_code=500, detail="vr_exporter failed: " + str(e))
    _EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ply_path, html_path = run_export(_world_graph, str(_EXPORTS_DIR))
    # Copy three.min.js from frontend node_modules if present
    three_src = _PROJECT / "frontend" / "node_modules" / "three" / "build" / "three.min.js"
    three_dst = _EXPORTS_DIR / "three.min.js"
    if three_src.exists() and not three_dst.exists():
        import shutil
        shutil.copy2(three_src, three_dst)
    return {"url": "/exports/vr_viewer.html", "success": True}


@app.get("/api/status")
def api_status():
    camera_ready = (
        _simple_camera_jpeg is not None
        or (phantom_camera_manager is not None and _phantom is not None)
    )
    # YOLO is loaded if Drone2 ONNX detector or Drone CPU fallback (models.yolo) is used
    yolo_loaded = (
        phantom_yolo_detector is not None
        or getattr(models, "yolo", None) is not None
    )
    yolo_error = phantom_state.get("yolo_error")
    return {
        "feeds": phantom_state.get("feeds", {}),
        "npu_provider": phantom_state.get("npu_provider", "?"),
        "yolo_latency_ms": round(phantom_state.get("yolo_latency_ms", 0), 1),
        "camera_ready": camera_ready,
        "yolo_loaded": yolo_loaded,
        "yolo_error": yolo_error,
        "depth_enabled": phantom_state.get("depth_enabled", False),
        "depth_latency_ms": round(phantom_state.get("depth_latency_ms", 0), 1),
        "depth_provider": phantom_state.get("depth_provider", "not loaded"),
    }


@app.post("/api/yolo/start")
def api_yolo_start():
    global _recording_path, _recording_writer
    phantom_state["yolo_enabled"] = True
    phantom_state["recording"] = True
    _recording_writer = None
    import tempfile
    _recording_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4", prefix="manual_clip_").name
    return {"yolo_enabled": True, "recording": True}


@app.post("/api/yolo/stop")
def api_yolo_stop():
    global _recording_writer, _recording_path
    phantom_state["yolo_enabled"] = False
    phantom_state["recording"] = False
    phantom_state["raw_detections"] = []
    phantom_state["detections"] = {}
    if _recording_writer is not None and _recording_path:
        try:
            _recording_writer.release()
        except Exception:
            pass
    _recording_writer = None
    _recording_path = None
    return {"yolo_enabled": False}


@app.post("/api/yolo/frame")
async def api_yolo_frame(request: Request):
    """Run YOLO on a single JPEG frame (raw bytes in request body). Returns detections."""
    body = await request.body()
    if not body:
        return JSONResponse({"detections": []})
    try:
        import cv2 as _cv2
        import numpy as _np
        nparr = _np.frombuffer(body, _np.uint8)
        frame = _cv2.imdecode(nparr, _cv2.IMREAD_COLOR)
        if frame is None:
            return JSONResponse({"detections": []})
        detections = _run_drone_yolo_on_frame(frame)
        # Run depth alongside YOLO for stream mode
        if _depth_estimator is not None and _depth_estimator.loaded and detections:
            h, w = frame.shape[:2]
            depth_map = _depth_estimator.infer(frame)
            if depth_map is not None:
                for d in detections:
                    d["distance_meters"] = _depth_estimator.depth_at_bbox(
                        depth_map, d.get("bbox", [0, 0, 1, 1]), w, h
                    )
        return JSONResponse({"detections": detections})
    except Exception as e:
        return JSONResponse({"detections": [], "error": str(e)})


@app.get("/api/detections")
def api_detections():
    # Return detections from either Drone2 YOLO or Drone CPU fallback (_simple_yolo_loop)
    return phantom_state.get("detections", {})


@app.get("/api/advisory")
def api_advisory():
    return phantom_state.get("advisory", {"text": "", "mission": "", "timestamp": ""}) if _phantom else {"text": "", "mission": "", "timestamp": ""}


@app.get("/api/live_frames")
def api_live_frames():
    """Return recent YOLO-processed frames as thumbnails for the Agent gallery."""
    return {"frames": list(_live_frame_buffer)}


class RunYoloBody(BaseModel):
    image_b64: str

@app.post("/api/run_yolo")
async def api_run_yolo(body: RunYoloBody):
    """Run fresh YOLO on a base64 JPEG. Returns detections."""
    try:
        import cv2 as _cv2
        import numpy as _np
        img_bytes = base64.b64decode(body.image_b64)
        nparr = _np.frombuffer(img_bytes, _np.uint8)
        bgr = _cv2.imdecode(nparr, _cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("Could not decode image")
        rgb = _cv2.cvtColor(bgr, _cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        loop = asyncio.get_event_loop()
        raw = await loop.run_in_executor(None, models.detect_objects, image)
        detections = [
            {"class": d.get("class", "?"), "confidence": round(float(d.get("confidence", 0)), 2)}
            for d in raw if d.get("class")
        ]
        return {"detections": detections}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/mission")
def api_mission(body: MissionBody):
    if not _phantom:
        return {"ok": False, "mission": "search_rescue"}
    if body.mission in getattr(_phantom["config"], "MISSIONS", {}):
        phantom_state["current_mission"] = body.mission
        return {"ok": True, "mission": body.mission}
    return {"ok": False, "mission": phantom_state["current_mission"]}


@app.get("/api/feed/{drone_id}")
def api_feed(drone_id: str):
    if phantom_camera_manager is not None and _phantom is not None:
        frame = phantom_camera_manager.grab_frame(drone_id)
        if frame is not None:
            _, jpeg = _phantom["cv2"].imencode(".jpg", frame)
            return Response(content=jpeg.tobytes(), media_type="image/jpeg")
    if _simple_camera_jpeg is not None:
        return Response(content=_simple_camera_jpeg, media_type="image/jpeg")
    if _placeholder_jpeg is not None:
        return Response(content=_placeholder_jpeg, media_type="image/jpeg")
    raise HTTPException(status_code=503, detail="Camera not available")


@app.get("/api/feed/{drone_id}/processed")
def api_feed_processed(drone_id: str):
    # Prefer YOLO-overlay frame (from phantom_background_loop or _simple_yolo_loop)
    jpeg = phantom_state.get("processed_frames", {}).get(drone_id)
    if jpeg:
        return Response(
            content=jpeg,
            media_type="image/jpeg",
            headers={"Cache-Control": "no-store, no-cache, must-revalidate", "Pragma": "no-cache"},
        )
    if _simple_camera_jpeg is not None:
        return Response(
            content=_simple_camera_jpeg,
            media_type="image/jpeg",
            headers={"Cache-Control": "no-store"},
        )
    if _placeholder_jpeg is not None:
        return Response(content=_placeholder_jpeg, media_type="image/jpeg")
    raise HTTPException(status_code=503, detail="Camera not available")


async def _mjpeg_stream_gen(drone_id: str):
    """Yield MJPEG frames for real-time webcam stream (YOLO + depth drawn on server)."""
    boundary = b"frame"
    while True:
        jpeg = phantom_state.get("processed_frames", {}).get(drone_id)
        if jpeg is None:
            jpeg = _simple_camera_jpeg
        if jpeg is None:
            jpeg = _placeholder_jpeg
        if jpeg is None:
            await asyncio.sleep(0.1)
            continue
        if not isinstance(jpeg, bytes) and hasattr(jpeg, "tobytes"):
            jpeg = jpeg.tobytes()
        chunk = b"--" + boundary + b"\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n"
        yield chunk
        await asyncio.sleep(0.033)


@app.get("/api/feed/{drone_id}/stream")
async def api_feed_stream(drone_id: str):
    """Real-time MJPEG stream (like a webcam). YOLO and depth are drawn on each frame on the server."""
    return StreamingResponse(
        _mjpeg_stream_gen(drone_id),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store, no-cache", "Pragma": "no-cache"},
    )


# ---------------------------------------------------------------------------
# Agent tab: tactical query (voice/text) -> vector DB + Genie -> answer + node_ids
# ---------------------------------------------------------------------------

@app.get("/api/agent_response")
def api_agent_response():
    """Last agent answer and node_ids for Agent tab UI."""
    r = phantom_state.get("agent_response", {})
    return {
        "answer": r.get("answer", ""),
        "node_ids": r.get("node_ids", []),
        "ts": r.get("ts", 0),
        "confidence": r.get("confidence", 0.75),
        "agent_used": r.get("agent_used", "KNOWLEDGE"),
        "recommended_action": r.get("recommended_action", ""),
    }


async def _run_voice_query_with_text(text: str):
    """Run query_agent with transcribed/text query; update phantom_state; return response dict."""
    _root = str(_DRONE2_ROOT)
    _backend = str(_DRONE2_ROOT / "backend")
    for p in (_root, _backend):
        if p not in sys.path:
            sys.path.insert(0, p)
    try:
        from ai.query_agent import query_agent
    except ImportError:
        import importlib.util
        _qpath = _DRONE2_ROOT / "backend" / "query_agent.py"
        _spec = importlib.util.spec_from_file_location("query_agent", _qpath, submodule_search_locations=[_backend])
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        query_agent = _mod.query_agent

    # Sync graph to vector DB so "bottle"/object questions can use semantic search when no spatial match
    if _world_graph is not None:
        try:
            if str(_DRONE2_ROOT) not in sys.path:
                sys.path.insert(0, str(_DRONE2_ROOT))
            from ai.vector_db import sync_graph_nodes
            sync_graph_nodes(_world_graph.get_graph)
        except Exception:
            pass

    spatial_answer = ""
    spatial_node_ids = []
    wg = _get_world_graph()
    q = text.lower()
    # Procedural questions (how to respond, what to do) → RAG/manuals only; do not use spatial
    procedural_triggers = (
        "how to", "how do i", "how can i", "what should", "what to do", "how do we",
        "respond to", "procedure", "procedures", "steps for", "guide", "handle ",
        "deal with", "extinguish", "evacuate", "treat ", "assess ", "classify",
    )
    is_procedural = any(t in q for t in procedural_triggers)
    # Run detection-based search only for location-seeking questions when we have nodes with frames
    if wg is not None and getattr(wg, "nodes", None) and len(wg.nodes) >= 1 and not is_procedural:
        nodes_with_frames = sum(1 for n in wg.nodes.values() if getattr(n, "image_b64", None))
        spatial_triggers = (
            "where", "find", "locate", "show me", "in the feed", "spot", "which node", "which nodes",
            "which frame", "which frames", "which image", "which images", "which instance", "which instances",
            "see the", "saw the", "was the", "seen", "detected", "extinguisher", "person", "exit", "door",
            "bottle", "cup", "chair", "table", "object", "cell phone", "book", "laptop",
        )
        # "fire" and "hazard" only as spatial when combined with location intent (where/find/which node)
        location_intent = any(x in q for x in ("where", "find", "locate", "show me", "which node", "which frame", "spot", "in the feed"))
        object_word = any(t in q for t in spatial_triggers) or any(w in q for w in ("node", "frame", "image", "instance"))
        if nodes_with_frames >= 1 and (object_word or (location_intent and ("fire" in q or "hazard" in q))):
            def _spatial_search():
                try:
                    from clip_navigator import (
                        find_best_node,
                        find_top_k_nodes,
                        describe_node,
                        find_nodes_by_detection_class,
                    )
                    # 1) Prefer nodes where YOLO detections match the query (e.g. "bottle", "person")
                    by_det = find_nodes_by_detection_class(text, wg)
                    if by_det:
                        best_id = by_det[0][0]
                        # Return all matching nodes so UI can highlight "at which nodes was X seen"
                        top_ids = [nid for nid, _ in by_det[:15]]
                        desc = describe_node(best_id, wg)
                        if len(by_det) > 1:
                            desc = f"Found at {len(by_det)} node(s)/frame(s): {', '.join(top_ids[:8])}{'...' if len(top_ids) > 8 else ''}. {desc}"
                        return best_id, top_ids, desc
                    # 2) Fall back to CLIP visual similarity
                    best = find_best_node(text, wg)
                    if best:
                        top3 = find_top_k_nodes(text, wg, k=3)
                        desc = describe_node(best, wg)
                        return best, top3, desc
                except Exception:
                    return None, [], ""
                return None, [], ""
            loop = asyncio.get_event_loop()
            best_id, top_ids, desc = await loop.run_in_executor(None, _spatial_search)
            if best_id:
                spatial_node_ids = [best_id] + [n for n in top_ids if n != best_id][:14]
                spatial_answer = f"Found in the feed (see highlighted nodes below). {desc}"

    # Only run knowledge (manuals/vector DB) when we don't have a spatial match — show just the answer to what was asked
    answer = ""
    node_ids: List[str] = []
    confidence = 0.75
    recommended_action = ""
    agent_used = "KNOWLEDGE"
    if spatial_answer:
        # Spatial question answered from feed: return only that, no manual dump
        answer = spatial_answer
        node_ids = list(spatial_node_ids)
        agent_used = "SPATIAL"
        confidence = 0.9
    else:
        # Safety companion: when user asked where exit/door/safety and we found nothing in footage, give RAG guidance
        q = text.lower()
        has_footage = wg is not None and getattr(wg, "nodes", None) and len(wg.nodes) >= 1
        safety_location_asked = (
            has_footage
            and any(phrase in q for phrase in ("exit", "door", "way out", "escape", "safety", "safe", "get out"))
        )
        if safety_location_asked:
            def _safety_companion_run():
                get_graph_callback = _world_graph.get_graph if _world_graph else None
                rag_result = query_agent(
                    "What should I do when I cannot find an exit or door? How do I stay safe and get out?",
                    top_k=4,
                    get_graph_callback=get_graph_callback,
                )
                return rag_result.get("answer", "").strip()
            try:
                safety_guidance = await asyncio.get_event_loop().run_in_executor(None, _safety_companion_run)
                if safety_guidance:
                    answer = (
                        "No exit or door detected in your footage. As your AI safety companion, here's what to do next: "
                        + safety_guidance
                    )
                    confidence = 0.85
            except Exception:
                pass
        if not answer:
            def _run():
                get_graph_callback = _world_graph.get_graph if _world_graph else None
                return query_agent(text, top_k=3, get_graph_callback=get_graph_callback)
            result = await asyncio.get_event_loop().run_in_executor(None, _run)
            answer = result.get("answer", "")
            node_ids = list(result.get("node_ids", []))
            confidence = float(result.get("confidence", 0.75))
            recommended_action = (result.get("recommended_action") or "").strip()
    phantom_state["agent_response"] = {
        "answer": answer, "node_ids": node_ids, "text": text, "ts": time.time(),
        "confidence": confidence, "agent_used": agent_used, "recommended_action": recommended_action,
    }
    return {
        "answer": answer, "node_ids": node_ids, "text": text,
        "confidence": confidence, "agent_used": agent_used, "recommended_action": recommended_action,
    }


@app.post("/api/voice_upload")
async def api_voice_upload(audio: UploadFile = File(..., alias="audio")):
    """Accept audio file (multipart form 'audio'); transcribe with Whisper and run tactical query. Use this for browser voice recording."""
    try:
        if str(_DRONE2_ROOT) not in sys.path:
            sys.path.insert(0, str(_DRONE2_ROOT))
        from ai.voice_input import transcribe_audio
        wav_bytes = await audio.read()
        if not wav_bytes or len(wav_bytes) == 0:
            return {"answer": "Audio was empty. Record for a few seconds then click STOP.", "node_ids": [], "text": ""}
        loop = asyncio.get_event_loop()
        text = await loop.run_in_executor(None, transcribe_audio, wav_bytes)
        if not (text or "").strip():
            return {"answer": "No speech detected. Try speaking clearly and recording a bit longer.", "node_ids": [], "text": ""}
        return await _run_voice_query_with_text((text or "").strip())
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"voice_upload: {e}", exc_info=True)
        return {"answer": f"Voice failed: {e}. Install: pip install transformers soundfile pydub; for WebM install ffmpeg.", "node_ids": [], "text": ""}


@app.post("/api/voice_query")
async def api_voice_query(request: Request):
    """Text (JSON) or legacy multipart: runs query_agent. For browser voice, use POST /api/voice_upload with form 'audio'."""
    try:
        content_type = (request.headers.get("content-type") or "").lower()
        text = ""
        if "application/json" in content_type:
            body = await request.json()
            text = (body.get("text") or "").strip()
        elif "multipart/form-data" in content_type:
            form = await request.form()
            upload = form.get("audio")
            if upload is None:
                for key in form:
                    v = form[key]
                    if hasattr(v, "read"):
                        upload = v
                        break
            if upload is not None and hasattr(upload, "read"):
                if str(_DRONE2_ROOT) not in sys.path:
                    sys.path.insert(0, str(_DRONE2_ROOT))
                from ai.voice_input import transcribe_audio
                loop = asyncio.get_event_loop()
                wav_bytes = await upload.read()
                if not wav_bytes or len(wav_bytes) == 0:
                    phantom_state["agent_response"] = {"answer": "Audio was empty. Record for a few seconds then stop.", "node_ids": [], "text": "", "ts": time.time()}
                    return {"answer": phantom_state["agent_response"]["answer"], "node_ids": [], "text": ""}
                text = await loop.run_in_executor(None, transcribe_audio, wav_bytes)
            else:
                text = (form.get("text") or "").strip()
        if not text:
            phantom_state["agent_response"] = {"answer": "No text or audio received. Use VOICE button and POST to /api/voice_upload, or send JSON { \"text\": \"...\" } to this endpoint.", "node_ids": [], "text": "", "ts": time.time()}
            return {"answer": phantom_state["agent_response"]["answer"], "node_ids": [], "text": ""}
        return await _run_voice_query_with_text(text)
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"voice_query: {e}", exc_info=True)
        reason = str(e).strip() or type(e).__name__
        msg = f"Query failed: {reason}. Try again."
        phantom_state["agent_response"] = {"answer": msg, "node_ids": [], "ts": time.time()}
        return {"answer": msg, "node_ids": [], "error": reason}


@app.post("/api/sync_vector_graph")
def api_sync_vector_graph():
    """Sync world graph nodes into vector DB for semantic search."""
    if _world_graph is None:
        return {"synced": 0}
    try:
        if str(_DRONE2_ROOT) not in sys.path:
            sys.path.insert(0, str(_DRONE2_ROOT))
        from ai.vector_db import sync_graph_nodes
        n = sync_graph_nodes(_world_graph.get_graph)
        return {"synced": n}
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"sync_vector_graph: {e}")
        return {"synced": 0}


# ---------------------------------------------------------------------------
# Combined agent (spatial + knowledge) for 3D Map AGENT mode
# ---------------------------------------------------------------------------

@app.get("/api/agent/status")
def api_agent_status():
    """Agent init status: ready, initializing, running. Last result highlights for tactical map."""
    last = agent_state.get("last_result")
    return {
        "ready": agent_state.get("ready", False),
        "initializing": agent_state.get("initializing", True),
        "running": agent_state.get("running", False),
        "highlighted_node_ids": getattr(last, "highlighted_node_ids", []) if last else [],
    }


class AgentRunBody(BaseModel):
    query: str


@app.post("/api/agent/run")
async def api_agent_run(body: AgentRunBody):
    """Run combined agent (spatial + knowledge). Returns answer, highlighted_node_ids, path_to_navigate, etc."""
    wg = _get_world_graph()
    if wg is None:
        raise HTTPException(status_code=400, detail="No world graph")
    agent_state["running"] = True
    try:
        from agent_orchestrator import run_agent
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, lambda: run_agent(body.query or "", wg))
        agent_state["last_result"] = result
        return {
            "answer_text": result.answer_text,
            "highlighted_node_ids": result.highlighted_node_ids,
            "path_to_navigate": result.path_to_navigate,
            "recommended_action": result.recommended_action,
            "confidence": result.confidence,
            "agent_used": result.agent_used,
        }
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"agent/run: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        agent_state["running"] = False


# ---------------------------------------------------------------------------
# Live Camera Analysis (YOLO + Llama on iPhone feed; fallback to Drone2 camera)
# ---------------------------------------------------------------------------

class AnalyzeRequest(BaseModel):
    image_b64: str
    query: Optional[str] = "Describe what you see. Identify any people, objects, hazards, or exits."
    run_llama: bool = True


@app.post("/analyze_frame")
async def analyze_frame(request: AnalyzeRequest):
    """Run YOLO + Llama 3.2 Vision on a single frame.
    
    Returns YOLO detections + Llama scene description.
    """
    try:
        # Decode image
        image_bytes = base64.b64decode(request.image_b64)
        image = Image.open(BytesIO(image_bytes)).convert("RGB")

        # --- YOLO ---
        yolo_detections = models.detect_objects(image)

        # --- Llama Vision (optional, slower) ---
        llama_description = None
        if request.run_llama:
            detected_names = [d["class"] for d in yolo_detections if d["confidence"] > 0.4]
            prompt = (
                f"{request.query}\n\n"
                f"YOLO already detected: {', '.join(detected_names) if detected_names else 'nothing yet'}.\n"
                "Give a concise 1-2 sentence description of the scene."
            )
            try:
                llama_description = models.infer_llama(image, prompt)
                # Strip the prompt echo that transformers sometimes returns
                if prompt in llama_description:
                    llama_description = llama_description.replace(prompt, "").strip()
            except Exception as e:
                llama_description = f"(Llama not available: {e})"

        return {
            "success": True,
            "detections": yolo_detections,
            "detection_count": len(yolo_detections),
            "llama_description": llama_description,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/live_detections")
async def live_detections(run_llama: bool = False):
    """SSE stream: iPhone stream if available; else laptop camera + YOLO + advisory."""
    import httpx

    IPHONE_STREAM = "http://localhost:8002/latest_frame"
    IPHONE_STATUS = "http://localhost:8002/status"

    async def event_stream():
        last_frame_time = None
        frame_counter = 0
        _smooth_dets: list = []
        _smooth_t: float = 0.0
        _SMOOTH_TTL = 0.5
        # Send first event immediately so client gets 200 and doesn't trigger EventSource onerror
        try:
            yield f"data: {json.dumps({'type': 'waiting', 'message': 'Starting camera feed...'})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'waiting', 'message': str(e)})}\n\n"
        while True:
            # Re-check on every iteration so we pick up the camera after it finishes starting
            has_laptop_feed = bool(
                phantom_state.get("feeds", {}).get("Drone-1")
                or phantom_state.get("feeds", {}).get("Drone-2")
                or (_simple_capture is not None and _simple_capture.isOpened())
            )
            try:
                # 1) Try iPhone stream first
                try:
                    async with httpx.AsyncClient(timeout=2.0) as client:
                        status_resp = await client.get(IPHONE_STATUS)
                        status = status_resp.json()
                        if status.get("has_frame"):
                            frame_time = status.get("last_frame_time")
                            if frame_time != last_frame_time:
                                last_frame_time = frame_time
                                frame_resp = await client.get(IPHONE_STREAM)
                                if frame_resp.status_code == 200:
                                    image = Image.open(BytesIO(frame_resp.content)).convert("RGB")
                                    yolo_detections = models.detect_objects(image)
                                    llama_description = None
                                    if run_llama:
                                        try:
                                            prompt = "Describe this scene in 1-2 sentences."
                                            llama_description = models.infer_llama(image, prompt)
                                        except Exception:
                                            pass
                                    event = {
                                        "type": "detections",
                                        "timestamp": frame_time,
                                        "detections": yolo_detections,
                                        "detection_count": len(yolo_detections),
                                        "llama_description": llama_description,
                                    }
                                    yield f"data: {json.dumps(event)}\n\n"
                                    await asyncio.sleep(0.05 if not run_llama else 2.0)
                                    continue
                except Exception:
                    pass

                # 2) Laptop camera: use phantom_state (filled by YOLO loop), or run YOLO inline if still empty
                if has_laptop_feed:
                    raw = phantom_state.get("raw_detections", [])
                    # If background loop hasn't filled detections yet (or YOLO failed there), run YOLO on current frame
                    if not raw and _simple_camera_frame is not None:
                        try:
                            import cv2
                            rgb = cv2.cvtColor(_simple_camera_frame, cv2.COLOR_BGR2RGB)
                            image = Image.fromarray(rgb)
                            raw = models.detect_objects(image)
                        except Exception:
                            pass
                    # Smooth: hold last non-empty detections for 500ms so boxes don't flicker
                    _now = time.time()
                    if raw:
                        _smooth_dets = raw
                        _smooth_t = _now
                    elif _now - _smooth_t < _SMOOTH_TTL:
                        raw = _smooth_dets
                    detections = [
                        {"class": d.get("class", "?"), "confidence": float(d.get("confidence", 0)), "bbox": list(d.get("bbox", [0, 0, 0, 0])), "distance_meters": d.get("distance_meters")}
                        for d in raw
                    ]
                    advisory = phantom_state.get("advisory", {})
                    llama_description = advisory.get("text", "") if run_llama else None
                    yield f"data: {json.dumps({'type': 'detections', 'timestamp': time.time(), 'detections': detections, 'detection_count': len(detections), 'llama_description': llama_description})}\n\n"
                    await asyncio.sleep(0.05)
                    continue

                # 3) Simple laptop webcam fallback (no Drone2)
                if _simple_camera_frame is not None:
                    try:
                        import cv2
                        frame = _simple_camera_frame
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        image = Image.fromarray(rgb)
                        yolo_detections = models.detect_objects(image)
                        llama_description = None
                        if run_llama:
                            try:
                                llama_description = models.infer_llama(image, "Describe this scene in 1-2 sentences.")
                            except Exception:
                                pass
                        detections = [{"class": d.get("class", "?"), "confidence": float(d.get("confidence", 0)), "bbox": list(d.get("bbox", [0, 0, 0, 0]))} for d in yolo_detections]
                        yield f"data: {json.dumps({'type': 'detections', 'timestamp': time.time(), 'detections': detections, 'detection_count': len(detections), 'llama_description': llama_description})}\n\n"
                    except Exception:
                        pass
                    await asyncio.sleep(0.05)
                    continue

                yield f"data: {json.dumps({'type': 'waiting', 'message': 'No camera feed yet'})}\n\n"
                await asyncio.sleep(1.0)
            except Exception as e:
                # Always send "waiting" so UI shows WAITING FOR CAMERA... not AI ERROR (webcam/yolo often need a few seconds)
                yield f"data: {json.dumps({'type': 'waiting', 'message': str(e)})}\n\n"
                await asyncio.sleep(1.0)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "Access-Control-Allow-Origin": "*"},
    )


# ---------------------------------------------------------------------------
# Serve Drone React UI (frontend/dist) — check at request time so no restart needed after build
# ---------------------------------------------------------------------------

# Serve VR export files (must be before catch-all)
_EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/exports", StaticFiles(directory=str(_EXPORTS_DIR)), name="exports")


@app.get("/")
def serve_root():
    index_path = _DRONE_FRONTEND_DIST / "index.html"
    if index_path.is_file():
        return FileResponse(index_path)
    return Response(
        content="<html><body><h1>Cipher</h1><p>API is running. Build the UI: <code>cd Drone/frontend && npm run build</code></p><p>Or run <code>run_drone_full.ps1</code> and open http://localhost:5173</p><p><a href='/health'>/health</a></p></body></html>",
        media_type="text/html",
    )


@app.get("/{full_path:path}")
def serve_drone_spa(full_path: str):
    if full_path.startswith("api/") or full_path in ("getImage", "stream_agents", "health", "live_detections", "analyze_frame"):
        raise HTTPException(status_code=404, detail="Not found")
    index_path = _DRONE_FRONTEND_DIST / "index.html"
    if not index_path.is_file():
        raise HTTPException(status_code=404, detail="Not found")
    safe = os.path.normpath(full_path).lstrip(os.sep)
    if ".." in safe:
        raise HTTPException(status_code=404, detail="Not found")
    if safe:
        file_path = _DRONE_FRONTEND_DIST / safe
        if file_path.is_file():
            return FileResponse(file_path)
    return FileResponse(index_path)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
