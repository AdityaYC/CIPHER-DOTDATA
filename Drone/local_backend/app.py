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
import csv
import json
import math
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional
from io import BytesIO

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse, JSONResponse
from pydantic import BaseModel
from PIL import Image

# Drone2 repo root (parent of Drone folder)
_HERE = Path(__file__).resolve().parent
_DRONE2_ROOT = _HERE.parent.parent
if str(_DRONE2_ROOT) not in sys.path:
    sys.path.insert(0, str(_DRONE2_ROOT))

# Optional Drone2 backend (camera, YOLO ONNX, advisory)
_phantom = None
try:
    import cv2
    import numpy as np
    from backend import config as phantom_config
    # So backend modules that "import config" resolve to backend.config
    sys.modules["config"] = phantom_config
    from backend.camera_manager import CameraManager as PhantomCameraManager
    from backend.perception import YOLODetector as PhantomYOLODetector
    from backend import detection_mapper as phantom_detection_mapper
    from backend.llm_advisory import get_advisory as phantom_get_advisory
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

app = FastAPI(title="Drone Vision — with Drone2 tactical features")

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
    
    def load_yolo(self):
        """Load YOLO model."""
        if self.yolo is not None:
            return
        
        print("Loading YOLO model...")
        from ultralytics import YOLO
        
        self.yolo = YOLO("yolov8n.pt")
        print("YOLO loaded")
    
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
        """Run YOLO object detection (conf=0.2 to get more detections)."""
        self.load_yolo()
        # conf=0.2 so person/objects are detected more readily
        results = self.yolo(image, conf=0.2, verbose=False)
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
try:
    from world_graph import WorldGraph
    _world_graph = WorldGraph()
except Exception:
    pass

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
    "current_mission": "search_rescue",
    "last_llm_time": 0.0,
    "yolo_error": None,  # set if Drone YOLO load or run fails
}
phantom_camera_manager = None
phantom_yolo_detector = None
_phantom_model_path = _DRONE2_ROOT / "models" / "yolov8_det.onnx"

# Simple laptop camera when Drone2 stack is not loaded (no PYTHONPATH or model missing)
_simple_capture = None
_simple_camera_frame = None  # BGR numpy array, updated by background task
_simple_camera_jpeg = None   # bytes for /api/feed/.../processed
_placeholder_jpeg = None     # "Camera starting..." placeholder to avoid 503 flicker


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


async def _simple_camera_loop():
    """Grab frames from laptop webcam (device 0) when Drone2 camera is not available."""
    global _simple_camera_frame, _simple_camera_jpeg, _simple_capture
    try:
        import cv2
    except ImportError:
        return
    if _simple_capture is None or not _simple_capture.isOpened():
        return
    while True:
        try:
            ret, frame = _simple_capture.read()
            if ret and frame is not None:
                frame = cv2.resize(frame, (640, 480))
                _simple_camera_frame = frame
                _, jpeg = cv2.imencode(".jpg", frame)
                _simple_camera_jpeg = jpeg.tobytes()
        except Exception:
            pass
        await asyncio.sleep(0.05)


async def phantom_background_loop():
    """Run YOLO (and optional advisory) on camera frames. Uses phantom_camera_manager or simple laptop frame."""
    global phantom_camera_manager, phantom_yolo_detector
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
                    if phantom_yolo_detector is not None:
                        detections = phantom_yolo_detector.detect(frame)
                        phantom_state["npu_provider"] = phantom_yolo_detector.get_provider()
                        phantom_state["yolo_latency_ms"] = phantom_yolo_detector.get_last_latency()
                    else:
                        # Fallback: Drone's YOLO (CPU) when Drone2 ONNX not loaded
                        detections = _run_drone_yolo_on_frame(frame)
                    h, w = frame.shape[:2]
                    mapped = p["detection_mapper"].map_detections(drone_id, detections, w, h)
                    phantom_state["detections"][drone_id] = mapped
                    if drone_id == "Drone-1":
                        phantom_state["raw_detections"] = list(detections)
                    vis = frame.copy()
                    for d in detections:
                        bbox = d.get("bbox", [0, 0, 0, 0])
                        x1, y1, x2, y2 = [int(round(x)) for x in bbox]
                        cls = d.get("class", "?")
                        conf = d.get("confidence", 0)
                        p["cv2"].rectangle(vis, (x1, y1), (x2, y2), (0, 255, 102), 2)
                        label = f"{cls} {conf:.0%}"
                        p["cv2"].putText(vis, label, (x1, y1 - 6), p["cv2"].FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 102), 1)
                    _, jpeg = p["cv2"].imencode(".jpg", vis)
                    phantom_state["processed_frames"][drone_id] = jpeg.tobytes()
                except Exception:
                    phantom_state["detections"][drone_id] = []
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
        await asyncio.sleep(0.05)


async def _simple_yolo_loop():
    """When Drone2 is not loaded: run Drone YOLO on laptop feed and fill phantom_state for UI + map."""
    import cv2
    while True:
        try:
            if _simple_camera_frame is None:
                await asyncio.sleep(0.1)
                continue
            frame = _simple_camera_frame.copy()
            h, w = frame.shape[:2]
            detections = _run_drone_yolo_on_frame(frame)
            phantom_state["raw_detections"] = list(detections)
            phantom_state["npu_provider"] = "CPUExecutionProvider"
            phantom_state["yolo_latency_ms"] = 0.0
            # Populate /api/detections for Drone2-style tactical map (map_x, map_y)
            mapped1 = _map_detections_to_zone(detections, TACTICAL_ZONE_DRONE1, w, h)
            mapped2 = _map_detections_to_zone(detections, TACTICAL_ZONE_DRONE2, w, h)
            phantom_state["detections"]["Drone-1"] = mapped1
            phantom_state["detections"]["Drone-2"] = mapped2
            vis = frame.copy()
            for d in detections:
                bbox = d.get("bbox", [0, 0, 0, 0])
                x1, y1, x2, y2 = [int(round(x)) for x in bbox]
                cls = d.get("class", "?")
                conf = d.get("confidence", 0)
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 102), 2)
                cv2.putText(vis, f"{cls} {conf:.0%}", (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 102), 1)
            _, jpeg = cv2.imencode(".jpg", vis)
            phantom_state["processed_frames"]["Drone-1"] = jpeg.tobytes()
            phantom_state["processed_frames"]["Drone-2"] = jpeg.tobytes()
        except Exception:
            pass
        await asyncio.sleep(0.2)


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
    image_db.load()

    # Placeholder image so /api/feed never 503-flickers before first frame
    try:
        import cv2
        img = _make_placeholder_frame(cv2)
        if img is not None:
            _, jpeg = cv2.imencode(".jpg", img)
            _placeholder_jpeg = jpeg.tobytes()
    except Exception:
        pass

    # 1) Laptop webcam first (same as Drone/Drone2: camera index 0)
    try:
        import cv2
        _simple_capture = cv2.VideoCapture(0)
        if _simple_capture.isOpened():
            asyncio.create_task(_simple_camera_loop())
            phantom_state["feeds"]["Drone-1"] = True
            phantom_state["feeds"]["Drone-2"] = True
            print("  Camera: laptop webcam (index 0)")
        else:
            _simple_capture.release()
            _simple_capture = None
            print("  Camera: failed to open index 0")
    except Exception as e:
        print(f"  Camera: {e}")

    # 2) YOLO: Drone2 ONNX if available, else Drone CPU YOLO on laptop feed
    if _phantom and _phantom_model_path.exists():
        try:
            p = _phantom
            phantom_yolo_detector = p["YOLODetector"](
                str(_phantom_model_path),
                qnn_dll_path=getattr(p["config"], "QNN_DLL_PATH", None),
                confidence_threshold=getattr(p["config"], "YOLO_CONFIDENCE_THRESHOLD", 0.45),
            )
            phantom_state["npu_provider"] = phantom_yolo_detector.get_provider()
            phantom_state["yolo_error"] = None
            asyncio.create_task(phantom_background_loop())
            print("  Drone2 YOLO: loaded (on laptop feed)")
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
        print("  YOLO: Drone CPU (on laptop feed)")

    # 3) World graph for tactical map: ingest laptop feed so map has nodes
    if _world_graph is not None and _simple_capture is not None and _simple_capture.isOpened():
        asyncio.create_task(_world_graph_ingest_loop())
        print("  World graph: ingesting (map will populate)")


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
    """Stream agent exploration events (SSE)."""
    from agent_runner import AgentRunner
    import asyncio
    
    async def event_generator():
        session_id = str(uuid.uuid4())
        runner = AgentRunner(models, image_db)
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
                    "yaw": agent_yaw
                },
            }
            yield f"data: {json.dumps(event)}\n\n"
        
        # Run agents (simplified - run sequentially for now)
        # In production, you'd want to run them in parallel
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
                    
                    # Check if this agent found the target
                    if event.get("type") == "agent_found" and winner_agent_id is None:
                        winner_agent_id = agent_id
                        # In a real implementation, you'd cancel other agents here
                        break
                    
                    # Small delay to prevent overwhelming the client
                    await asyncio.sleep(0.1)
                
                # If winner found, stop launching more agents
                if winner_agent_id is not None:
                    break
                    
            except Exception as e:
                error_event = {
                    "type": "error",
                    "agent_id": agent_id,
                    "message": str(e),
                }
                yield f"data: {json.dumps(error_event)}\n\n"
        
        # Send session complete
        complete_event = {
            "type": "session_complete",
            "winner_agent_id": winner_agent_id,
            "description": "Target found" if winner_agent_id is not None else "No target found",
        }
        yield f"data: {json.dumps(complete_event)}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
        },
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


@app.get("/api/status")
def api_status():
    camera_ready = (
        _simple_camera_jpeg is not None
        or (phantom_camera_manager is not None and _phantom is not None)
    )
    yolo_loaded = phantom_yolo_detector is not None
    yolo_error = phantom_state.get("yolo_error")
    return {
        "feeds": phantom_state.get("feeds", {}),
        "npu_provider": phantom_state.get("npu_provider", "?"),
        "yolo_latency_ms": round(phantom_state.get("yolo_latency_ms", 0), 1),
        "camera_ready": camera_ready,
        "yolo_loaded": yolo_loaded,
        "yolo_error": yolo_error,
    }


@app.get("/api/detections")
def api_detections():
    return phantom_state.get("detections", {}) if _phantom else {}


@app.get("/api/advisory")
def api_advisory():
    return phantom_state.get("advisory", {"text": "", "mission": "", "timestamp": ""}) if _phantom else {"text": "", "mission": "", "timestamp": ""}


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
    if phantom_camera_manager is not None and _phantom is not None:
        jpeg = phantom_state.get("processed_frames", {}).get(drone_id)
        if jpeg:
            return Response(content=jpeg, media_type="image/jpeg")
    if _simple_camera_jpeg is not None:
        return Response(content=_simple_camera_jpeg, media_type="image/jpeg")
    # Avoid 503 flicker: serve placeholder until first frame is ready
    if _placeholder_jpeg is not None:
        return Response(content=_placeholder_jpeg, media_type="image/jpeg")
    raise HTTPException(status_code=503, detail="Camera not available")


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
    """SSE stream: iPhone stream if available; else Drone2 laptop camera + YOLO + advisory."""
    import httpx

    IPHONE_STREAM = "http://localhost:8002/latest_frame"
    IPHONE_STATUS = "http://localhost:8002/status"

    async def event_stream():
        last_frame_time = None
        frame_counter = 0
        # Use laptop state whenever feeds are active (from _simple_yolo_loop or phantom_background_loop)
        has_laptop_feed = bool(
            phantom_state.get("feeds", {}).get("Drone-1") or phantom_state.get("feeds", {}).get("Drone-2")
        )

        while True:
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
                                    await asyncio.sleep(0.2 if not run_llama else 2.0)
                                    continue
                except Exception:
                    pass

                # 2) Laptop camera: use phantom_state (filled by YOLO loop)
                if has_laptop_feed:
                    raw = phantom_state.get("raw_detections", [])
                    detections = [
                        {"class": d.get("class", "?"), "confidence": float(d.get("confidence", 0)), "bbox": list(d.get("bbox", [0, 0, 0, 0]))}
                        for d in raw
                    ]
                    advisory = phantom_state.get("advisory", {})
                    llama_description = advisory.get("text", "") if run_llama else None
                    yield f"data: {json.dumps({'type': 'detections', 'timestamp': time.time(), 'detections': detections, 'detection_count': len(detections), 'llama_description': llama_description})}\n\n"
                    await asyncio.sleep(0.2)
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
                    await asyncio.sleep(0.2)
                    continue

                yield f"data: {json.dumps({'type': 'waiting', 'message': 'No camera feed yet'})}\n\n"
                await asyncio.sleep(1.0)
            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
                await asyncio.sleep(1.0)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "Access-Control-Allow-Origin": "*"},
    )


# ---------------------------------------------------------------------------
# Serve Drone React UI (frontend/dist) — check at request time so no restart needed after build
# ---------------------------------------------------------------------------

@app.get("/")
def serve_root():
    index_path = _DRONE_FRONTEND_DIST / "index.html"
    if index_path.is_file():
        return FileResponse(index_path)
    return Response(
        content="<html><body><h1>Drone Vision</h1><p>API is running. Build the UI: <code>cd Drone/frontend && npm run build</code></p><p><a href='/health'>/health</a></p></body></html>",
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
