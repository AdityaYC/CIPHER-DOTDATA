"""Import video into world graph: keyframes, YOLO, optical-flow pose, optional depth. Runs in background thread."""

import base64
import io
import math
import threading
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# Progress state (shared with app)
_import_status: Dict[str, Any] = {
    "status": "idle",  # idle | running | complete | error
    "current": 0,
    "total": 0,
    "message": "",
    "nodes_added": 0,
}


def get_import_status() -> Dict[str, Any]:
    return dict(_import_status)


def _set_status(status: str, current: int = 0, total: int = 0, message: str = "", nodes_added: Optional[int] = None):
    _import_status["status"] = status
    _import_status["current"] = current
    _import_status["total"] = total
    _import_status["message"] = message
    if nodes_added is not None:
        _import_status["nodes_added"] = nodes_added


def run_import(
    video_path: str,
    world_graph: Any,
    on_progress: Optional[Callable[[int, int, str], None]] = None,
) -> int:
    """Extract ~150 keyframes, run YOLO + optical flow pose, add imported nodes. Returns nodes added."""
    import cv2
    import numpy as np

    try:
        _run_import_impl(video_path, world_graph, on_progress)
    except Exception as e:
        _set_status("error", 0, 0, f"{e}\n{traceback.format_exc()}")
        return 0
    return _import_status.get("nodes_added", 0)


def _run_import_impl(
    video_path: str,
    world_graph: Any,
    on_progress: Optional[Callable[[int, int, str], None]] = None,
) -> None:
    import cv2
    import numpy as np

    _set_status("running", 0, 0, "Opening video...")
    world_graph.clear_imported_nodes()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        _set_status("error", 0, 0, "Could not open video")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    duration_sec = total_frames / max(1.0, fps)
    # Keyframe interval: max(0.5s, duration/120) — target ~120 keyframes, min 0.5s between
    interval_sec = max(0.5, duration_sec / 120.0)
    step = max(1, int(round(interval_sec * fps)))
    frames: List[np.ndarray] = []
    frame_indices: List[int] = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frame_indices.append(idx)
        idx += 1
    cap.release()

    n_keyframes = len(frames)
    if n_keyframes == 0:
        _set_status("error", 0, 0, "No frames read")
        return

    _set_status("running", 0, n_keyframes, f"Processing frame 0 of {n_keyframes}")

    # YOLO: try ultralytics (always available) or ONNX/QNN if present
    def run_yolo(rgb: np.ndarray) -> List[Dict]:
        try:
            from ultralytics import YOLO
            model = YOLO("yolov8n.pt")
            results = model(rgb, conf=0.25, verbose=False)
            out = []
            for r in results:
                if r.boxes is None:
                    continue
                for box in r.boxes:
                    try:
                        cls_idx = int(np.asarray(box.cls).flatten()[0])
                        conf = float(np.asarray(box.conf).flatten()[0])
                        bbox = np.asarray(box.xyxyn).reshape(-1, 4)[0].tolist()
                        out.append({"class": r.names[cls_idx], "confidence": conf, "bbox": bbox})
                    except (IndexError, ValueError, KeyError):
                        continue
            return out
        except Exception:
            return []

    # Optical flow: accumulate pose (x, y, z, yaw). First frame = origin. z from scale, yaw from flow.
    prev_gray = None
    poses: List[Tuple[float, float, float, float]] = []  # (x, y, z, yaw_deg)
    h, w = frames[0].shape[:2]
    for i, rgb in enumerate(frames):
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        if prev_gray is None:
            poses.append((0.0, 0.0, 0.0, 0.0))
            prev_gray = gray
            continue
        # LK optical flow
        pts_prev = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=10)
        if pts_prev is None or len(pts_prev) < 4:
            px, py, pz, pyaw = poses[-1]
            poses.append((px, py, pz, pyaw))
            prev_gray = gray
            continue
        pts_next, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts_prev, None)
        if pts_next is None:
            px, py, pz, pyaw = poses[-1]
            poses.append((px, py, pz, pyaw))
            prev_gray = gray
            continue
        mask = (st.ravel() == 1)
        try:
            pprev = pts_prev[mask]
            pnext = pts_next[mask]
            if pprev.size == 0 or pnext.size == 0:
                raise ValueError("empty")
            # OpenCV can return (M,1,2); ensure 2D (M,2) so [:,0] and [:,1] are valid
            good_prev = np.asarray(pprev, dtype=np.float64).squeeze().reshape(-1, 2)
            good_next = np.asarray(pnext, dtype=np.float64).squeeze().reshape(-1, 2)
            if good_prev.ndim != 2 or good_next.ndim != 2:
                raise ValueError("not 2D")
            if good_prev.shape[1] != 2 or good_next.shape[1] != 2:
                raise ValueError("axis 1 size != 2")
        except (ValueError, IndexError):
            px, py, pz, pyaw = poses[-1]
            poses.append((px, py, pz, pyaw))
            prev_gray = gray
            continue
        if good_prev.shape[0] < 4 or good_next.shape[0] < 4:
            px, py, pz, pyaw = poses[-1]
            poses.append((px, py, pz, pyaw))
            prev_gray = gray
            continue
        try:
            dx = float(np.median(good_next[:, 0] - good_prev[:, 0]))
            dy = float(np.median(good_next[:, 1] - good_prev[:, 1]))
        except (IndexError, Exception):
            px, py, pz, pyaw = poses[-1]
            poses.append((px, py, pz, pyaw))
            prev_gray = gray
            continue
        px, py, pz, pyaw = poses[-1]
        scale = 0.001
        px += dx * scale
        py += dy * scale
        pyaw += dx * 0.05
        pyaw = (pyaw + 180) % 360 - 180
        poses.append((px, py, pz, pyaw))
        prev_gray = gray

    # Fallback: if pose list is missing or wrong length, use simple sequential poses
    if len(poses) != n_keyframes:
        poses = [(float(i) * 0.01, 0.0, 0.0, 0.0) for i in range(n_keyframes)]

    # CLIP for edge logic: 0.3 <= sim < 0.95 => edge; >= 0.95 => skip frame; < 0.3 => no edge
    def _clip_sim(rgb_a: np.ndarray, rgb_b: np.ndarray) -> float:
        try:
            from clip_navigator import embed_frame, _cosine
            ea, eb = embed_frame(rgb_a), embed_frame(rgb_b)
            if ea is None or eb is None:
                return 0.5  # fallback: allow edge
            return _cosine(ea, eb)
        except Exception:
            return 0.5

    nodes_added = 0
    last_added_node_id: Optional[str] = None
    last_added_rgb: Optional[np.ndarray] = None

    for i in range(n_keyframes):
        if _import_status.get("status") == "error":
            break
        _set_status("running", i + 1, n_keyframes, f"Processing frame {i + 1} of {n_keyframes}", nodes_added=nodes_added)
        if on_progress:
            on_progress(i + 1, n_keyframes, f"Processing frame {i + 1} of {n_keyframes}")

        rgb = frames[i]
        sim = 0.5
        if i > 0 and last_added_rgb is not None:
            sim = _clip_sim(last_added_rgb, rgb)
            if sim >= 0.95:
                continue  # skip: too similar, do not create node
        x, y, z, yaw = poses[i]
        try:
            detections = run_yolo(rgb)
        except Exception:
            detections = []
        _, jpeg = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        image_b64 = base64.b64encode(jpeg.tobytes()).decode("ascii")
        try:
            new_node = world_graph.add_imported_node(
                local_x=x, local_y=y, local_z=z,
                yaw=yaw,
                yolo_detections=detections,
                image_b64=image_b64,
                depth_b64=None,
            )
        except Exception:
            try:
                new_node = world_graph.add_imported_node(local_x=x, local_y=y, local_z=z, yaw=yaw, yolo_detections=[], image_b64=image_b64, depth_b64=None)
            except Exception:
                continue
        new_id = new_node.node_id
        nodes_added += 1
        if i > 0 and last_added_node_id is not None and 0.3 <= sim < 0.95:
            world_graph.link_imported(last_added_node_id, new_id)
        last_added_node_id = new_id
        last_added_rgb = rgb.copy()

    _set_status("complete", n_keyframes, n_keyframes, f"IMPORT COMPLETE — {nodes_added} nodes ready to navigate", nodes_added=nodes_added)
    return nodes_added


def run_import_async(video_path: str, world_graph: Any) -> threading.Thread:
    """Start import in a background thread. Returns the thread. Deletes video_path when done."""
    def task():
        try:
            run_import(video_path, world_graph)
        except Exception as e:
            _set_status("error", 0, 0, f"{e}\n{traceback.format_exc()}")
        finally:
            try:
                import os
                if os.path.isfile(video_path):
                    os.unlink(video_path)
            except Exception:
                pass

    t = threading.Thread(target=task, daemon=True)
    t.start()
    return t
