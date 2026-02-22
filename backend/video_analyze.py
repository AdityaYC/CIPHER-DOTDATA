"""
Video analysis worker — YOLO per-frame on uploaded video.
Provides create_job / run_analyze_async / get_job / get_analyze_dir / generate_report_pdf
for use by app.py /api/video/analyze endpoints.
"""

import os
import threading
import uuid
from pathlib import Path
from typing import Optional

# In-memory job store (cleared on backend restart — acceptable for demo)
_jobs: dict = {}
_jobs_lock = threading.Lock()

# Where processed video files and outputs are stored
_ANALYZE_DIR = Path(__file__).resolve().parent.parent / "exports" / "video_analyze"


def get_analyze_dir() -> Path:
    return _ANALYZE_DIR


def create_job() -> str:
    job_id = str(uuid.uuid4())[:8]
    with _jobs_lock:
        _jobs[job_id] = {
            "status": "running",
            "current": 0,
            "total": 0,
            "message": "Starting…",
            "error": None,
            "video_url": None,
            "fps": None,
            "total_frames": None,
            "summary": {},
            "detections_by_frame": [],
        }
    return job_id


def get_job(job_id: str) -> Optional[dict]:
    with _jobs_lock:
        job = _jobs.get(job_id)
        return dict(job) if job else None


def _update_job(job_id: str, **kwargs):
    with _jobs_lock:
        if job_id in _jobs:
            _jobs[job_id].update(kwargs)


def _run_analyze(job_id: str, video_path: str, use_depth: bool = False):
    """Worker: read video, run YOLO on every frame, save output video, build summary."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        _update_job(job_id, status="error", error="cv2 not installed — run: pip install opencv-python")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        _update_job(job_id, status="error", error="Could not open video file")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    _update_job(job_id, total=max(total_frames, 1), message="Analysing frames…")

    # Output video path
    _ANALYZE_DIR.mkdir(parents=True, exist_ok=True)
    out_video_path = _ANALYZE_DIR / f"{job_id}_annotated.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_video_path), fourcc, fps, (width, height))

    # Try to get the YOLO detector from app state; fall back to ultralytics
    yolo_fn = _get_yolo_fn()

    detections_by_frame = []
    objects_found: dict = {}
    frame_idx = 0
    colors = {
        "person": (0, 255, 102),
        "default": (100, 149, 237),
    }

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_dets = []
        if yolo_fn is not None:
            try:
                raw_dets = yolo_fn(frame)
                for det in raw_dets:
                    cls = str(det.get("class_name") or det.get("class", "object"))
                    conf = float(det.get("confidence", 0))
                    bbox = det.get("bbox") or det.get("box") or []
                    dist = det.get("distance_meters")

                    if len(bbox) == 4:
                        x1, y1, x2, y2 = [int(v) for v in bbox]
                        color = colors.get(cls, colors["default"])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        label = cls if dist is None else f"{cls} {round(dist * 100 / 25)}u"
                        cv2.putText(frame, label, (x1, max(y1 - 6, 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
                        frame_dets.append({
                            "class": cls,
                            "confidence": round(conf, 3),
                            "bbox": [x1, y1, x2, y2],
                            "distance_meters": dist,
                        })
                        objects_found[cls] = max(objects_found.get(cls, 0),
                                                  sum(1 for d in frame_dets if d["class"] == cls))
            except Exception:
                pass

        detections_by_frame.append(frame_dets)
        writer.write(frame)
        frame_idx += 1

        if frame_idx % 10 == 0 or frame_idx == total_frames:
            _update_job(job_id, current=frame_idx,
                        message=f"Frame {frame_idx} / {total_frames}")

    cap.release()
    writer.release()

    # Re-encode with H.264 for browser compatibility if ffmpeg available
    final_path = out_video_path
    try:
        import subprocess
        h264_path = _ANALYZE_DIR / f"{job_id}_annotated_h264.mp4"
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", str(out_video_path),
             "-vcodec", "libx264", "-crf", "23", "-preset", "fast",
             "-movflags", "+faststart", str(h264_path)],
            capture_output=True, timeout=120
        )
        if result.returncode == 0 and h264_path.exists():
            final_path = h264_path
            out_video_path.unlink(missing_ok=True)
    except Exception:
        pass  # keep mp4v version if ffmpeg unavailable

    video_url = f"/api/video/analysis/{job_id}/video"

    # Store video path for serving
    with _jobs_lock:
        _jobs[job_id]["_video_file"] = str(final_path)

    _update_job(
        job_id,
        status="complete",
        current=frame_idx,
        total=frame_idx,
        message="Done",
        video_url=video_url,
        fps=fps,
        total_frames=frame_idx,
        summary={"objects_found": objects_found},
        detections_by_frame=detections_by_frame,
    )

    # Clean up temp input
    try:
        os.unlink(video_path)
    except Exception:
        pass


def get_job_video_path(job_id: str) -> Optional[str]:
    with _jobs_lock:
        job = _jobs.get(job_id)
        return job.get("_video_file") if job else None


def run_analyze_async(job_id: str, video_path: str, use_depth: bool = False):
    t = threading.Thread(target=_run_analyze, args=(job_id, video_path, use_depth), daemon=True)
    t.start()


def _get_yolo_fn():
    """Return a callable(frame) -> list[dict] using existing app YOLO or ultralytics fallback."""
    # Try app's phantom YOLO detector first (already loaded, NPU-accelerated)
    try:
        import sys
        app_mod = sys.modules.get("backend.app") or sys.modules.get("app")
        if app_mod:
            detector = getattr(app_mod, "phantom_yolo_detector", None)
            if detector is not None:
                def _phantom_detect(frame):
                    import numpy as np
                    from PIL import Image
                    img = Image.fromarray(frame[:, :, ::-1])  # BGR→RGB
                    return detector.detect(img)
                return _phantom_detect

        models = getattr(app_mod, "models", None) if app_mod else None
        if models and hasattr(models, "detect_objects"):
            def _models_detect(frame):
                from PIL import Image
                img = Image.fromarray(frame[:, :, ::-1])
                return models.detect_objects(img)
            return _models_detect
    except Exception:
        pass

    # Ultralytics fallback
    try:
        from ultralytics import YOLO
        _model = YOLO("yolov8n.pt")

        def _ultralytics_detect(frame):
            results = _model(frame, verbose=False)
            dets = []
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    cls_name = r.names.get(cls_id, str(cls_id))
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                    dets.append({
                        "class_name": cls_name,
                        "confidence": conf,
                        "bbox": [x1, y1, x2, y2],
                        "distance_meters": None,
                    })
            return dets
        return _ultralytics_detect
    except Exception:
        return None


def generate_report_pdf(job_id: str, pdf_path: Path) -> bool:
    """Generate a simple PDF report for the analysis job. Returns True on success."""
    job = get_job(job_id)
    if not job or job["status"] != "complete":
        return False

    try:
        from fpdf import FPDF
    except ImportError:
        # Fallback: write a plain text file as .pdf (not ideal but never crashes)
        try:
            summary = job.get("summary", {})
            objects = summary.get("objects_found", {})
            lines = [
                "CIPHER — Video Analysis Report",
                f"Job ID: {job_id}",
                f"Total frames: {job.get('total_frames', 0)}",
                f"FPS: {job.get('fps', 0):.1f}",
                "",
                "Objects detected (max per frame):",
            ]
            for cls, count in sorted(objects.items(), key=lambda x: -x[1]):
                lines.append(f"  {cls}: {count}")
            pdf_path.write_text("\n".join(lines), encoding="utf-8")
            return True
        except Exception:
            return False

    summary = job.get("summary", {})
    objects = summary.get("objects_found", {})
    total_frames = job.get("total_frames", 0)
    fps = job.get("fps", 0)

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "CIPHER — Video Analysis Report", ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 8, f"Job ID: {job_id}", ln=True)
    pdf.cell(0, 8, f"Total frames analysed: {total_frames}   FPS: {fps:.1f}", ln=True)
    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "Objects Detected (max count in any single frame)", ln=True)
    pdf.set_font("Helvetica", "", 11)
    for cls, count in sorted(objects.items(), key=lambda x: -x[1]):
        pdf.cell(0, 7, f"  {cls}: {count}", ln=True)
    pdf.ln(4)
    pdf.set_font("Helvetica", "I", 9)
    pdf.cell(0, 6, "Generated by CIPHER — On-device disaster response AI. No cloud. No signal.", ln=True)

    try:
        pdf.output(str(pdf_path))
        return True
    except Exception:
        return False
