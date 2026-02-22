"""
PHANTOM CODE — FastAPI server: coordinates cameras, YOLO, mapping, LLM; serves API + frontend.
"""

import os
import sys
import asyncio
import json
import logging
import time

# Ensure repo root is on path so "backend" package (vector_db, query_agent, etc.) resolves
_backend_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_backend_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import cv2
import numpy as np
import config
from camera_manager import CameraManager
from perception import YOLODetector
import detection_mapper
from llm_advisory import get_advisory
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import FileResponse, RedirectResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Resolve paths relative to project root (only Drone UI is used)
_model_path = os.path.join(_project_root, "models", "yolov8_det.onnx")
_drone_frontend_dir = os.path.join(_project_root, "Drone", "frontend", "dist")
_ssl_dir = os.path.join(_project_root, "ssl")
_ssl_certfile = os.path.join(_ssl_dir, "cert.pem")
_ssl_keyfile = os.path.join(_ssl_dir, "key.pem")

app = FastAPI(title="PHANTOM CODE — Tactical Drone Intelligence")

# CORS so Drone React app can call this backend
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared state (updated by background loop, read by API)
state = {
    "detections": {},       # {drone_id: [mapped detections]}
    "raw_detections": [],   # last raw YOLO detections for Drone-1 (class, confidence, bbox) for /live_detections
    "feeds": {},            # {drone_id: True/False}
    "processed_frames": {}, # {drone_id: jpeg_bytes} — frame with YOLO boxes drawn
    "advisory": {"text": "", "mission": "search_rescue", "timestamp": ""},
    "npu_provider": "CPUExecutionProvider",
    "yolo_latency_ms": 0.0,
    "current_mission": "search_rescue",
    "last_llm_time": 0.0,
    "camera_url": None,     # current source (URL or "built-in") for UI
    "phone_frame": None,    # BGR numpy array from /api/phone-frame (Safari on phone)
    "phone_frame_time": 0.0,
    # Agentic query (voice + vector DB + Genie): for tactical map UI
    "agent_response": {"answer": "", "node_ids": [], "ts": 0.0},
    "get_graph_callback": None,  # set by app that has world_graph for query fallback
}

camera_manager: CameraManager | None = None
yolo_detector: YOLODetector | None = None


def _run_llm_sync(summary: str, mission: str) -> str:
    return get_advisory(summary, mission)


# How long (seconds) to use the last phone frame before falling back to built-in/URL camera
PHONE_FRAME_MAX_AGE = 3.0


async def background_loop():
    global camera_manager, yolo_detector
    while True:
        try:
            if camera_manager is None:
                await asyncio.sleep(0.5)
                continue
            frames = camera_manager.grab_all_frames()
            # If phone is streaming, use it for Drone-1 (and Drone-2 for demo)
            phone = state.get("phone_frame")
            phone_time = state.get("phone_frame_time", 0)
            if phone is not None and (time.time() - phone_time) < PHONE_FRAME_MAX_AGE:
                frames["Drone-1"] = phone
                frames["Drone-2"] = phone
                state["feeds"]["Drone-1"] = True
                state["feeds"]["Drone-2"] = True
            state["detections"] = {}
            for drone_id, frame in frames.items():
                state["feeds"][drone_id] = frame is not None
                if frame is None:
                    state["detections"][drone_id] = []
                    continue
                try:
                    if yolo_detector is not None:
                        detections = yolo_detector.detect(frame)
                        state["npu_provider"] = yolo_detector.get_provider()
                        state["yolo_latency_ms"] = yolo_detector.get_last_latency()
                    else:
                        detections = []
                    h, w = frame.shape[:2]
                    mapped = detection_mapper.map_detections(drone_id, detections, w, h)
                    state["detections"][drone_id] = mapped
                    if drone_id == "Drone-1":
                        state["raw_detections"] = list(detections)

                    # Draw YOLO boxes on frame for live stream view
                    vis = frame.copy()
                    for d in detections:
                        bbox = d.get("bbox", [0, 0, 0, 0])
                        x1, y1, x2, y2 = [int(round(x)) for x in bbox]
                        cls = d.get("class", "?")
                        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 102), 2)
                        label = cls
                        if d.get("distance_meters") is not None:
                            label += f" {int(d['distance_meters'] * 100 / 25)}"
                        cv2.putText(vis, label, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 102), 1)
                    _, jpeg = cv2.imencode(".jpg", vis)
                    state["processed_frames"][drone_id] = jpeg.tobytes()
                except Exception as e:
                    logger.warning(f"Frame processing error [{drone_id}]: {e}")
                    state["detections"][drone_id] = []
                    state["feeds"][drone_id] = True  # keep showing feed as attempted

            now = time.time()
            if now - state["last_llm_time"] >= config.LLM_UPDATE_INTERVAL:
                state["last_llm_time"] = now
                try:
                    summary = detection_mapper.get_detection_summary(state["detections"])
                    mission = state["current_mission"]
                    loop = asyncio.get_event_loop()
                    text = await loop.run_in_executor(None, _run_llm_sync, summary, mission)
                    state["advisory"] = {
                        "text": text,
                        "mission": mission,
                        "timestamp": time.strftime("%H:%M:%S", time.localtime()),
                    }
                except Exception as e:
                    logger.warning(f"LLM advisory error: {e}")
        except Exception as e:
            logger.warning(f"Background loop error: {e}")
        await asyncio.sleep(0.05)


@app.get("/api/status")
def api_status():
    return {
        "feeds": state.get("feeds", {}),
        "npu_provider": state.get("npu_provider", "?"),
        "yolo_latency_ms": round(state.get("yolo_latency_ms", 0), 1),
    }


@app.get("/api/detections")
def api_detections():
    return state.get("detections", {})


@app.get("/api/advisory")
def api_advisory():
    return state.get("advisory", {"text": "", "mission": "", "timestamp": ""})


@app.get("/live_detections")
async def live_detections(run_llama: bool = False):
    """SSE stream for Drone UI Manual page: YOLO detections + optional advisory text from this backend's camera."""
    async def event_stream():
        while True:
            try:
                feeds = state.get("feeds", {})
                if not feeds.get("Drone-1") and not feeds.get("Drone-2"):
                    yield f"data: {json.dumps({'type': 'waiting', 'message': 'No camera feed yet'})}\n\n"
                    await asyncio.sleep(1.0)
                    continue
                raw = state.get("raw_detections", [])
                # bbox may be numpy; ensure list for JSON
                detections = [
                    {
                        "class": d.get("class", "?"),
                        "confidence": float(d.get("confidence", 0)),
                        "bbox": list(d.get("bbox", [0, 0, 0, 0])),
                    }
                    for d in raw
                ]
                advisory = state.get("advisory", {})
                llama_description = advisory.get("text", "") if run_llama else None
                event = {
                    "type": "detections",
                    "timestamp": time.time(),
                    "detections": detections,
                    "detection_count": len(detections),
                    "llama_description": llama_description,
                }
                yield f"data: {json.dumps(event)}\n\n"
            except Exception as e:
                logger.warning(f"live_detections: {e}")
                yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
            await asyncio.sleep(0.2)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


# --- Drone UI compatibility: stub endpoints when trajectory data not loaded ---
@app.get("/getImage")
async def get_image_stub(x: float = 0, y: float = 0, z: float = 0, yaw: float = 0):
    """Stub for Drone Manual/Replay: trajectory data not loaded in this backend. Use Drone data or /api/feed for live camera."""
    raise HTTPException(
        status_code=503,
        detail="Trajectory database not loaded. Add Drone/data and mount Drone backend for getImage.",
    )


class AgentRequestStub(BaseModel):
    query: str
    start_x: float = 0.0
    start_y: float = 0.0
    start_z: float = 0.0
    start_yaw: float = 0.0
    num_agents: int = 2


@app.post("/stream_agents")
async def stream_agents_stub(request: AgentRequestStub):
    """Stub for Drone Agent mode: sends session_complete immediately. Mount Drone backend for full agent exploration."""
    async def event_stream():
        for agent_id in range(request.num_agents):
            yield f"data: {json.dumps({'type': 'agent_started', 'agent_id': agent_id, 'start_pose': {'x': request.start_x, 'y': request.start_y, 'z': request.start_z, 'yaw': (request.start_yaw + (agent_id % 2) * 180) % 360}})}\n\n"
        yield f"data: {json.dumps({'type': 'session_complete', 'winner_agent_id': None, 'description': 'Trajectory backend not loaded. Add Drone/data for agent exploration.'})}\n\n"
    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@app.get("/health")
async def health():
    """Drone UI health check; also reports frame count when trajectory is loaded."""
    return {"status": "ok", "frames": 0}


class MissionBody(BaseModel):
    mission: str


class CameraBody(BaseModel):
    url: str


def _reopen_camera(feeds: dict):
    """Release current camera and open with new feeds. Updates global camera_manager."""
    global camera_manager
    if camera_manager:
        try:
            camera_manager.release_all()
        except Exception:
            pass
    camera_manager = CameraManager(feeds)
    camera_manager.open_all()
    for did in feeds:
        state["feeds"][did] = camera_manager.captures.get(did) is not None and camera_manager.captures[did] is not None
    state["processed_frames"] = {}  # clear stale frames


def _get_local_ip():
    try:
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return None


@app.get("/api/server-ip")
def api_server_ip(request: Request):
    """Return this machine's local IP and scheme so the phone can open /phone (use https when server has SSL)."""
    scheme = request.scope.get("scheme", "http")
    return {"ip": _get_local_ip(), "port": config.API_PORT, "scheme": scheme}


@app.get("/api/camera")
def api_camera_get():
    """Current camera source (for UI)."""
    return {"url": state.get("camera_url"), "feeds": state.get("feeds", {})}


@app.post("/api/camera")
def api_camera_set(body: CameraBody):
    """Switch to phone camera: paste your IP Webcam URL (e.g. http://192.168.1.5:8080/video). No restart needed."""
    url = (body.url or "").strip()
    if not url:
        return {"ok": False, "message": "URL required"}
    if not url.startswith("http://") and not url.startswith("https://"):
        url = "http://" + url
    if "/video" not in url and not url.endswith("/"):
        url = url.rstrip("/") + "/video"
    try:
        _reopen_camera({"Drone-1": url, "Drone-2": url})
        state["camera_url"] = url
        logger.info(f"Switched camera to {url}")
        return {"ok": True, "url": url, "feeds": state["feeds"]}
    except Exception as e:
        logger.warning(f"Camera switch failed: {e}")
        return {"ok": False, "message": str(e)}


@app.post("/api/mission")
def api_mission(body: MissionBody):
    if body.mission in config.MISSIONS:
        state["current_mission"] = body.mission
        return {"ok": True, "mission": body.mission}
    return {"ok": False, "mission": state["current_mission"]}


@app.get("/api/feed/{drone_id}")
def api_feed(drone_id: str):
    if camera_manager is None:
        return Response(status_code=503)
    frame = camera_manager.grab_frame(drone_id)
    if frame is None:
        return Response(status_code=404)
    _, jpeg = cv2.imencode(".jpg", frame)
    return Response(content=jpeg.tobytes(), media_type="image/jpeg")


@app.get("/api/feed/{drone_id}/processed")
def api_feed_processed(drone_id: str):
    """Live frame with YOLO bounding boxes drawn (for /live demo page)."""
    jpeg = state.get("processed_frames", {}).get(drone_id)
    if not jpeg:
        return Response(status_code=404)
    return Response(content=jpeg, media_type="image/jpeg")


@app.get("/")
def index():
    """Serve Drone React UI only. Build with: cd Drone/frontend && npm run build"""
    path = os.path.join(_drone_frontend_dir, "index.html")
    if os.path.isfile(path):
        return FileResponse(path)
    return Response(
        content="<html><body><h1>Cipher</h1><p>Build the UI: <code>cd Drone/frontend && npm install && npm run build</code></p><p>Or run <code>run_drone_full.ps1</code> and open http://localhost:5173</p><p><a href='/health'>/health</a></p></body></html>",
        media_type="text/html",
    )


@app.post("/api/phone-frame")
async def api_phone_frame(request: Request):
    """Accept a JPEG frame from the phone (Safari /phone page). No IP Webcam app needed."""
    try:
        body = await request.body()
        if not body:
            return {"ok": False, "error": "empty body"}
        nparr = np.frombuffer(body, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return {"ok": False, "error": "invalid image"}
        # Resize to match expected size (camera_manager uses 640x480)
        frame = cv2.resize(frame, (640, 480))
        state["phone_frame"] = frame
        state["phone_frame_time"] = time.time()
        return {"ok": True}
    except Exception as e:
        logger.warning(f"phone-frame: {e}")
        return {"ok": False, "error": str(e)}


@app.get("/phone")
def phone():
    """Redirect to Drone Manual tab (live/phone feed)."""
    return RedirectResponse(url="/manual", status_code=302)


@app.get("/live")
def live():
    """Redirect to Drone Manual tab (live stream)."""
    return RedirectResponse(url="/manual", status_code=302)


# SPA fallback for Drone React app when dist exists
if os.path.isdir(_drone_frontend_dir):
    @app.get("/{full_path:path}")
    def serve_drone_spa(full_path: str):
        if full_path.startswith("api/") or full_path.startswith("phantom"):
            raise HTTPException(status_code=404, detail="Not found")
        if full_path in ("live", "phone", "getImage", "stream_agents", "health"):
            raise HTTPException(status_code=404, detail="Not found")
        safe_path = os.path.normpath(full_path).lstrip(os.sep)
        if ".." in safe_path or os.path.isabs(safe_path):
            raise HTTPException(status_code=404, detail="Not found")
        file_path = os.path.join(_drone_frontend_dir, safe_path) if safe_path else _drone_frontend_dir
        if safe_path and os.path.isfile(file_path):
            return FileResponse(file_path)
        index_path = os.path.join(_drone_frontend_dir, "index.html")
        if os.path.isfile(index_path):
            return FileResponse(index_path)
        raise HTTPException(status_code=404, detail="Not found")


# ---------------------------------------------------------------------------
# Agentic query + voice (emerGen-style): vector DB, Genie, Whisper — all local
# ---------------------------------------------------------------------------

@app.get("/api/agent_response")
def api_agent_response():
    """Last agent answer and node_ids for UI to display and highlight nodes (e.g. 3s)."""
    r = state.get("agent_response", {})
    return {"answer": r.get("answer", ""), "node_ids": r.get("node_ids", []), "ts": r.get("ts", 0)}


@app.post("/api/sync_vector_graph")
def api_sync_vector_graph():
    """Sync current world graph nodes into vector DB for semantic search. No-op if get_graph_callback not set."""
    get_graph = state.get("get_graph_callback")
    if not get_graph:
        return {"synced": 0, "message": "No graph callback set"}
    try:
        from backend.vector_db import sync_graph_nodes
        n = sync_graph_nodes(get_graph)
        return {"synced": n}
    except Exception as e:
        logger.warning(f"sync_vector_graph: {e}")
        return {"synced": 0, "error": str(e)}


@app.post("/api/voice_query")
async def api_voice_query(request: Request):
    """
    Voice or text query: optional audio file (transcribed by Whisper) or JSON { "text": "..." }.
    Runs query_agent (vector DB + Genie); returns answer and node_ids. Stores in state for UI.
    """
    try:
        content_type = request.headers.get("content-type", "")
        text = ""
        if "application/json" in content_type:
            body = await request.json()
            text = (body.get("text") or "").strip()
        elif "multipart/form-data" in content_type:
            form = await request.form()
            if "audio" in form:
                from backend.voice_input import record_and_transcribe, transcribe_audio
                loop = asyncio.get_event_loop()
                file = form["audio"]
                if file and hasattr(file, "read"):
                    wav_bytes = await file.read()
                    text = await loop.run_in_executor(None, transcribe_audio, wav_bytes)
                else:
                    text = await loop.run_in_executor(None, record_and_transcribe)
            else:
                text = (form.get("text") or "").strip()
        if not text:
            return {"answer": "No text or audio received.", "node_ids": []}
        from backend.query_agent import query_agent
        get_graph = state.get("get_graph_callback")
        result = await asyncio.get_event_loop().run_in_executor(
            None, lambda: query_agent(text, top_k=3, get_graph_callback=get_graph)
        )
        state["agent_response"] = {
            "answer": result.get("answer", ""),
            "node_ids": result.get("node_ids", []),
            "ts": time.time(),
        }
        return {"answer": state["agent_response"]["answer"], "node_ids": state["agent_response"]["node_ids"]}
    except Exception as e:
        logger.warning(f"voice_query failed: {e}")
        fallback = "Query failed. Try again or use text."
        state["agent_response"] = {"answer": fallback, "node_ids": [], "ts": time.time()}
        return {"answer": fallback, "node_ids": []}


@app.on_event("startup")
async def startup():
    global camera_manager, yolo_detector
    # Vector DB: load emergency manuals from /data (local, no cloud)
    try:
        from vector_db import load_manuals_from_data_dir
        n = load_manuals_from_data_dir()
        print(f"  Vector DB: {n} manual(s) loaded from data/")
    except Exception as e:
        logger.warning(f"Vector DB manuals: {e}")
    print("Checking NPU...")
    if not os.path.isfile(_model_path):
        print(f"  ONNX model not found at {_model_path}. Place yolov8_det.onnx in phantom_code/models/")
        yolo_detector = None
    else:
        try:
            yolo_detector = YOLODetector(
                _model_path,
                qnn_dll_path=config.QNN_DLL_PATH,
                confidence_threshold=config.YOLO_CONFIDENCE_THRESHOLD,
            )
            state["npu_provider"] = yolo_detector.get_provider()
            print(f"  Provider: {yolo_detector.get_provider()}")
        except Exception as e:
            logger.error(f"YOLO init failed: {e}")
            print("  WARNING: YOLO not loaded. Run with ONNX model in models/ for detection.")
            yolo_detector = None

    print("Connecting cameras...")
    camera_manager = CameraManager(config.CAMERA_FEEDS)
    camera_manager.open_all()
    for did in config.CAMERA_FEEDS:
        state["feeds"][did] = camera_manager.captures.get(did) is not None and camera_manager.captures[did] is not None
        print(f"  {did}: {'OK' if state['feeds'][did] else 'FAIL'}")
    state["camera_url"] = config.IP_CAMERA_URL or ("built-in webcam" if (config.USE_MAC_WEBCAM_DEMO or config.USE_BUILTIN_WEBCAM) else "IP Webcam URLs from config")

    print("Checking LLM...")
    try:
        r = get_advisory("No detections yet.", "search_rescue")
        print(f"  Ollama ({config.LLM_MODEL}): ONLINE" if r != "LLM OFFLINE — Advisory unavailable" else "  Ollama: OFFLINE")
    except Exception as e:
        print(f"  Ollama: OFFLINE ({e})")

    asyncio.create_task(background_loop())


@app.on_event("shutdown")
def shutdown():
    if camera_manager:
        camera_manager.release_all()


if __name__ == "__main__":
    import uvicorn
    force_http = os.environ.get("PHANTOM_HTTP_ONLY", "").strip().lower() in ("1", "true", "yes")
    use_ssl = not force_http and os.path.isfile(_ssl_certfile) and os.path.isfile(_ssl_keyfile)
    if use_ssl:
        print(f"Starting server on https://localhost:{config.API_PORT} (SSL — use https://YOUR_MAC_IP:{config.API_PORT}/phone on iPhone)")
    else:
        print(f"Starting server on http://localhost:{config.API_PORT}")
        print("  For iPhone camera: run scripts/generate_ssl_certs.sh then restart to use HTTPS.")
    uvicorn.run(
        "main:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=False,
        ssl_keyfile=_ssl_keyfile if use_ssl else None,
        ssl_certfile=_ssl_certfile if use_ssl else None,
    )
