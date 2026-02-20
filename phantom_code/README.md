# PHANTOM CODE — On-Device AI Tactical Drone Copilot

A tactical AI copilot for drone operators that runs entirely on a single Qualcomm Snapdragon X Elite laptop with **zero cloud dependency**. Phones act as drone cameras; the laptop runs YOLOv8 on the NPU for real-time object detection and a local LLM (Ollama + Phi-3 Mini) for actionable tactical advice.

**This is NOT autonomous drone control. This is an AI advisor that tells a human operator what to do.**

## Quick start (on Mac — local prototype)

On Mac the app automatically uses your **built-in webcam** for both sectors (no phones needed). YOLO runs on CPU. Optional: install [Ollama](https://ollama.com) and run `ollama pull phi3:mini` for live tactical advice.

```bash
cd phantom_code
pip install -r backend/requirements.txt
# Optional: get YOLO model if not present
python scripts/download_model.py
python backend/main.py
```

Open **http://localhost:8000** — you should see the tactical map, your webcam feed in both sectors, and real-time detections (person, phone, etc.). Mission selector and (if Ollama is running) advisory panel work as on Windows.

**Live stream (like an IP Webcam link):** Open **http://localhost:8000/live** to see your camera with YOLO boxes drawn in real time — one URL to share or use for demos.

**Use an IP Webcam URL instead of built-in camera:** Set the env var before starting:
```bash
export PHANTOM_IP_CAMERA_URL="http://192.168.1.5:8080/video"
python backend/main.py
```
Replace with your phone’s IP Webcam URL (same idea as visiting an IP link in your previous iteration). The app will use that stream for YOLO processing and the tactical map.

**iPhone camera in Safari (no app):** Safari needs HTTPS for camera. Run once: `./scripts/generate_ssl_certs.sh` (creates `ssl/cert.pem` and `ssl/key.pem`). Then start the server — it will use HTTPS if certs exist. On iPhone (same Wi‑Fi) open **https://YOUR_MAC_IP:8000/phone**. First time: tap **Advanced** → **Continue** for the cert, then **Turn on camera**. No IP Webcam app needed.

## Quick start (on Qualcomm laptop)

1. **Prep**
   - Place `yolov8_det.onnx` in `phantom_code/models/` (export from Qualcomm AI Hub).
   - Install Ollama and run: `ollama pull phi3:mini`
   - Turn on Windows Mobile Hotspot; connect phones; open IP Webcam app and note their IPs.

2. **Config**
   - Edit `backend/config.py`: set `CAMERA_FEEDS` to your phone URLs (e.g. `http://192.168.137.101:8080/video`) and `QNN_DLL_PATH` if needed.

3. **Run**
   ```bash
   cd phantom_code
   pip install -r backend/requirements.txt
   python backend/main.py
   ```
   Open browser to **http://localhost:8000**

## Project layout

- `backend/` — FastAPI server, camera manager, YOLO (ONNX + QNN), detection mapper, LLM advisory
- `frontend/` — Single-page tactical map UI (HTML/CSS/JS)
- `models/` — Put `yolov8_det.onnx` here

## API

- `GET /api/status` — Feed status, NPU provider, YOLO latency
- `GET /api/detections` — Mapped detections per drone
- `GET /api/advisory` — Latest LLM advice and mission
- `POST /api/mission` — Set mission (e.g. `{"mission": "perimeter"}`)
- `GET /api/feed/{drone_id}` — Live frame JPEG for a drone

## Missions

- **Search & Rescue** — Prioritize persons, suggest focus areas
- **Perimeter Surveillance** — Intrusions, coverage gaps
- **Threat Detection** — Threats, unusual patterns, safety
- **Damage Assessment** — Structural damage, routes, survey order

## Requirements

- **Mac:** Built-in webcam, CPU inference. Optional: Ollama + `phi3:mini`.
- **Windows (hackathon):** Windows 11 (Surface Laptop 7 / Snapdragon X Elite), Qualcomm AI Stack (QNN) for NPU; falls back to CPU if unavailable.
- Python 3.10+, FastAPI, OpenCV, ONNX Runtime, httpx, uvicorn
- Ollama with `phi3:mini` for LLM advisory (optional; app runs without it)

See the full spec in the project brief for setup, demo script, and timeline.
