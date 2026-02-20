# PHANTOM CODE — On-Device AI Tactical Drone Copilot

A tactical AI copilot for drone operators that runs entirely on a single Qualcomm Snapdragon X Elite laptop with **zero cloud dependency**. Phones act as drone cameras; the laptop runs YOLOv8 on the NPU for real-time object detection and a local LLM (Ollama + Phi-3 Mini) for actionable tactical advice.

**This is NOT autonomous drone control. This is an AI advisor that tells a human operator what to do.**

---

## Quick Start (copy-paste commands)

From the **repo root** (the folder that contains `backend/`, `frontend/`, `scripts/`):

```bash
# 1. Create a virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r backend/requirements.txt

# 3. Download the YOLO model (first time only)
python scripts/download_model.py

# 4. Run the server (HTTP — works in browser without cert warnings)
PHANTOM_HTTP_ONLY=1 python backend/main.py
```

Then open in your browser:

- **Tactical map:** http://localhost:8000  
- **Live stream (camera + YOLO):** http://localhost:8000/live  

**One-liner after setup:** from repo root, run `./run.sh` (or `PHANTOM_HTTP_ONLY=1 python backend/main.py`).

---

## Quick Start (Mac — with built-in webcam)

On Mac the app uses your **built-in webcam** by default. YOLO runs on CPU.

```bash
pip install -r backend/requirements.txt
python scripts/download_model.py    # first time only
PHANTOM_HTTP_ONLY=1 python backend/main.py
```

Open **http://localhost:8000** — you’ll see the tactical map, webcam in both sectors, and real-time detections.

---

## Quick Start (use your phone as camera)

**Option A — IP Webcam app (works on iPhone and Android)**  
1. Install **IP Webcam** on your phone and start the server.  
2. Note the URL (e.g. `http://192.168.1.5:8080`).  
3. On the **laptop**, open http://localhost:8000/live and paste that URL in the “Use your phone camera” box → **Connect phone camera**.

**Option B — iPhone camera in Safari (no app)**  
Safari needs HTTPS. Generate certs once, then run without `PHANTOM_HTTP_ONLY`:

```bash
./scripts/generate_ssl_certs.sh     # first time only
python backend/main.py              # uses HTTPS
```

On your iPhone (same Wi‑Fi), open **https://YOUR_MAC_IP:8000/phone**. Tap **Advanced** → **Continue** for the certificate, then **Turn on camera**.  
*(To use the app on the laptop with HTTPS, open https://localhost:8000 and accept the cert.)*

---

## Project layout

| Path        | Description                                      |
|------------|---------------------------------------------------|
| `backend/` | FastAPI server, camera manager, YOLO, LLM advisory |
| `frontend/`| Tactical map and live stream UI (HTML/CSS/JS)      |
| `models/`  | Put `yolov8_det.onnx` here (or run `download_model.py`) |
| `scripts/` | `download_model.py`, `generate_ssl_certs.sh`      |
| `run.sh`   | Run the app (installs deps and model if needed)   |

---

## Commands reference

| What              | Command |
|-------------------|--------|
| Install deps      | `pip install -r backend/requirements.txt` |
| Download YOLO     | `python scripts/download_model.py` |
| Run (HTTP, no SSL)| `PHANTOM_HTTP_ONLY=1 python backend/main.py` |
| Run (with HTTPS)  | `python backend/main.py` (after `./scripts/generate_ssl_certs.sh`) |
| Run via script    | `./run.sh` |

---

## API

- `GET /api/status` — Feed status, NPU provider, YOLO latency  
- `GET /api/detections` — Mapped detections per drone  
- `GET /api/advisory` — Latest LLM advice and mission  
- `POST /api/mission` — Set mission (e.g. `{"mission": "perimeter"}`)  
- `GET /api/feed/{drone_id}` — Raw frame; `GET /api/feed/{drone_id}/processed` — frame with YOLO boxes  

---

## Requirements

- **Mac:** Built-in webcam, CPU inference. Optional: Ollama + `phi3:mini` for LLM advice.  
- **Windows (Qualcomm):** Windows 11, Qualcomm AI Stack (QNN) for NPU; falls back to CPU if unavailable.  
- Python 3.10+, FastAPI, OpenCV, ONNX Runtime, httpx, uvicorn  
