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
# Option A — Qualcomm AI Hub (v8 detection, NPU-optimized):
python scripts/download_model_qualcomm.py
# Option B — Ultralytics only (CPU ONNX):
# python scripts/download_model.py

# 4. Run the server (HTTP — works in browser without cert warnings)
PHANTOM_HTTP_ONLY=1 python backend/main.py
```

Then open in your browser:

- **Drone UI (landing, manual, agent, replay, tactical):** http://localhost:8000 (after building the Drone frontend — see below)
- **Legacy tactical map:** http://localhost:8000/phantom  
- **Live stream (camera + YOLO):** http://localhost:8000/live  

**Drone React UI:** To use the Drone folder’s UI as the default, build it once: `cd Drone/frontend && npm install && npm run build`. Then restart the server; `/` will serve the Drone app. It includes **Live Tactical** (Drone2 features: map, advisory, mission selector, dual feeds).

**One-liner after setup:** from repo root, run `./run.sh` (or `PHANTOM_HTTP_ONLY=1 python backend/main.py`).

---

## Run Drone (main app) with Drone2 features

**Drone** is the main UI (landing, Manual Control, Agent, Replay). It includes **Drone2 features**: laptop webcam, YOLO detections, tactical advisory, mission selector, and status in the Manual tab.

From repo root (Windows PowerShell):

```powershell
.\run_drone_full.ps1
```

- Backend starts in a new window (port 8000).  
- Frontend starts in this terminal (port 5173).  
- **Open http://localhost:5173** in your browser.

Manual Control shows the laptop camera, YOLO detections, and the Tactical panel (advisory, mission, status). Ensure Node is installed and on PATH (or run `$env:Path = "${env:ProgramFiles}\nodejs;$env:Path"` first).

---

## Verify it works (after clone or pull)

Run these from the **repo root** to confirm everything works:

```bash
# Clone (if starting fresh)
git clone https://github.com/Adi0224/Drone2.git
cd Drone2

# Setup
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r backend/requirements.txt
python scripts/download_model.py   # first time only (~12 MB)

# Run
PHANTOM_HTTP_ONLY=1 python backend/main.py
```

You should see: `Starting server on http://localhost:8000`. Then:

1. Open **http://localhost:8000** — tactical map and webcam feed should load.
2. Open **http://localhost:8000/live** — live stream with YOLO boxes.

If both load, the app is working. Stop the server with `Ctrl+C`.

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
| `models/`  | Put `yolov8_det.onnx` here (run `download_model_qualcomm.py` or `download_model.py`) |
| `scripts/` | `download_model_qualcomm.py` (Qualcomm Hub v8 det), `download_model.py`, `generate_ssl_certs.sh` |
| `run.sh`   | Run the app (installs deps and model if needed)   |

---

## Commands reference

| What              | Command |
|-------------------|--------|
| Install deps      | `pip install -r backend/requirements.txt` |
| Download YOLO (Qualcomm Hub v8) | `python scripts/download_model_qualcomm.py` |
| Download YOLO (Ultralytics)     | `python scripts/download_model.py` |
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
