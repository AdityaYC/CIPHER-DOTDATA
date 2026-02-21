# Cipher

On-device AI tactical system: live webcam, YOLO object detection, tactical map, and agent exploration. Runs on a single laptop with **zero cloud dependency**.

---

## How to start

**Windows (PowerShell) — from the repo root:**

```powershell
.\run_drone_full.ps1
```

- A **backend** window opens (port 8000).
- This terminal starts the **Cipher frontend** (Vite). If port 5173 is in use, Vite may use 5174 or 5175 — check the terminal for the URL.
- **Open in your browser** the URL shown (e.g. **http://localhost:5173** or **http://localhost:5174**).
- Go to **Manual** → click **START AI** for live webcam and YOLO detections.

**If you see “Backend not running” or connection errors:**

1. Make sure the **backend window** is open and shows no Python errors.
2. Or start only the backend in a separate terminal:
   ```powershell
   .\start_backend.ps1
   ```
3. Refresh the Manual page.

**Requirements:** Python 3.12 (or 3.10+), Node.js (for the frontend). Install YOLO once: `py -m pip install ultralytics`.

---

## First-time setup (after clone)

From the **repo root**:

```powershell
# Optional: create venv
py -m venv .venv
.\.venv\Scripts\Activate.ps1

# Backend deps
pip install -r Drone\local_backend\requirements.txt
py -m pip install ultralytics

# Frontend deps (once)
cd Drone\frontend
npm install
cd ../..
```

Then run `.\run_drone_full.ps1` as above.

---

## What’s in the app

| Page    | Description |
|---------|-------------|
| **Landing** | Cipher home; link to launch. |
| **Manual**  | Live webcam, YOLO boxes, tactical map, mission selector. Click **START AI** to enable feed and detections. |
| **Agent**   | Tactical query and agent exploration. |
| **Replay**  | Trajectory replay. |

Backend serves the API (camera, YOLO, status, detections) on **http://localhost:8000**. The Cipher UI runs on the port Vite prints (usually 5173).

---

## Project layout

| Path                 | Description |
|----------------------|-------------|
| `Drone/frontend`     | Cipher React UI (Vite). |
| `Drone/local_backend`| Backend used by `run_drone_full.ps1` (FastAPI, webcam, YOLO). |
| `backend/`           | Optional alternate backend (root). |
| `run_drone_full.ps1` | Start backend + Cipher frontend. |
| `start_backend.ps1`  | Start only the backend (when frontend is already running). |

---

## Commands

| Action        | Command |
|---------------|--------|
| Run app       | `.\run_drone_full.ps1` |
| Backend only  | `.\start_backend.ps1` |
| Install YOLO  | `py -m pip install ultralytics` |
| Backend deps  | `pip install -r Drone\local_backend\requirements.txt` |

---

## API (backend)

- `GET /health` — Backend health.
- `GET /api/status` — Feed status, YOLO latency.
- `GET /api/feed/{drone_id}/processed` — Live frame with YOLO overlay.
- `GET /live_detections` — SSE stream of detections (used when you click START AI).

---

## Requirements

- **Windows:** Python 3.12 (or 3.10/3.11/3.13), Node.js, webcam. `ultralytics` for YOLO.
- **Mac/Linux:** Same; built-in webcam and CPU inference supported.
