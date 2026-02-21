# Cipher

On-device AI tactical system: live webcam, YOLO object detection, tactical map, and agent exploration. Runs on a single laptop with **zero cloud dependency**.

---

## How to run

### Option 1: Run everything (backend + frontend)

From the **repo root** in PowerShell:

```powershell
.\run_drone_full.ps1
```

- A **backend** window opens on **http://localhost:8000**
- This terminal starts the **frontend** (Vite). Note the URL (e.g. **http://localhost:5173** or **http://localhost:5174**)
- **Open that URL in your browser** → go to **Manual** → click **START AI** for webcam + YOLO

---

### Option 2: Run backend and frontend separately

**Terminal 1 — Backend (from repo root):**

```powershell
cd "C:\Users\hackathon user\Drone2"
$env:PYTHONPATH = (Get-Location).Path
.\start_backend.ps1
```

Or manually:

```powershell
cd "C:\Users\hackathon user\Drone2"
$env:PYTHONPATH = (Get-Location).Path
py -3.12 -m uvicorn Drone.local_backend.app:app --host 0.0.0.0 --port 8000
```

Backend runs at **http://localhost:8000**.

**Terminal 2 — Frontend (from repo root):**

```powershell
cd "C:\Users\hackathon user\Drone2\Drone\frontend"
npm run dev
```

Frontend runs at **http://localhost:5173** (or 5174 if 5173 is in use). Open that URL in your browser.

---

## First-time setup

From the **repo root**:

```powershell
# Backend: Python deps (use your Python 3.12 path or py -3.12)
pip install -r Drone\local_backend\requirements.txt
py -m pip install ultralytics

# Frontend: Node deps (once)
cd Drone\frontend
npm install
cd ../..
```

**Requirements:** Python 3.12 (install from [python.org](https://www.python.org/downloads/), not the Microsoft Store), Node.js, webcam.

---

## What’s in the app

| Page     | Description |
|----------|-------------|
| **Landing** | Cipher home. |
| **Manual**  | Live webcam, YOLO detections, tactical map. Click **START AI**. |
| **Agent**   | Type a question → tactical answer + agent exploration. |
| **Replay**  | Trajectory replay. |

- **Backend:** http://localhost:8000 (API, webcam, YOLO, Agent query)
- **Frontend:** http://localhost:5173 or 5174 (Cipher UI)

---

## Troubleshooting

| Problem | Fix |
|--------|-----|
| "Python was not found" | Install Python 3.12 from python.org. Disable the Store alias in Settings → App execution aliases. |
| "Backend not running" | Start backend first: `.\start_backend.ps1` or run the `py -3.12 -m uvicorn ...` command above. |
| Port 5173 in use | Vite will use 5174 or 5175 — use the URL shown in the terminal. |
| YOLO not loading | Run `py -m pip install ultralytics`. The first run downloads `yolov8n.pt`. |

---

## Project layout

| Path                  | Description |
|-----------------------|-------------|
| `Drone/frontend`      | Cipher React UI (Vite). |
| `Drone/local_backend` | Backend (FastAPI, webcam, YOLO, Agent). |
| `run_drone_full.ps1`  | Start backend + frontend. |
| `start_backend.ps1`   | Start backend only. |
