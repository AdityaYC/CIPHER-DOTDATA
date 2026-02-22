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
cd "C:\Users\hackathon user\cipher3\CIPHER"
$env:PYTHONPATH = (Get-Location).Path
.\start_backend.ps1
```

Or manually:

```powershell
cd "C:\Users\hackathon user\cipher3\CIPHER"
$env:PYTHONPATH = (Get-Location).Path
py -3.12 -m uvicorn Drone.local_backend.app:app --host 0.0.0.0 --port 8000
```

Backend runs at **http://localhost:8000**.

**Terminal 2 — Frontend (from repo root):**

```powershell
cd "C:\Users\hackathon user\cipher3\CIPHER\Drone\frontend"
npm run dev
```

Frontend runs at **http://localhost:5173** (or 5174 if 5173 is in use). Open that URL in your browser.

---

## First-time setup (download dependencies)

From the **repo root** in PowerShell:

**Option A — One-shot (recommended): install deps, YOLO, and depth model**

```powershell
.\scripts\setup_yolo_and_depth.ps1
```

Then install the frontend once:

```powershell
cd Drone\frontend
npm install
cd ../..
```

**Option B — Step by step**

```powershell
# Backend: Python deps (use your Python 3.12 path or py -3.12)
.\scripts\install_deps.ps1
# YOLO model (downloads yolov8n.pt / ONNX)
.\scripts\download_model.ps1
# Optional: depth model for minimap (caches Depth-Anything-V2)
py -3.12 scripts\download_depth_model.py
# Frontend (once)
cd Drone\frontend
npm install
cd ../..
```

Or step by step:

```powershell
# Backend: Python deps (use python if py is not on PATH)
python -m pip install -r Drone\local_backend\requirements.txt
python -m pip install ultralytics

# Frontend: Node deps (once)
cd Drone\frontend
npm install
cd ../..
```

**If pip install fails with "No such file or directory" / long path (e.g. when installing torch/ultralytics):**

1. Enable Windows Long Paths (run **PowerShell as Administrator**):
   ```powershell
   New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name LongPathsEnabled -Value 1 -PropertyType DWORD -Force
   ```
2. **Reboot**, then run the install commands again.

**Alternative (no torch):** Use core deps + ONNX YOLO so you don’t need Long Paths:

```powershell
python -m pip install -r Drone\local_backend\requirements-core.txt
python scripts\download_yolo_onnx.py
```

If the ONNX download fails (404), enable Long Paths, then run `python -m pip install ultralytics` and `python scripts\download_model.py` to build the ONNX model.

**Requirements:** Python 3.12 (install from [python.org](https://www.python.org/downloads/), not the Microsoft Store), Node.js, webcam.

**Agent on Qualcomm (Llama 3.2 3B on-device):** Run `python scripts/setup_llama_qualcomm.py` from repo root (needs [Qualcomm AI Hub](https://app.aihub.qualcomm.com) API token). Then add Genie runtime to `genie_bundle/`. See [genie_bundle/README.md](genie_bundle/README.md).

---

## What’s in the app

| Page     | Description |
|----------|-------------|
| **Landing** | Cipher home. |
| **Manual**  | Live webcam, YOLO detections, depth estimates on tactical map (minimap). Click **START AI**. |
| **Agent**   | Type a question → tactical answer + agent exploration. |
| **Replay**  | Trajectory replay. |

- **Backend:** http://localhost:8000 (API, webcam, YOLO, Agent query)
- **Frontend:** http://localhost:5173 or 5174 (Cipher UI)

---

## Troubleshooting

| Problem | Fix |
|--------|-----|
| "Running scripts is disabled" | Run once: `powershell -ExecutionPolicy Bypass -File .\run_drone_full.ps1` or use `.\run_with_bypass.cmd`. Or allow scripts: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`. |
| "Python was not found" | Install Python 3.12 from python.org. Disable the Store alias in Settings → App execution aliases. |
| "Backend not running" | Start backend first: `.\start_backend.ps1` or run the `py -3.12 -m uvicorn ...` command above. |
| Port 5173 in use | Vite will use 5174 or 5175 — use the URL shown in the terminal. |
| YOLO not loading | Run `py -m pip install ultralytics`. The first run downloads `yolov8n.pt`. |
| Agent answers are generic or "No LLM" | Install Ollama, run `ollama run llama3.2`, then restart the backend. See **Agent tab — Ollama** above. |
| Depth on minimap | Optional: run `python Drone/local_backend/get_depth_model.py` (from repo root) to export Qualcomm AI Hub Depth Anything V2 ONNX to `models/`. Otherwise the backend may use HuggingFace Depth Anything (requires `transformers`). |

---

## Project layout

| Path                  | Description |
|-----------------------|-------------|
| `Drone/frontend`      | Cipher React UI (Vite). |
| `Drone/local_backend` | Backend (FastAPI, webcam, YOLO, Agent). |
| `run_drone_full.ps1`  | Start backend + frontend. |
| `start_backend.ps1`   | Start backend only. |
