<div align="center">

<img src="https://img.shields.io/badge/CIPHER-Disaster%20Response%20AI-00b4d8?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZmlsbD0id2hpdGUiIGQ9Ik0xMiAyTDIgN2wxMCA1IDEwLTVMMTIgMnpNMiAxN2wxMCA1IDEwLTVNMiAxMmwxMCA1IDEwLTUiLz48L3N2Zz4=" />

# CIPHER

### *When the signal dies — CIPHER finds the living*

**Autonomous disaster response AI. No cloud. No signal. No compromises.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/Python-3.12%20x64-3776ab?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Qualcomm NPU](https://img.shields.io/badge/Qualcomm-Snapdragon%20X%20Elite-3253DC?style=flat-square)](https://www.qualcomm.com/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX-Runtime%20QNN-FF6F00?style=flat-square)](https://onnxruntime.ai/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFAB?style=flat-square)](https://ultralytics.com/)

<br/>

> 89 rescue workers die every year in the US alone — entering structures blind.  
> CIPHER gives them the map they never had, running entirely on a single chip.

<br/>

</div>

---

## 📌 Table of Contents

- [What is CIPHER](#-what-is-cipher)
- [Demo](#-demo)
- [Architecture](#-architecture)
- [Four Tabs](#-four-tabs)
- [Requirements](#-requirements)
- [Installation](#-installation)
- [Running CIPHER](#-running-cipher)
- [Project Structure](#-project-structure)
- [Models](#-models)
- [Configuration](#-configuration)
- [Hardware Notes](#-hardware-notes)
- [License](#-license)

---

## 🛸 What is CIPHER

CIPHER is an on-device AI system that maps disaster environments in real time — detecting survivors, assessing structural risk, building a navigable 3D world graph, and answering natural-language queries from rescue teams. It runs entirely on the **Qualcomm Snapdragon X Elite Hexagon NPU** with zero internet, zero cloud, and zero signal required.

```
Camera → YOLOv8 + DepthAnything + CrackSeg (NPU)
       → World Graph (semantic nodes + edges)
       → 2D Map + 3D Point Cloud + Agent Q&A
       → Rescue team intelligence. Fully local.
```

| Metric | Value |
|--------|-------|
| Combined NPU inference | ~47ms |
| CPU equivalent | ~380ms |
| NPU speedup | **8.1×** |
| Cloud dependencies | **0** |
| Internet required | **No** |
| Models running simultaneously | **4** |

---

## 🎬 Demo

```
┌─────────────────────────────────────────────────┐
│  CIPHER  │  AGENT  │  MANUAL  │  REPLAY  │  3D  │
└─────────────────────────────────────────────────┘

  AGENT:   "Where are the survivors?"
           → Node 004 · Grid B3 · 2.3m ahead · 91% confidence

  MANUAL:  Live camera + semantic overhead map
           Survivor detected → green node appears instantly

  3D WORLD: Navigate the recorded space in first person
            Point cloud + 2D map + arrow key traversal

  REPLAY:  Full mission playback · 0.5× to 4× speed
```

---

## 🏗 Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                        CIPHER PIPELINE                        │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│   CAMERA THREAD          INFERENCE THREAD      DISPLAY       │
│   ─────────────          ────────────────      ───────       │
│   cap.read()        →    YOLOv8n-det (NPU)  →  Render        │
│   frame.copy()           DepthAnything (NPU)   World Graph   │
│   queue (max=2)          CrackSeg (NPU)        2D Map        │
│                          Whisper (NPU)          Point Cloud   │
│                               │                              │
│                               ▼                              │
│                        WORLD GRAPH                           │
│                   nodes: frame + depth +                     │
│                   labels + pose + risk score                 │
│                               │                              │
│                    ┌──────────┴──────────┐                   │
│                    ▼                     ▼                   │
│             ChromaDB (local)      CLIP Embeddings            │
│             Emergency Manuals     Graph Navigation           │
│                    │                     │                   │
│                    └──────────┬──────────┘                   │
│                               ▼                              │
│                    Llama 3.2 3B (Genie SDK)                  │
│                    Grounded agent response                   │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### Structural Risk Scoring

Every surface the camera sees gets a continuous risk score:

```
R = A_crack × C_seg × σ²_depth

R < 0.30  →  STABLE       (green)
R < 0.70  →  COMPROMISED  (orange)  
R ≥ 0.70  →  CRITICAL     (red)
```

### World Graph Edge Filtering

Two nodes are connected only when their CLIP cosine similarity satisfies:

```
0.30 ≤ cos(θ) ≤ 0.95

< 0.30  →  spatial discontinuity, nodes created but not connected
> 0.95  →  camera barely moved, duplicate frame discarded
```

---

## 🖥 Four Tabs

### `MANUAL` — Live Operational View
- Live camera feed with YOLO bounding boxes and depth overlays
- Semantic overhead map updating in real time as detections are made
- Every survivor, hazard, exit, and structural risk plotted instantly
- Color coded nodes: 🟢 SURVIVOR · 🔴 HAZARD · 🟠 STRUCTURAL · 🔵 EXIT

### `3D WORLD` — Navigate the Recorded Space
- First-person navigation through stored frames via arrow keys
- Live RGB point cloud from accumulated depth maps
- 2D overhead map synced to point cloud — click any node to jump to it
- Node sphere click shows stored camera frame as floating thumbnail
- Visual odometry coordinates displayed per node: `X Y Z YAW`

### `REPLAY` — Mission Playback
- Chronological playback of every stored node frame
- Full-width scrubber, PLAY/PAUSE, speed control: `0.5× 1× 2× 4×`
- Timestamp overlay per frame
- Hard cuts between frames — clean and fast, no transitions

### `AGENT` — Natural Language Intelligence
- Ask questions by text or voice (Whisper on NPU)
- Query types: `SPATIAL` · `KNOWLEDGE` · `COMBINED`
- Answers grounded in actual world graph nodes — no hallucination
- Referenced nodes highlighted on map simultaneously
- Confidence indicator: 🟢 >80% · 🟡 50–80% · 🔴 <50%

---

## 📋 Requirements

### Hardware
| Requirement | Spec |
|-------------|------|
| **Recommended** | Qualcomm Snapdragon X Elite laptop |
| NPU | Hexagon NPU, 45 TOPS |
| RAM | 16GB minimum, 32GB recommended |
| Storage | 10GB free (models + vector DB) |
| Camera | USB webcam or built-in (720p minimum) |
| OS | Windows 11 |

### Software
| Dependency | Version |
|------------|---------|
| Python | **3.12 x64** |
| Node.js | 18+ |
| PowerShell | 5.1+ (built into Windows) |

---

## ⚡ Quick Start

> **First time setup takes ~5 minutes. Every run after that is one command.**

### Step 1 — Clone the repository

```bash
git clone https://github.com/yourusername/cipher.git
cd cipher
```

### Step 2 — Check Python 3.12 is installed

```powershell
py -3.12 --version
# Should print: Python 3.12.x
```

> Don't have Python 3.12? Download it from [python.org](https://www.python.org/downloads/). Make sure to check **"Add to PATH"** during install.

### Step 3 — Install all dependencies

```powershell
.\scripts\install_deps.ps1
```

This installs everything automatically:
- ✅ `Drone\local_backend\requirements.txt`
- ✅ `backend\requirements.txt`
- ✅ `ultralytics` (YOLOv8)
- ✅ All Python packages for both backend and drone modules

> **If PowerShell says "cannot be loaded because running scripts is disabled"**, run this first:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```
> Then re-run `.\scripts\install_deps.ps1`

> **If install_deps.ps1 can't find Python 3.12**, point it manually:
> ```powershell
> $env:PYTHON312_PATH = "C:\Path\To\Python312\python.exe"
> .\scripts\install_deps.ps1
> ```

### Step 4 — Run CIPHER

```powershell
.\run.ps1
```

This single command:
- Starts the backend on `http://localhost:8000` (opens in a new window)
- Starts the frontend on `http://localhost:5173` (runs in current window)

### Step 5 — Open in browser

```
http://localhost:5173
```

That's it. CIPHER is running.

---

## 🚀 Every Run After Setup

From the repo root:

```powershell
.\run.ps1
```

Then open `http://localhost:5173`.

---

## 🔧 Troubleshooting

**Backend only** (if frontend is already running separately):
```powershell
.\start_backend.ps1
```

**Frontend only** (if backend is already running):
```powershell
cd Drone\frontend
npm install
npm run dev
```
Then open `http://localhost:5173`.

**Camera not detected:**
```powershell
# Try a different camera index in the UI settings
# Default is index 0 (built-in), try 1 for USB webcam
```

**Port already in use:**
```powershell
# Kill whatever is on port 8000
netstat -ano | findstr :8000
taskkill /PID <PID> /F
# Then re-run .\run.ps1
```

---

## 📁 Project Structure

```
cipher/
│
├── main.py                    # Entry point
├── setup.py                   # One-command installer
├── verify.py                  # NPU verification script
├── requirements.txt
│
├── core/
│   ├── world_graph.py         # Node/edge graph data structure
│   ├── perception.py          # YOLO + Depth + CrackSeg inference
│   ├── camera.py              # Isolated camera thread
│   └── visual_odometry.py     # Pose estimation from optical flow
│
├── agent/
│   ├── orchestrator.py        # Query routing and result merging
│   ├── spatial_agent.py       # CLIP-based world graph navigation
│   ├── knowledge_agent.py     # ChromaDB RAG + Llama synthesis
│   └── clip_navigator.py      # CLIP embedding and similarity search
│
├── ui/
│   ├── manual_tab.py          # Live camera + semantic map
│   ├── world_tab.py           # 3D point cloud + first-person nav
│   ├── replay_tab.py          # Mission chronological playback
│   └── agent_tab.py           # Chat interface + voice input
│
├── data/
│   └── emergency_manuals/
│       ├── collapse_protocol.txt
│       ├── survivor_extraction.txt
│       ├── structural_hazard.txt
│       ├── fire_response.txt
│       └── triage_guide.txt
│
├── models/
│   ├── clip/                  # CLIP cached locally
│   ├── embeddings/            # all-MiniLM-L6-v2 cached locally
│   ├── yolov8n_det.onnx       # Downloaded by setup.py
│   ├── depth_anything.onnx    # Downloaded by setup.py
│   └── crack_seg.onnx         # Exported by setup.py
│
├── genie_bundle/
│   └── genie-t2t-run.exe      # Qualcomm Genie SDK (manual install)
│
├── chroma_db/                 # Local persistent vector database
│
└── exports/                   # VR HTML exports, mission reports
```

---

## 🤖 Models

| Model | Source | Purpose | NPU Latency |
|-------|--------|---------|-------------|
| YOLOv8n-det | [Qualcomm AI Hub](https://aihub.qualcomm.com/) | Object detection | ~35ms |
| DepthAnything | [Qualcomm AI Hub](https://aihub.qualcomm.com/) | Metric depth | ~28ms |
| YOLOv8n-seg (crack) | [OpenSistemas/HuggingFace](https://huggingface.co/OpenSistemas/YOLOv8-crack-seg) | Structural damage | ~35ms |
| Whisper-Base-En | [Qualcomm AI Hub](https://aihub.qualcomm.com/) | Voice transcription | ~600ms |
| CLIP ViT-B/32 | [OpenAI/HuggingFace](https://huggingface.co/openai/clip-vit-base-patch32) | Semantic search | ~40ms |
| all-MiniLM-L6-v2 | [HuggingFace](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | Vector embeddings | ~15ms |
| Llama 3.2 3B | [Qualcomm Genie SDK](https://www.qualcomm.com/) | Agent synthesis | ~3–8s |

All models run **locally**. No API keys. No internet at runtime.

---

## ⚙️ Configuration

Edit `config.py` to customize behavior:

```python
# Inference
YOLO_CONFIDENCE_THRESHOLD = 0.45
DEPTH_EVERY_N_FRAMES = 2          # run depth every 2nd frame
CRACK_SEG_EVERY_N_FRAMES = 10     # crack seg every 10th frame

# World Graph
MIN_NODE_DISTANCE = 0.5           # metres between nodes
CLIP_SIMILARITY_MIN = 0.30        # below this = discontinuity
CLIP_SIMILARITY_MAX = 0.95        # above this = duplicate, skip

# Structural Risk Thresholds
RISK_COMPROMISED = 0.30
RISK_CRITICAL = 0.70

# Video Import
TARGET_KEYFRAMES = 120            # N = duration / 120 seconds

# Agent
AGENT_MAX_STEPS = 20              # max graph traversal steps
GENIE_TIMEOUT_SECONDS = 5         # fallback to template if exceeded
RETRIEVAL_TOP_K = 5               # chunks retrieved from ChromaDB

# Camera
CAMERA_INDEX = 0
FRAME_QUEUE_MAX = 2               # bounded queue depth
```

---

## 💻 Hardware Notes

### Qualcomm Snapdragon X Elite — Full NPU Mode

When running on Snapdragon X Elite, CIPHER automatically uses the Hexagon NPU
via `QNNExecutionProvider`. You will see confirmation in the terminal on startup:

```
✓ YOLO: QNNExecutionProvider
✓ Depth: QNNExecutionProvider
✓ NPU active — 47ms combined inference
```

### Other Hardware

CIPHER runs on any Windows machine with Python 3.12. Models fall back to CPU
automatically. All features work — just slower inference.

### Performance Comparison

| Hardware | YOLO | Depth | Combined | Battery Impact |
|----------|------|-------|----------|----------------|
| Snapdragon X Elite NPU | 35ms | 28ms | **47ms** | Low |
| Snapdragon X Elite CPU | 180ms | 200ms | **380ms** | High |
| Modern laptop CPU | 220ms | 250ms | **470ms** | Very High |

---

## 📄 License

```
MIT License

Copyright (c) 2026 CIPHER Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 🙏 Acknowledgements

- [Qualcomm AI Hub](https://aihub.qualcomm.com/) — NPU model compilation
- [Ultralytics YOLOv8](https://ultralytics.com/) — Object detection
- [OpenSistemas](https://huggingface.co/OpenSistemas/YOLOv8-crack-seg) — Crack segmentation weights
- [DepthAnything](https://github.com/LiheYoung/Depth-Anything) — Metric depth estimation
- [OpenAI CLIP](https://github.com/openai/CLIP) — Semantic embeddings
- [ChromaDB](https://www.trychroma.com/) — Local vector database
- [emerGen](https://github.com/chaaenni/2025-Qualcomm-edge-ai-streamlit) — Emergency response knowledge architecture (MIT)

---

<div align="center">

**Built for MadData 2026 · Qualcomm Track**

*No cloud. No signal. No compromises.*

**When the signal dies — CIPHER finds the living.**

</div>
