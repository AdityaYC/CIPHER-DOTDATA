# Complete Codebase Analysis for Combined Agent

## 1. What query system already exists?

- **Backend (Drone2):** `backend/query_agent.py` implements a **tactical RAG pipeline**:
  - `vector_query(question, top_k)` over ChromaDB (manuals + graph node texts).
  - `run_genie(prompt)` for on-device LLM (Genie SDK).
  - Returns `{"answer": str, "node_ids": List[str]}`; `node_ids` are doc IDs from vector DB (including `node_*` for graph nodes).
- **API:** `POST /api/voice_query` (in `Drone/local_backend/app.py`) accepts JSON `{ "text": "..." }` or multipart with `audio`; calls `query_agent(text, top_k=3, get_graph_callback=world_graph.get_graph)` and stores result in `phantom_state["agent_response"]`. Frontend can read last response via `GET /api/agent_response`.
- **No step-by-step spatial exploration over the world graph.** The Agent tab (`/agent`) runs two things in parallel: (1) **visual agent stream** via `POST /stream_agents` (trajectory DB + Llama Vision over pre-recorded frames from `image_db`), and (2) **tactical query** via `POST /api/voice_query`. The world graph (nodes with poses, detections, `image_b64`) is only used as context for vector DB and keyword fallback; there is no “move through graph nodes and look at each frame” agent.

---

## 2. Is CLIP or any embedding model already loaded?

- **CLIP:** **No.** Nothing in the codebase loads CLIP or `openai/clip-vit-base-patch32`.
- **Embeddings:** **Yes, but for text only.** `backend/vector_db.py` loads **sentence-transformers** `all-MiniLM-L6-v2` (cached at `PROJECT_ROOT/models/embeddings`) for embedding manual excerpts and graph node *text* descriptions. It is used for ChromaDB semantic search, not for image–text or image–image similarity.
- **Vision:** `Drone/local_backend/app.py` has a `ModelManager` that can load **Qwen2.5-VL-3B** via `mlx-vlm` for `infer_llama(image, prompt)` (used by `agent_runner.py` and `/analyze_frame`). That is a vision-language model over trajectory frames from `image_db`, not over world graph node frames. So: no CLIP; vision is VLM (Qwen) for a different pipeline.

---

## 3. Does Genie SDK integration already exist?

- **Yes.** `backend/genie_runner.py`:
  - `GENIE_BUNDLE = PROJECT_ROOT/genie_bundle`
  - `GENIE_EXE = "genie-t2t-run.exe"`
  - `run_genie(prompt, config_path=None)` runs subprocess with `-c` config and `-p` prompt, **5 second timeout** (`GENIE_TIMEOUT_SEC = 5`), returns response text or `""` on failure.
  - `is_available()` checks for exe and `genie_config.json`.
- Used by `query_agent.py`: if Genie is available and vector search returns context, it formats a prompt and calls `run_genie()`; otherwise template/keyword fallback.

---

## 4. Is ChromaDB already set up?

- **Yes.** `backend/vector_db.py`:
  - `CHROMA_DIR = os.path.join(PROJECT_ROOT, "chroma_db")` — **local persistent**, no cloud.
  - `PersistentClient(path=CHROMA_DIR)`, collection `"phantom_nodes_and_manuals"`.
  - `load_manuals_from_data_dir()` loads all `.txt` from `DATA_DIR` (= `PROJECT_ROOT/data`), not from `data/emergency_manuals/`.
  - `sync_graph_nodes(get_graph_callback)` embeds each graph node as text and upserts into Chroma.
  - No CLIP; embeddings are from sentence-transformers (text only).

---

## 5. Exact file paths of existing ONNX models

- **YOLO:**  
  - `backend/config.py`: `YOLO_ONNX_PATH = "models/yolov8_det.onnx"` (relative to project root).  
  - `Drone/local_backend/app.py`: `_phantom_model_path = _DRONE2_ROOT / "models" / "yolov8_det.onnx"`.  
  So: **`<repo_root>/models/yolov8_det.onnx`** (when backend runs with repo root as cwd or PYTHONPATH).
- **Whisper:**  
  - `backend/voice_input.py`: `WHISPER_ONNX_PATH = os.path.join(_PROJECT_ROOT, "models", "whisper_base.onnx")`.  
  So: **`<repo_root>/models/whisper_base.onnx`**.

---

## 6. How is the current AGENT button in the interactive world panel wired?

- **There is no AGENT button in the 3D Map tab.** The **3D Map** tab (`Map3DPage.tsx`, route `/3d-map`) has:
  - **MANUAL** — toggles manual mode (first-person frame, PREV/NEXT, arrow keys, full screen).
  - **IMPORT VIDEO** — uploads video, polls `/api/import_video/status`, on complete refetches graph and enters manual at first node.
  - **EXPORT VR** — POST `/api/export_vr`.
- **Agent** is a **separate tab** (`AgentPage.tsx`, route `/agent`). There, the user types a query; the frontend:
  1. Calls `POST /stream_agents` (SSE) to run the **trajectory-based** agent (Llama Vision + `image_db` from `Drone/local_backend/agent_runner.py`).
  2. In parallel calls `POST /api/voice_query` with the same text to get the **tactical** answer (vector DB + Genie) and displays answer + `node_ids`.
- So: the “interactive world panel” in the 3D Map currently has **MANUAL** and **IMPORT VIDEO** only. The prompt’s “AGENT button” in that panel does **not** exist yet; it is to be **added** as a new mode (e.g. press **A** or click **AGENT**) that uses the new combined agent (spatial + knowledge) over the **world graph** and shows the chat + path + highlights there.

---

## Summary table

| Component              | Exists? | Where / notes |
|------------------------|---------|----------------|
| Query system           | Yes     | `query_agent.py` + `/api/voice_query` (RAG + Genie + graph callback). No graph-step exploration. |
| CLIP                   | No      | Not loaded anywhere. |
| Embedding model        | Yes     | sentence-transformers `all-MiniLM-L6-v2` in `vector_db.py`, cache `models/embeddings/`. |
| Genie SDK              | Yes     | `genie_runner.py`, `genie_bundle/genie-t2t-run.exe`, 5s timeout. |
| ChromaDB               | Yes     | `chroma_db/`, collection `phantom_nodes_and_manuals`. |
| YOLO ONNX              | Yes     | `models/yolov8_det.onnx`. |
| Whisper ONNX           | Path set| `models/whisper_base.onnx` (decode not fully wired). |
| AGENT in 3D Map panel  | No      | Only MANUAL + IMPORT VIDEO + EXPORT VR; Agent is separate tab using trajectory + voice_query. |

---

## World graph structure (for agent implementation)

- **Module:** `Drone/local_backend/world_graph.py` (used by `app.py` via `_get_world_graph()`).
- **GraphNode:** `node_id`, `timestamp`, `gps_lat/lon`, `altitude_m`, `yaw_deg`, `detections` (list of Detection: class_name, confidence, bbox, distance_meters, category), `image_b64`, `source` ("live" | "imported"), `depth_b64`, `local_x`, `local_y`, `local_z` (for imported nodes).
- **Pose:** `get_pose_at_node(node_id)` → (x, y, z) in local meters; uses `local_x/y/z` when set, else GPS-derived.
- **Neighbors:** `get_neighbor_direction(node_id, "forward"|"back"|"left"|"right")` (spatial); `get_neighbor_by_order(node_id, "next"|"prev")` (visit order).
- **Path:** `get_path(start_id, end_id)` → list of node_ids (ordered segment).
- **3D:** `to_3d_pointcloud()` → list of (x, y, z, r, g, b).

All agent logic must use only these nodes and neighbors (no hallucinated positions).
