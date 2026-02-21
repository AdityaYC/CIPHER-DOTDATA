# Local Backend for Drone Vision System (No Modal Required)

This is a local FastAPI backend that replaces Modal for running the drone navigation project locally.

## What You Need

### 1. Models

**Llama Vision** (11B or 90B):
```bash
# You need to get access from Meta/Hugging Face
# https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct
# or
# https://huggingface.co/meta-llama/Llama-3.2-90B-Vision-Instruct

# Login to Hugging Face
huggingface-cli login

# Model will auto-download on first run
```

**YOLO** (auto-downloads):
```bash
# YOLOv8 nano will download automatically from Ultralytics
# Or download from Qualcomm AI Hub for optimized version
```

### 2. Hardware Requirements

**Minimum:**
- 16GB RAM
- 8GB GPU (NVIDIA with CUDA) or Apple Silicon M1/M2/M3
- 50GB disk space (for model weights)

**Recommended:**
- 32GB RAM
- 16GB+ GPU (NVIDIA RTX 3090/4090 or A100)
- Apple M3 Max/Ultra

### 3. Installation

```bash
cd local_backend

# Install dependencies
pip install -r requirements.txt

# For NVIDIA GPU (Linux/Windows):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# For Apple Silicon (Mac):
# torch is already optimized for Metal
```

### 4. Run the Backend

```bash
# Start the server
uvicorn app:app --host 0.0.0.0 --port 8000 --reload

# Or using Python directly
python app.py
```

The backend will be available at: `http://localhost:8000`

### 5. Update Frontend Config

Edit `frontend/src/config.ts`:

```typescript
export const API_BASE_URL = "http://localhost:8000";
export const AGENT_STREAM_URL = "http://localhost:8000";
export const AGENT_API_URL = "http://localhost:8000";
```

### 6. Test the Backend

```bash
# Check health
curl http://localhost:8000/health

# Test image endpoint
curl "http://localhost:8000/getImage?x=0&y=0&z=0&yaw=0"
```

## API Endpoints

### GET /getImage
Returns image and navigation metadata for a position.

**Parameters:**
- `x`, `y`, `z`: Position coordinates
- `yaw`: Facing direction (0, 90, 180, 270)

**Response:**
```json
{
  "image": "base64_jpeg_string",
  "x": 0.0,
  "y": 0.0,
  "z": 0.0,
  "yaw": 0.0,
  "filename": "frame_0000.jpg",
  "allowed": {
    "forward": true,
    "backward": true,
    "left": false,
    "right": true,
    "turnLeft": true,
    "turnRight": true
  }
}
```

### POST /stream_agents
Streams agent exploration events via Server-Sent Events (SSE).

**Request Body:**
```json
{
  "query": "find the fire extinguisher",
  "start_x": 0.0,
  "start_y": 0.0,
  "start_z": 0.0,
  "start_yaw": 0.0,
  "num_agents": 2
}
```

**Response:** SSE stream of events (agent_started, agent_step, agent_found, agent_done, session_complete)

## Performance Tips

### For Faster Inference:

1. **Use 4-bit quantization:**
```python
# In app.py ModelManager.load_llama_vision():
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

self.llama_vision = MllamaForConditionalGeneration.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    device_map="auto",
)
```

2. **Use smaller YOLO model:**
```python
# yolov8n.pt (nano) - fastest
# yolov8s.pt (small) - balanced
# yolov8m.pt (medium) - more accurate
```

3. **Reduce image resolution:**
```python
# In agent_runner.py, resize images before inference
image = image.resize((512, 512), Image.LANCZOS)
```

## Troubleshooting

### Out of Memory
- Use Llama-3.2-11B instead of 90B
- Enable 4-bit quantization
- Reduce batch size
- Close other applications

### Slow Inference
- Use GPU instead of CPU
- Enable Flash Attention 2
- Use smaller models
- Reduce image resolution

### Model Not Found
- Run `huggingface-cli login` first
- Check you have access to Llama 3.2 Vision models
- Verify model ID is correct

## Architecture

```
Frontend (React/Vite)
    ↓ HTTP/SSE
Local Backend (FastAPI)
    ↓
┌─────────────────────────┐
│  Image Database         │ ← Loads trajectory data
│  (frames + metadata)    │
└─────────────────────────┘
    ↓
┌─────────────────────────┐
│  Agent Runner           │
│  - Llama Vision (VLM)   │ ← Navigation reasoning
│  - YOLO (Detection)     │ ← Object detection
└─────────────────────────┘
```

## Next Steps

1. Install dependencies
2. Get Llama Vision access
3. Start backend server
4. Update frontend config
5. Run frontend: `cd frontend && npm run dev`
6. Open browser to `http://localhost:5173`

Enjoy exploring! 🚁
