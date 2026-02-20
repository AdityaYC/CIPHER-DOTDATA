#!/usr/bin/env bash
# PHANTOM CODE — Run the app (Mac or Windows WSL/Linux)
set -e
cd "$(dirname "$0")"

# Use venv if present
if [ -d ".venv" ]; then
  source .venv/bin/activate
fi

# Install deps if needed
if ! python -c "import fastapi" 2>/dev/null; then
  echo "Installing dependencies..."
  pip install -r backend/requirements.txt -q
fi

# Ensure YOLO model exists
if [ ! -f "models/yolov8_det.onnx" ]; then
  echo "Downloading YOLO model..."
  python scripts/download_model.py
fi

echo "Starting PHANTOM CODE (HTTP — use http://localhost:8000 in browser)..."
echo "  Tactical map:  http://localhost:8000"
echo "  Live stream:   http://localhost:8000/live"
PHANTOM_HTTP_ONLY=1 python backend/main.py
