#!/bin/bash
# Setup script for local backend

set -e

echo "🚀 Setting up local backend for Drone Vision System..."

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $python_version"

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Check for GPU
if command -v nvidia-smi &> /dev/null; then
    echo "🎮 NVIDIA GPU detected"
    echo "   Installing CUDA-enabled PyTorch..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
elif [[ $(uname -m) == "arm64" ]] && [[ $(uname) == "Darwin" ]]; then
    echo "🍎 Apple Silicon detected"
    echo "   PyTorch with Metal support already installed"
else
    echo "💻 CPU-only mode"
fi

# Check for Hugging Face token
if [ -z "$HF_TOKEN" ]; then
    echo ""
    echo "⚠️  WARNING: HF_TOKEN not set"
    echo "   You need to login to Hugging Face to download Llama Vision:"
    echo "   Run: huggingface-cli login"
    echo ""
fi

# Download YOLO model
echo "📥 Downloading YOLO model..."
python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')" 2>/dev/null || echo "   (will download on first run)"

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Login to Hugging Face: huggingface-cli login"
echo "2. Start backend: uvicorn app:app --host 0.0.0.0 --port 8000"
echo "3. Update frontend config to use http://localhost:8000"
echo "4. Start frontend: cd ../frontend && npm run dev"
echo ""
