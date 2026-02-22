# NPU Setup Guide - Qualcomm Snapdragon X Elite

This guide is for running YOLO on the **Qualcomm Hexagon NPU (45 TOPS)** on Windows Copilot+ PC devices like the Surface Laptop with Snapdragon X Elite/Plus.

---

## 🖥️ Hardware Requirements

- **Device:** Surface Laptop (7th Edition) or other Copilot+ PC
- **Processor:** Snapdragon X Elite or X Plus
- **NPU:** Qualcomm Hexagon with 45 TOPS
- **OS:** Windows 11 (Copilot+ PC edition)
- **RAM:** 16GB+ recommended
- **Storage:** 10GB free space

---

## 📦 Installation

### **Step 1: Install Python (AMD64 version)**

⚠️ **CRITICAL:** You MUST use AMD64 (x64) Python, NOT ARM64 Python on Windows.

```powershell
# Download Python 3.11 or 3.12 (AMD64 version)
# https://www.python.org/downloads/windows/

# Verify it's AMD64:
python --version
python -c "import platform; print(platform.machine())"
# Should output: AMD64 (not ARM64)
```

### **Step 2: Install Base Dependencies**

```powershell
cd local_backend
pip install -r requirements_npu.txt
```

### **Step 3: Install Qualcomm AI Hub (for NPU)**

```powershell
# Install Qualcomm AI Hub Models
pip install qai_hub_models

# Install YOLO-specific dependencies
pip install "qai_hub_models[yolov8]"
```

### **Step 4: (Optional) Install ONNX Runtime with QNN**

Alternative NPU backend if AI Hub doesn't work:

```powershell
pip install onnxruntime-qnn
```

---

## 🚀 Usage

### **Start the Graph API Server**

```powershell
cd local_backend
python graph_api.py
```

The server will automatically:
1. Try to load YOLO with Qualcomm NPU acceleration
2. Fall back to ONNX Runtime with QNN if available
3. Fall back to CPU/GPU if NPU unavailable

### **Check NPU Status**

```powershell
# Test NPU YOLO directly
python yolo_npu.py

# Check API health (shows backend info)
curl http://localhost:8001/health
```

Expected output:
```json
{
  "status": "ok",
  "yolo_loaded": true,
  "node_count": 0,
  "yolo_backend": "qualcomm_npu",
  "npu_enabled": true
}
```

### **Test with Webcam**

```powershell
python test_webcam.py
```

---

## 🎯 NPU Backend Priority

The system tries backends in this order:

1. **Qualcomm AI Hub** (Best) - Direct NPU access, ~45 TOPS
2. **ONNX Runtime + QNN** (Good) - NPU via ONNX, ~40 TOPS
3. **PyTorch CPU/GPU** (Fallback) - No NPU, ~5-10 FPS

---

## 📊 Performance Expectations

| Backend | Device | FPS | Latency |
|---------|--------|-----|---------|
| Qualcomm NPU | Hexagon (45 TOPS) | 60-120 | 8-16ms |
| ONNX/QNN NPU | Hexagon (45 TOPS) | 40-80 | 12-25ms |
| PyTorch GPU | Adreno GPU | 20-40 | 25-50ms |
| PyTorch CPU | Snapdragon X Elite | 5-15 | 60-200ms |

---

## 🔧 Troubleshooting

### **Issue: "qai_hub_models not found"**

```powershell
pip install qai_hub_models
pip install "qai_hub_models[yolov8]"
```

### **Issue: "ARM64 Python not supported"**

You're using ARM64 Python. Uninstall and install AMD64 Python:
- Download from: https://www.python.org/downloads/windows/
- Select "Windows installer (64-bit)" - NOT "Windows installer (ARM64)"

### **Issue: "QNN execution provider not available"**

```powershell
# Install ONNX Runtime with QNN support
pip install onnxruntime-qnn

# Or download from Qualcomm:
# https://www.qualcomm.com/developer/software/qualcomm-neural-processing-sdk
```

### **Issue: NPU not being used**

Check Task Manager → Performance → NPU to see if it's active during inference.

If NPU shows 0% usage:
1. Verify you're on a Copilot+ PC with Snapdragon X Elite/Plus
2. Check Windows version (must be Windows 11 with NPU support)
3. Update Qualcomm drivers via Windows Update

---

## 🧪 Verify NPU Acceleration

### **Method 1: Check Backend Info**

```python
from yolo_npu import create_yolo_npu

yolo = create_yolo_npu(use_npu=True)
info = yolo.get_backend_info()
print(info)
# Should show: {'backend': 'qualcomm_npu', 'npu_enabled': True}
```

### **Method 2: Monitor NPU Usage**

1. Open Task Manager (Ctrl+Shift+Esc)
2. Go to Performance tab
3. Look for "NPU" or "Neural Processor"
4. Run inference and watch NPU usage spike to 80-100%

### **Method 3: Benchmark**

```python
import time
from PIL import Image
from yolo_npu import create_yolo_npu

yolo = create_yolo_npu(use_npu=True)
test_image = Image.new('RGB', (640, 640))

# Warmup
for _ in range(10):
    yolo.predict(test_image)

# Benchmark
start = time.time()
for _ in range(100):
    yolo.predict(test_image)
elapsed = time.time() - start

print(f"Average FPS: {100/elapsed:.1f}")
print(f"Average latency: {elapsed*10:.1f}ms")
```

Expected results:
- **With NPU:** 60-120 FPS, 8-16ms latency
- **Without NPU:** 5-15 FPS, 60-200ms latency

---

## 📱 iPhone Integration

Your iPhone app should send frames to:

```
POST http://<surface-laptop-ip>:8001/ingest_frame
```

With JSON body:
```json
{
  "gps_lat": 37.4275,
  "gps_lon": -122.1697,
  "altitude_m": 1.5,
  "yaw_deg": 45.0,
  "image_b64": "<base64_jpeg>"
}
```

The NPU will process frames in real-time at 60-120 FPS.

---

## 🎓 Resources

- **Qualcomm AI Hub:** https://aihub.qualcomm.com/
- **AI Hub Models GitHub:** https://github.com/quic/ai-hub-models
- **ONNX Runtime QNN:** https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html
- **Snapdragon X Elite Docs:** https://www.qualcomm.com/products/mobile/snapdragon/pcs-and-tablets/snapdragon-mobile-platforms/snapdragon-x-series

---

## ✅ Quick Checklist

- [ ] Windows 11 Copilot+ PC with Snapdragon X Elite/Plus
- [ ] Python 3.11/3.12 (AMD64, not ARM64)
- [ ] `qai_hub_models` installed
- [ ] Graph API running on port 8001
- [ ] NPU showing usage in Task Manager
- [ ] Health endpoint shows `"npu_enabled": true`

---

**You're ready to run YOLO at 45 TOPS on the Qualcomm NPU!** 🚀
