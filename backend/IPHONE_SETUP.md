# iPhone Camera Live Feed Setup

Use your iPhone camera as the live feed for the drone vision system.

## 🎯 Quick Start

### **Step 1: Start the iPhone Stream Server**

```bash
cd local_backend
python3 iphone_stream.py
```

Server runs on: `http://0.0.0.0:8002`

### **Step 2: Connect Your iPhone**

You have 3 options:

---

## 📱 Option 1: Use IP Webcam App (Easiest)

### **Install App:**
- **iOS:** Download "IP Webcam" or "EpocCam" from App Store
- **Alternative:** "DroidCam" or "iVCam"

### **Setup:**
1. Open app on iPhone
2. Start camera server
3. Note the URL shown (e.g., `http://192.168.1.100:8080`)
4. Update frontend config to use this URL

### **Configure Frontend:**
```typescript
// frontend/src/config.ts
export const CAMERA_STREAM_URL = "http://192.168.1.100:8080/video";
```

---

## 📱 Option 2: Use Custom iOS App

### **Create Simple iOS App:**

```swift
// ViewController.swift
import UIKit
import AVFoundation

class CameraViewController: UIViewController {
    var captureSession: AVCaptureSession!
    let serverURL = "http://YOUR_LAPTOP_IP:8002/upload_frame"
    
    override func viewDidLoad() {
        super.viewDidLoad()
        setupCamera()
        startStreaming()
    }
    
    func setupCamera() {
        captureSession = AVCaptureSession()
        captureSession.sessionPreset = .medium
        
        guard let camera = AVCaptureDevice.default(for: .video),
              let input = try? AVCaptureDeviceInput(device: camera) else {
            return
        }
        
        captureSession.addInput(input)
        
        let output = AVCaptureVideoDataOutput()
        output.setSampleBufferDelegate(self, queue: DispatchQueue(label: "camera"))
        captureSession.addOutput(output)
        
        captureSession.startRunning()
    }
    
    func startStreaming() {
        // Capture and send frames every 100ms
    }
}

extension CameraViewController: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput, 
                      didOutput sampleBuffer: CMSampleBuffer, 
                      from connection: AVCaptureConnection) {
        guard let imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            return
        }
        
        let ciImage = CIImage(cvPixelBuffer: imageBuffer)
        let context = CIContext()
        guard let cgImage = context.createCGImage(ciImage, from: ciImage.extent) else {
            return
        }
        
        let uiImage = UIImage(cgImage: cgImage)
        guard let jpegData = uiImage.jpegData(compressionQuality: 0.7) else {
            return
        }
        
        let base64 = jpegData.base64EncodedString()
        
        // Send to server
        let json: [String: Any] = [
            "image_b64": base64,
            "timestamp": Date().timeIntervalSince1970
        ]
        
        guard let jsonData = try? JSONSerialization.data(withJSONObject: json),
              let url = URL(string: serverURL) else {
            return
        }
        
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = jsonData
        
        URLSession.shared.dataTask(with: request).resume()
    }
}
```

---

## 📱 Option 3: Use Python Script on iPhone (Pythonista)

If you have Pythonista app:

```python
import requests
import base64
import time
from PIL import Image
import io

SERVER_URL = "http://YOUR_LAPTOP_IP:8002/upload_frame"

def capture_and_send():
    # Use iPhone camera via Pythonista
    import photos
    
    while True:
        # Capture photo
        img = photos.capture_image()
        
        # Convert to JPEG
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=70)
        img_b64 = base64.b64encode(buffer.getvalue()).decode()
        
        # Send to server
        try:
            response = requests.post(SERVER_URL, json={
                "image_b64": img_b64,
                "timestamp": time.time()
            })
            print(f"Sent: {response.status_code}")
        except Exception as e:
            print(f"Error: {e}")
        
        time.sleep(0.1)  # 10 FPS

capture_and_send()
```

---

## 🌐 Frontend Integration

### **Update Frontend to Show iPhone Feed:**

```typescript
// frontend/src/components/iPhoneFeed.tsx
import { useEffect, useRef } from 'react';

export function IPhoneFeed() {
  const imgRef = useRef<HTMLImageElement>(null);
  
  useEffect(() => {
    // Option 1: Use MJPEG stream
    if (imgRef.current) {
      imgRef.current.src = "http://localhost:8002/stream";
    }
    
    // Option 2: Poll for latest frame
    const interval = setInterval(async () => {
      try {
        const response = await fetch("http://localhost:8002/latest_frame");
        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        if (imgRef.current) {
          imgRef.current.src = url;
        }
      } catch (e) {
        console.error("Failed to fetch frame:", e);
      }
    }, 100); // 10 FPS
    
    return () => clearInterval(interval);
  }, []);
  
  return (
    <div className="iphone-feed">
      <img ref={imgRef} alt="iPhone Camera" />
    </div>
  );
}
```

---

## 🔧 Network Setup

### **Find Your Laptop's IP Address:**

**On Mac:**
```bash
ifconfig | grep "inet " | grep -v 127.0.0.1
```

**On Windows:**
```bash
ipconfig
```

### **Make Sure iPhone and Laptop are on Same WiFi Network**

### **Test Connection:**
```bash
# On laptop
python3 iphone_stream.py

# On iPhone browser
# Visit: http://YOUR_LAPTOP_IP:8002/status
```

---

## 📊 API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/upload_frame` | POST | iPhone uploads frames |
| `/latest_frame` | GET | Get latest frame as JPEG |
| `/stream` | GET | MJPEG stream (video) |
| `/ws` | WebSocket | Real-time updates |
| `/status` | GET | Check streaming status |

---

## 🧪 Test iPhone Stream

### **1. Start Server:**
```bash
python3 iphone_stream.py
```

### **2. Test with curl (simulate iPhone):**
```bash
# Capture from Mac webcam and send
python3 -c "
import cv2
import base64
import requests
import time

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if ret:
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        img_b64 = base64.b64encode(buffer).decode()
        
        requests.post('http://localhost:8002/upload_frame', json={
            'image_b64': img_b64,
            'timestamp': time.time()
        })
        print('Frame sent')
    
    time.sleep(0.1)
"
```

### **3. View Stream in Browser:**
```
http://localhost:8002/stream
```

---

## 🎯 Integration with Drone Vision System

Once iPhone stream is working, integrate with main system:

```python
# In graph_api.py or app.py
import requests

# Get frame from iPhone stream
response = requests.get("http://localhost:8002/latest_frame")
image_bytes = response.content

# Process with YOLO
from PIL import Image
import io
image = Image.open(io.BytesIO(image_bytes))
detections = yolo_model.predict(image)

# Add to world graph
world_graph.add_node(...)
```

---

## 🚀 Ready to Use!

1. Start iPhone stream server: `python3 iphone_stream.py`
2. Connect iPhone using one of the 3 options above
3. View stream at: `http://localhost:8002/stream`
4. Frontend will automatically display iPhone camera feed

---

**Your iPhone is now the live camera feed for the drone vision system!** 📱🚁
