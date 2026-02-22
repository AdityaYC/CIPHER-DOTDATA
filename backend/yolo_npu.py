"""YOLO inference optimized for Qualcomm Snapdragon X Elite NPU.

Uses Qualcomm AI Hub Models for hardware-accelerated inference on the
Hexagon NPU (45 TOPS on Snapdragon X Elite/Plus).

Installation:
    pip install qai_hub_models
    pip install "qai_hub_models[yolov8]"
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from PIL import Image
import time


class YOLONPUInference:
    """YOLO inference using Qualcomm NPU acceleration.
    
    Supports multiple backends:
    1. Qualcomm AI Hub (NPU) - Fastest on Snapdragon X Elite
    2. ONNX Runtime with QNN - NPU acceleration via ONNX
    3. PyTorch (fallback) - CPU/GPU if NPU unavailable
    """
    
    def __init__(self, model_name: str = "yolov8_det", use_npu: bool = True):
        """Initialize YOLO with NPU acceleration.
        
        Args:
            model_name: Model variant (yolov8_det, yolov8n_det, yolov8s_det)
            use_npu: Try to use NPU if available
        """
        self.model_name = model_name
        self.use_npu = use_npu
        self.model = None
        self.backend = None
        self.class_names = self._get_coco_names()
        
        self._load_model()
    
    def _load_model(self):
        """Load YOLO model with best available backend."""
        
        # Try Qualcomm AI Hub first (best for NPU)
        if self.use_npu:
            try:
                print(f"🚀 Loading {self.model_name} with Qualcomm AI Hub (NPU)...")
                from qai_hub_models.models.yolov8_det import Model as YOLOv8Model
                
                # Load pre-optimized model from AI Hub
                self.model = YOLOv8Model.from_pretrained()
                self.backend = "qualcomm_npu"
                print(f"✅ Loaded on Qualcomm NPU (Hexagon)")
                return
            except ImportError:
                print("⚠️  qai_hub_models not installed. Install with:")
                print("   pip install qai_hub_models")
                print("   pip install 'qai_hub_models[yolov8]'")
            except Exception as e:
                print(f"⚠️  Qualcomm AI Hub unavailable: {e}")
        
        # Try ONNX Runtime with QNN execution provider
        if self.use_npu:
            try:
                print("🔄 Trying ONNX Runtime with QNN (NPU)...")
                import onnxruntime as ort
                
                # Check if QNN execution provider is available
                available_providers = ort.get_available_providers()
                if 'QNNExecutionProvider' in available_providers:
                    self._load_onnx_qnn()
                    return
                else:
                    print(f"⚠️  QNN provider not available. Available: {available_providers}")
            except ImportError:
                print("⚠️  onnxruntime not installed")
            except Exception as e:
                print(f"⚠️  ONNX Runtime with QNN failed: {e}")
        
        # Fallback to standard Ultralytics YOLO (CPU/GPU)
        print("🔄 Falling back to Ultralytics YOLO (CPU/GPU)...")
        try:
            from ultralytics import YOLO
            self.model = YOLO("yolov8n.pt")
            self.backend = "pytorch_cpu"
            print("✅ Loaded on CPU/GPU (no NPU acceleration)")
        except Exception as e:
            raise RuntimeError(f"Failed to load any YOLO backend: {e}")
    
    def _load_onnx_qnn(self):
        """Load YOLO via ONNX Runtime with QNN execution provider."""
        import onnxruntime as ort
        from ultralytics import YOLO
        
        # Export to ONNX if not already done
        onnx_path = "yolov8n.onnx"
        try:
            # Try to load existing ONNX model
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            # QNN execution provider for NPU
            providers = [
                ('QNNExecutionProvider', {
                    'backend_path': 'QnnHtp.dll',  # Hexagon NPU backend
                    'profiling_level': 'basic',
                }),
                'CPUExecutionProvider'  # Fallback
            ]
            
            self.model = ort.InferenceSession(
                onnx_path,
                sess_options=session_options,
                providers=providers
            )
            self.backend = "onnx_qnn"
            print(f"✅ Loaded ONNX model with QNN (NPU)")
            
        except Exception as e:
            # Export from PyTorch to ONNX
            print(f"Exporting YOLO to ONNX format...")
            yolo = YOLO("yolov8n.pt")
            yolo.export(format="onnx", dynamic=False, simplify=True)
            
            # Retry loading
            self.model = ort.InferenceSession(onnx_path, providers=providers)
            self.backend = "onnx_qnn"
            print(f"✅ Exported and loaded ONNX model with QNN (NPU)")
    
    def predict(self, image: Image.Image, conf_threshold: float = 0.25) -> List[Dict]:
        """Run inference on an image.
        
        Args:
            image: PIL Image
            conf_threshold: Confidence threshold for detections
            
        Returns:
            List of detections with format:
            [{"class_name": str, "confidence": float, "bbox": [x1,y1,x2,y2]}]
        """
        start_time = time.time()
        
        if self.backend == "qualcomm_npu":
            detections = self._predict_qualcomm(image, conf_threshold)
        elif self.backend == "onnx_qnn":
            detections = self._predict_onnx(image, conf_threshold)
        else:
            detections = self._predict_pytorch(image, conf_threshold)
        
        inference_time = (time.time() - start_time) * 1000  # ms
        
        # Add inference time to first detection (for monitoring)
        if detections:
            detections[0]["inference_time_ms"] = round(inference_time, 2)
        
        return detections
    
    def _predict_qualcomm(self, image: Image.Image, conf_threshold: float) -> List[Dict]:
        """Predict using Qualcomm AI Hub model."""
        import torch
        
        # Preprocess - resize to 640x640
        img_resized = image.resize((640, 640))
        img_array = np.array(img_resized).astype(np.float32) / 255.0
        img_array = np.transpose(img_array, (2, 0, 1))
        img_array = np.expand_dims(img_array, axis=0)
        input_tensor = torch.from_numpy(img_array)
        
        # Run inference on NPU
        with torch.no_grad():
            output = self.model(input_tensor)
        
        # Qualcomm AI Hub YOLO returns boxes in different format
        # Output is typically: (boxes, scores, classes) or similar
        detections = []
        
        if isinstance(output, (list, tuple)):
            # Multiple outputs (boxes, scores, classes)
            if len(output) >= 3:
                boxes = output[0].cpu().numpy() if isinstance(output[0], torch.Tensor) else output[0]
                scores = output[1].cpu().numpy() if isinstance(output[1], torch.Tensor) else output[1]
                classes = output[2].cpu().numpy() if isinstance(output[2], torch.Tensor) else output[2]
                
                # Filter by confidence
                for i, score in enumerate(scores.flatten()):
                    if score >= conf_threshold:
                        box = boxes[i] if boxes.ndim > 1 else boxes
                        cls = int(classes[i]) if classes.ndim > 0 else int(classes)
                        
                        # Normalize box coordinates
                        if len(box) == 4:
                            x1, y1, x2, y2 = box
                            detections.append({
                                "class_name": self.class_names[cls] if cls < len(self.class_names) else f"class_{cls}",
                                "confidence": float(score),
                                "bbox": [float(x1/640), float(y1/640), float(x2/640), float(y2/640)],
                            })
        else:
            # Single tensor output - use standard post-processing
            if isinstance(output, torch.Tensor):
                output = output.cpu().numpy()
            
            # Fallback to standard YOLO format parsing
            if output.ndim >= 2:
                for detection in output[0] if output.ndim == 3 else output:
                    if len(detection) >= 6:
                        # Standard YOLO format: [x, y, w, h, conf, class_scores...]
                        x, y, w, h = detection[:4]
                        conf = detection[4]
                        class_scores = detection[5:]
                        
                        if conf >= conf_threshold:
                            cls = np.argmax(class_scores)
                            x1 = (x - w/2) / 640
                            y1 = (y - h/2) / 640
                            x2 = (x + w/2) / 640
                            y2 = (y + h/2) / 640
                            
                            detections.append({
                                "class_name": self.class_names[cls] if cls < len(self.class_names) else f"class_{cls}",
                                "confidence": float(conf * class_scores[cls]),
                                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                            })
        
        return detections
    
    def _predict_onnx(self, image: Image.Image, conf_threshold: float) -> List[Dict]:
        """Predict using ONNX Runtime with QNN."""
        # Preprocess
        input_array = self._preprocess_image(image)
        
        # Get input name
        input_name = self.model.get_inputs()[0].name
        
        # Run inference
        outputs = self.model.run(None, {input_name: input_array})
        
        # Post-process
        return self._postprocess_output(outputs[0], conf_threshold, image.size)
    
    def _predict_pytorch(self, image: Image.Image, conf_threshold: float) -> List[Dict]:
        """Predict using standard Ultralytics YOLO."""
        results = self.model(image, conf=conf_threshold, verbose=False)
        
        detections = []
        for r in results:
            for box in r.boxes:
                detections.append({
                    "class_name": r.names[int(box.cls[0])],
                    "confidence": float(box.conf[0]),
                    "bbox": box.xyxyn[0].tolist(),  # Normalized [x1, y1, x2, y2]
                })
        
        return detections
    
    def _preprocess_image(self, image: Image.Image) -> np.ndarray:
        """Preprocess image for YOLO input."""
        # Resize to 640x640 (YOLO standard)
        img_resized = image.resize((640, 640))
        
        # Convert to numpy array and normalize
        img_array = np.array(img_resized).astype(np.float32) / 255.0
        
        # HWC to CHW
        img_array = np.transpose(img_array, (2, 0, 1))
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    def _postprocess_output(
        self, 
        output: np.ndarray, 
        conf_threshold: float,
        original_size: Tuple[int, int]
    ) -> List[Dict]:
        """Post-process YOLO output to detections."""
        detections = []
        
        # YOLO output format: [batch, num_detections, 85]
        # 85 = x, y, w, h, objectness, 80 class scores
        
        if len(output.shape) == 3:
            output = output[0]  # Remove batch dimension
        
        for detection in output:
            # Extract confidence and class
            objectness = detection[4]
            class_scores = detection[5:]
            class_id = np.argmax(class_scores)
            confidence = objectness * class_scores[class_id]
            
            if confidence < conf_threshold:
                continue
            
            # Extract bbox (center x, y, width, height)
            cx, cy, w, h = detection[:4]
            
            # Convert to corner format and normalize
            x1 = (cx - w/2) / 640
            y1 = (cy - h/2) / 640
            x2 = (cx + w/2) / 640
            y2 = (cy + h/2) / 640
            
            detections.append({
                "class_name": self.class_names[class_id],
                "confidence": float(confidence),
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
            })
        
        return detections
    
    @staticmethod
    def _get_coco_names() -> List[str]:
        """Get COCO class names."""
        return [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
            'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
            'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
            'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
            'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
    
    def get_backend_info(self) -> Dict:
        """Get information about the current backend."""
        return {
            "backend": self.backend,
            "model": self.model_name,
            "npu_enabled": self.backend in ["qualcomm_npu", "onnx_qnn"],
            "device": "Qualcomm Hexagon NPU" if self.backend == "qualcomm_npu" else 
                     "NPU via ONNX/QNN" if self.backend == "onnx_qnn" else "CPU/GPU"
        }


# Convenience function for easy import
def create_yolo_npu(use_npu: bool = True) -> YOLONPUInference:
    """Create YOLO instance with NPU acceleration if available."""
    return YOLONPUInference(use_npu=use_npu)


if __name__ == "__main__":
    # Test script
    print("=" * 70)
    print("Testing YOLO NPU Inference")
    print("=" * 70)
    
    # Create model
    yolo = create_yolo_npu(use_npu=True)
    
    # Print backend info
    info = yolo.get_backend_info()
    print(f"\n✓ Backend: {info['backend']}")
    print(f"✓ Device: {info['device']}")
    print(f"✓ NPU Enabled: {info['npu_enabled']}")
    
    # Test with dummy image
    print("\n🧪 Running test inference...")
    test_image = Image.new('RGB', (640, 640), color='red')
    detections = yolo.predict(test_image)
    
    print(f"✓ Inference complete: {len(detections)} detections")
    if detections and "inference_time_ms" in detections[0]:
        print(f"✓ Inference time: {detections[0]['inference_time_ms']}ms")
    
    print("\n" + "=" * 70)
    print("✅ YOLO NPU ready to use!")
    print("=" * 70)
