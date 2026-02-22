# ============================================================
# PHANTOM CODE — Configuration
# ============================================================

import platform

# --- Mac vs Windows ---
# When no IP camera URL is set, use built-in laptop webcam (index 0) on all platforms.
USE_MAC_WEBCAM_DEMO = platform.system() == "Darwin"
USE_BUILTIN_WEBCAM = True  # use laptop webcam when no IP camera URL

# --- Optional: Use an IP Webcam URL for demo (like your previous iteration) ---
# Set this to your phone's IP Webcam URL to use that stream instead of built-in camera.
# Example: "http://192.168.1.5:8080/video" — open IP Webcam app on phone, copy the URL shown.
# Leave None to use built-in webcam (Mac) or CAMERA_FEEDS below (Windows).
import os as _os
IP_CAMERA_URL = _os.environ.get("PHANTOM_IP_CAMERA_URL", "").strip() or None  # e.g. "http://192.168.1.5:8080/video"

# --- Camera Feeds ---
# IP Webcam URLs (Windows + phones), or camera indices (Mac demo: 0 = built-in webcam).
if IP_CAMERA_URL:
    # Demo: one IP link used for Drone-1 (and mirrored to Drone-2 so both sectors show it).
    CAMERA_FEEDS = {
        "Drone-1": IP_CAMERA_URL,
        "Drone-2": IP_CAMERA_URL,
    }
elif USE_MAC_WEBCAM_DEMO or USE_BUILTIN_WEBCAM:
    CAMERA_FEEDS = {
        "Drone-1": 0,
        "Drone-2": 0,
    }
else:
    CAMERA_FEEDS = {
        "Drone-1": "http://192.168.137.101:8080/video",
        "Drone-2": "http://192.168.137.102:8080/video",
    }

# --- Zone Mapping ---
# Each drone camera maps to a zone on the 800x600 tactical map.
# YOLO detections in pixel coords get linearly mapped into these zones.
CAMERA_ZONES = {
    "Drone-1": {"x_min": 50, "y_min": 50, "x_max": 370, "y_max": 550},
    "Drone-2": {"x_min": 430, "y_min": 50, "x_max": 750, "y_max": 550},
}

# --- Zone Labels ---
ZONE_LABELS = {
    "Drone-1": "SECTOR ALPHA",
    "Drone-2": "SECTOR BRAVO",
}

# --- Map ---
MAP_WIDTH = 800
MAP_HEIGHT = 600

# --- YOLO ---
# Use 640 for better accuracy (NPU is fast enough). Set YOLO_USE_FAST = True for speed over accuracy.
YOLO_USE_FAST = False  # False = 640 model (accuracy); True = 320 model (faster, less accurate)
YOLO_ONNX_PATH = "models/yolov8_det.onnx"
YOLO_ONNX_FAST_PATH = "models/yolov8_det_320.onnx"
YOLO_CONFIDENCE_THRESHOLD = 0.45
YOLO_INPUT_SIZE = 640  # Used when loading 640 model; 320 when loading fast model

# --- NPU (Qualcomm Hexagon) — required for track; Task Manager will show NPU utilisation ---
# Run .\scripts\install_onnxruntime_qnn.ps1 from repo root so QNNExecutionProvider is available.
PREFER_NPU_OVER_GPU = True  # YOLO and Depth use NPU first (faster); do not set False
USE_GPU = True  # Fallback when NPU unavailable
SPLIT_NPU_GPU = False  # Keep False so YOLO and Depth both use NPU
# QNN backend DLL — leave None to auto-detect (onnxruntime-qnn bundles it in capi/QnnHtp.dll)
# If YOLO stays on CPU, set to full path to QnnHtp.dll, e.g. <Python site-packages>/onnxruntime/capi/QnnHtp.dll
QNN_DLL_PATH = None  # If None, backend auto-resolves QnnHtp.dll at startup (Qualcomm AIStack / ort package).

# --- LLM ---
LLM_MODEL = "phi3:mini"
LLM_ENDPOINT = "http://localhost:11434/api/generate"
LLM_UPDATE_INTERVAL = 12  # seconds between LLM advisory updates

# --- Mission Presets ---
MISSIONS = {
    "search_rescue": {
        "label": "Search & Rescue",
        "prompt": (
            "You are a tactical AI advisor for a search and rescue drone operation. "
            "Based on the detected objects and their zones, provide concise actionable advice "
            "to help locate and assist survivors. Prioritize human detections. "
            "Keep responses under 3 sentences. Be direct and specific about which drone "
            "should do what."
        ),
    },
    "perimeter": {
        "label": "Perimeter Surveillance",
        "prompt": (
            "You are a tactical AI advisor for perimeter surveillance drones. "
            "Based on the detected objects and their zones, identify potential intrusions, "
            "suspicious activity, or coverage gaps. Recommend repositioning to maintain "
            "full perimeter coverage. Keep responses under 3 sentences."
        ),
    },
    "threat_detection": {
        "label": "Threat Detection",
        "prompt": (
            "You are a tactical AI advisor for threat detection drones in a high-risk zone. "
            "Based on the detected objects and their zones, identify potential threats, "
            "flag unusual object combinations, and recommend evasive or investigative actions. "
            "Keep responses under 3 sentences. Prioritize operator safety."
        ),
    },
    "damage_assessment": {
        "label": "Damage Assessment",
        "prompt": (
            "You are a tactical AI advisor for post-disaster damage assessment drones. "
            "Based on the detected objects and their zones, identify structural damage indicators, "
            "blocked routes, and areas requiring immediate attention. Recommend survey priorities. "
            "Keep responses under 3 sentences."
        ),
    },
}

# --- Server ---
API_HOST = "0.0.0.0"
API_PORT = 8000
