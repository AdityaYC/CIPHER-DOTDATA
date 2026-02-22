"""API Server for World Graph - receives frames from iPhone and serves graph data.

Three main endpoints:
1. POST /ingest_frame - Receive frame + GPS from iPhone, run YOLO, add to graph
2. GET /graph - Return full world graph for tactical map
3. GET /stats - Return coverage stats and node counts
"""

import base64
import io
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image

from world_graph import WorldGraph, ObjectCategory

# Initialize FastAPI app
app = FastAPI(title="World Graph API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global world graph instance
world_graph = WorldGraph()

# YOLO model (lazy loaded)
yolo_model = None


def get_yolo_model():
    """Lazy load standard YOLO model."""
    global yolo_model
    if yolo_model is None:
        from ultralytics import YOLO
        yolo_model = YOLO("yolov8n.pt")
        print("✅ Graph API YOLO loaded")
    return yolo_model


# ---------------------------------------------------------------------------
# Request/Response Models
# ---------------------------------------------------------------------------

class IngestFrameRequest(BaseModel):
    """Request body for frame ingestion."""
    gps_lat: float
    gps_lon: float
    altitude_m: float
    yaw_deg: float
    image_b64: str  # Base64 encoded JPEG/PNG


class IngestFrameResponse(BaseModel):
    """Response after ingesting a frame."""
    success: bool
    node_added: bool
    node_id: Optional[str] = None
    detections_count: int = 0
    message: str = ""


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.post("/ingest_frame", response_model=IngestFrameResponse)
async def ingest_frame(request: IngestFrameRequest):
    """Receive a frame from iPhone, run YOLO, add to world graph.
    
    This is the main ingestion endpoint that your iPhone app calls.
    """
    try:
        # Decode image
        image_bytes = base64.b64decode(request.image_b64)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Run YOLO detection
        yolo = get_yolo_model()
        results = yolo(image, conf=0.25, verbose=False)
        detections = []
        for r in results:
            for box in r.boxes:
                detections.append({
                    "class_name": r.names[int(box.cls[0])],
                    "confidence": float(box.conf[0]),
                    "bbox": box.xyxyn[0].tolist(),
                })
        
        # Create thumbnail for storage (optional)
        thumbnail = image.copy()
        thumbnail.thumbnail((256, 256))
        thumb_buffer = io.BytesIO()
        thumbnail.save(thumb_buffer, format="JPEG", quality=85)
        thumb_b64 = base64.b64encode(thumb_buffer.getvalue()).decode("ascii")
        
        # Add to world graph
        node = world_graph.add_node(
            lat=request.gps_lat,
            lon=request.gps_lon,
            alt=request.altitude_m,
            yaw=request.yaw_deg,
            yolo_detections=detections,
            image_b64=thumb_b64,
        )
        
        if node:
            return IngestFrameResponse(
                success=True,
                node_added=True,
                node_id=node.node_id,
                detections_count=len(detections),
                message=f"Node added with {len(detections)} detections",
            )
        else:
            return IngestFrameResponse(
                success=True,
                node_added=False,
                detections_count=len(detections),
                message="Position too close to last node, not added",
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest_frame_multipart")
async def ingest_frame_multipart(
    gps_lat: float = Form(...),
    gps_lon: float = Form(...),
    altitude_m: float = Form(...),
    yaw_deg: float = Form(...),
    image: UploadFile = File(...),
):
    """Alternative endpoint using multipart form data (easier for some clients)."""
    try:
        # Read image
        image_bytes = await image.read()
        pil_image = Image.open(io.BytesIO(image_bytes))
        
        # Run YOLO
        yolo = get_yolo_model()
        
        if hasattr(yolo, 'predict'):
            detections = yolo.predict(pil_image, conf_threshold=0.25)
        else:
            results = yolo(pil_image, verbose=False)
            detections = []
            for r in results:
                for box in r.boxes:
                    detections.append({
                        "class_name": r.names[int(box.cls[0])],
                        "confidence": float(box.conf[0]),
                        "bbox": box.xyxyn[0].tolist(),
                    })
        
        # Add to graph
        node = world_graph.add_node(
            lat=gps_lat,
            lon=gps_lon,
            alt=altitude_m,
            yaw=yaw_deg,
            yolo_detections=detections,
        )
        
        return {
            "success": True,
            "node_added": node is not None,
            "node_id": node.node_id if node else None,
            "detections_count": len(detections),
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/graph")
async def get_graph():
    """Return the complete world graph.
    
    Your tactical map frontend calls this to visualize the graph.
    """
    return world_graph.get_graph()


@app.get("/stats")
async def get_stats():
    """Return statistics about the world graph.
    
    Returns coverage percentage, node counts, detection counts by category.
    """
    return world_graph.get_stats()


@app.get("/nodes/survivors")
async def get_survivor_nodes():
    """Get all nodes that detected survivors."""
    nodes = world_graph.get_nodes_by_category(ObjectCategory.SURVIVOR)
    return {
        "count": len(nodes),
        "nodes": [node.to_dict() for node in nodes],
    }


@app.get("/nodes/hazards")
async def get_hazard_nodes():
    """Get all nodes that detected hazards."""
    nodes = world_graph.get_nodes_by_category(ObjectCategory.HAZARD)
    return {
        "count": len(nodes),
        "nodes": [node.to_dict() for node in nodes],
    }


@app.get("/nodes/exits")
async def get_exit_nodes():
    """Get all nodes that detected exits."""
    nodes = world_graph.get_nodes_by_category(ObjectCategory.EXIT)
    return {
        "count": len(nodes),
        "nodes": [node.to_dict() for node in nodes],
    }


@app.post("/clear")
async def clear_graph():
    """Clear all nodes from the graph (useful for testing)."""
    world_graph.clear()
    return {"success": True, "message": "Graph cleared"}


@app.get("/health")
async def health():
    """Health check endpoint."""
    backend_info = {}
    if yolo_model is not None and hasattr(yolo_model, 'get_backend_info'):
        backend_info = yolo_model.get_backend_info()
    
    return {
        "status": "ok",
        "yolo_loaded": yolo_model is not None,
        "node_count": world_graph.node_count,
        "yolo_backend": backend_info.get("backend", "unknown"),
        "npu_enabled": backend_info.get("npu_enabled", False),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
