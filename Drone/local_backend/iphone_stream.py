"""iPhone Camera Streaming Server

Receives camera feed from iPhone and makes it available to the frontend.

Two modes:
1. HTTP POST - iPhone sends frames as base64 JPEG
2. WebSocket - Real-time bidirectional streaming

Usage:
    python3 iphone_stream.py
"""

import asyncio
import base64
import json
import time
from typing import Optional
from io import BytesIO

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel
from PIL import Image

app = FastAPI(title="iPhone Camera Stream Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state for latest frame
latest_frame: Optional[bytes] = None
latest_frame_time: float = 0
frame_lock = asyncio.Lock()

# Connected clients
connected_clients = set()


class FrameUpload(BaseModel):
    """Frame from iPhone camera."""
    image_b64: str
    timestamp: Optional[float] = None
    gps_lat: Optional[float] = None
    gps_lon: Optional[float] = None
    altitude_m: Optional[float] = None
    yaw_deg: Optional[float] = None


@app.post("/upload_frame")
async def upload_frame(frame: FrameUpload):
    """Receive frame from iPhone camera.
    
    iPhone app should POST to this endpoint with base64 encoded JPEG.
    """
    global latest_frame, latest_frame_time
    
    try:
        # Decode base64 image
        image_bytes = base64.b64decode(frame.image_b64)
        
        # Validate it's a valid image
        img = Image.open(BytesIO(image_bytes))
        img.verify()
        
        # Store latest frame
        async with frame_lock:
            latest_frame = image_bytes
            latest_frame_time = frame.timestamp or time.time()
        
        # Notify connected WebSocket clients
        if connected_clients:
            message = json.dumps({
                "type": "new_frame",
                "timestamp": latest_frame_time,
                "has_gps": frame.gps_lat is not None,
            })
            disconnected = set()
            for client in connected_clients:
                try:
                    await client.send_text(message)
                except:
                    disconnected.add(client)
            connected_clients.difference_update(disconnected)
        
        return {
            "success": True,
            "timestamp": latest_frame_time,
            "size": len(image_bytes),
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")


@app.get("/latest_frame")
async def get_latest_frame():
    """Get the latest frame from iPhone camera.
    
    Returns JPEG image directly.
    """
    async with frame_lock:
        if latest_frame is None:
            raise HTTPException(status_code=404, detail="No frame available")
        
        return Response(
            content=latest_frame,
            media_type="image/jpeg",
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "X-Frame-Timestamp": str(latest_frame_time),
            }
        )


@app.get("/stream")
async def stream_frames():
    """Stream frames as MJPEG (Motion JPEG).
    
    This creates a continuous stream that browsers can display as video.
    """
    async def generate():
        last_sent_time = 0
        
        while True:
            async with frame_lock:
                if latest_frame and latest_frame_time > last_sent_time:
                    frame_data = latest_frame
                    last_sent_time = latest_frame_time
                else:
                    frame_data = None
            
            if frame_data:
                # MJPEG format: each frame is a separate JPEG with boundary
                yield (
                    b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n'
                )
            
            await asyncio.sleep(0.033)  # ~30 FPS max
    
    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Connection": "keep-alive",
        }
    )


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time frame streaming.
    
    Clients connect here to receive frame updates.
    """
    await websocket.accept()
    connected_clients.add(websocket)
    
    try:
        # Send initial status
        await websocket.send_json({
            "type": "connected",
            "has_frame": latest_frame is not None,
            "timestamp": latest_frame_time if latest_frame else None,
        })
        
        # Keep connection alive and handle incoming messages
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("type") == "request_frame":
                # Send latest frame
                async with frame_lock:
                    if latest_frame:
                        frame_b64 = base64.b64encode(latest_frame).decode('ascii')
                        await websocket.send_json({
                            "type": "frame",
                            "image_b64": frame_b64,
                            "timestamp": latest_frame_time,
                        })
                    else:
                        await websocket.send_json({
                            "type": "error",
                            "message": "No frame available",
                        })
            
    except WebSocketDisconnect:
        connected_clients.discard(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        connected_clients.discard(websocket)


@app.get("/status")
async def get_status():
    """Get streaming status."""
    return {
        "has_frame": latest_frame is not None,
        "last_frame_time": latest_frame_time if latest_frame else None,
        "age_seconds": time.time() - latest_frame_time if latest_frame else None,
        "connected_clients": len(connected_clients),
        "frame_size": len(latest_frame) if latest_frame else 0,
    }


@app.get("/health")
async def health():
    """Health check."""
    return {"status": "ok", "service": "iphone_stream"}


@app.get("/iphone_camera.html")
async def serve_iphone_page():
    """Serve the iPhone camera webpage."""
    from fastapi.responses import FileResponse
    from pathlib import Path
    
    html_path = Path(__file__).parent / "iphone_camera.html"
    if html_path.exists():
        return FileResponse(html_path, media_type="text/html")
    else:
        raise HTTPException(status_code=404, detail="iPhone camera page not found")


if __name__ == "__main__":
    import uvicorn
    print("=" * 70)
    print("iPhone Camera Stream Server")
    print("=" * 70)
    print("\nEndpoints:")
    print("  POST /upload_frame  - iPhone uploads frames here")
    print("  GET  /latest_frame  - Get latest frame as JPEG")
    print("  GET  /stream        - MJPEG stream for browsers")
    print("  WS   /ws            - WebSocket for real-time updates")
    print("  GET  /status        - Check streaming status")
    print("\nStarting server on http://0.0.0.0:8002...")
    print("=" * 70)
    
    uvicorn.run(app, host="0.0.0.0", port=8002)
