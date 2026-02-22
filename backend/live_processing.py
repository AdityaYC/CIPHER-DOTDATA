"""Live Processing Pipeline: iPhone Camera → YOLO → World Graph → Llama Vision

This connects all the pieces:
1. iPhone sends camera frames
2. YOLO NPU detects objects in real-time
3. World Graph builds spatial map
4. Llama Vision reasons about what to do next
5. Frontend displays everything

Usage:
    python3 live_processing.py
"""

import asyncio
import time
import requests
from PIL import Image
import io
import json

# Configuration
IPHONE_STREAM_URL = "http://localhost:8002/latest_frame"
GRAPH_API_URL = "http://localhost:8001/ingest_frame"
MAIN_BACKEND_URL = "http://localhost:8000"

# Simulated GPS (in real drone, this would come from GPS sensor)
current_gps = {
    "lat": 37.4275,
    "lon": -122.1697,
    "altitude": 1.5,
    "yaw": 0.0
}

class LiveProcessor:
    """Processes live iPhone camera feed through the complete pipeline."""
    
    def __init__(self):
        self.frame_count = 0
        self.start_time = time.time()
        self.detections_total = 0
        
    async def process_frame(self):
        """Get frame from iPhone and process it."""
        try:
            # 1. Get latest frame from iPhone
            response = requests.get(IPHONE_STREAM_URL, timeout=2)
            if response.status_code != 200:
                return None
            
            image_bytes = response.content
            image = Image.open(io.BytesIO(image_bytes))
            
            # 2. Convert to base64 for API
            import base64
            buffer = io.BytesIO()
            image.save(buffer, format='JPEG', quality=85)
            img_b64 = base64.b64encode(buffer.getvalue()).decode()
            
            # 3. Send to Graph API (YOLO NPU + World Graph)
            # This will:
            # - Run YOLO detection on NPU
            # - Categorize objects (survivor/hazard/exit/clear)
            # - Add to spatial world graph
            # - Calculate distances
            graph_response = requests.post(
                GRAPH_API_URL,
                json={
                    "gps_lat": current_gps["lat"],
                    "gps_lon": current_gps["lon"],
                    "altitude_m": current_gps["altitude"],
                    "yaw_deg": current_gps["yaw"],
                    "image_b64": img_b64
                },
                timeout=5
            )
            
            if graph_response.status_code == 200:
                result = graph_response.json()
                self.frame_count += 1
                
                if result.get("node_added"):
                    detections = result.get("detections_count", 0)
                    self.detections_total += detections
                    
                    return {
                        "success": True,
                        "node_id": result.get("node_id"),
                        "detections": detections,
                        "frame_count": self.frame_count
                    }
            
            return {"success": False, "error": "Graph API error"}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def run(self, duration_seconds=60):
        """Run live processing for specified duration."""
        print("=" * 70)
        print("🚀 Live Processing Pipeline")
        print("=" * 70)
        print("\nPipeline:")
        print("  📱 iPhone Camera")
        print("  ↓")
        print("  🤖 YOLO NPU (Object Detection)")
        print("  ↓")
        print("  🗺️  World Graph (Spatial Mapping)")
        print("  ↓")
        print("  🧠 Llama Vision (Reasoning)")
        print("  ↓")
        print("  💻 Frontend (Visualization)")
        print("\n" + "=" * 70)
        print(f"\nRunning for {duration_seconds} seconds...")
        print("Press Ctrl+C to stop\n")
        
        start = time.time()
        
        try:
            while time.time() - start < duration_seconds:
                result = await self.process_frame()
                
                if result and result.get("success"):
                    elapsed = time.time() - self.start_time
                    fps = self.frame_count / elapsed if elapsed > 0 else 0
                    
                    print(f"✓ Frame {self.frame_count} | "
                          f"Detections: {result.get('detections', 0)} | "
                          f"FPS: {fps:.1f} | "
                          f"Node: {result.get('node_id', 'N/A')}", 
                          end='\r')
                elif result:
                    print(f"⚠ {result.get('error', 'Unknown error')}", end='\r')
                
                # Process at ~5 FPS (enough for spatial mapping)
                await asyncio.sleep(0.2)
                
                # Simulate GPS movement (in real drone, GPS updates automatically)
                current_gps["lat"] += 0.00001
                current_gps["lon"] += 0.00001
                current_gps["yaw"] = (current_gps["yaw"] + 5) % 360
                
        except KeyboardInterrupt:
            print("\n\n✅ Stopped by user")
        
        # Print final statistics
        elapsed = time.time() - self.start_time
        print("\n\n" + "=" * 70)
        print("📊 Final Statistics")
        print("=" * 70)
        print(f"Frames processed: {self.frame_count}")
        print(f"Total detections: {self.detections_total}")
        print(f"Duration: {elapsed:.1f}s")
        print(f"Average FPS: {self.frame_count/elapsed:.1f}")
        
        # Get world graph stats
        try:
            stats_response = requests.get("http://localhost:8001/stats")
            if stats_response.status_code == 200:
                stats = stats_response.json()
                print(f"\n🗺️  World Graph:")
                print(f"  Nodes: {stats['node_count']}")
                print(f"  Coverage: {stats['coverage_m2']}m²")
                print(f"  Categories:")
                for cat, count in stats['category_counts'].items():
                    if count > 0:
                        print(f"    {cat}: {count}")
        except:
            pass
        
        print("=" * 70)


async def main():
    """Main entry point."""
    print("\n⚠️  Prerequisites:")
    print("  1. iPhone stream server running (port 8002)")
    print("  2. Graph API running (port 8001)")
    print("  3. iPhone sending camera frames")
    print("\nChecking services...\n")
    
    # Check services
    services = {
        "iPhone Stream": "http://localhost:8002/health",
        "Graph API": "http://localhost:8001/health",
    }
    
    all_ok = True
    for name, url in services.items():
        try:
            response = requests.get(url, timeout=2)
            if response.status_code == 200:
                print(f"✓ {name}: OK")
            else:
                print(f"✗ {name}: Error {response.status_code}")
                all_ok = False
        except:
            print(f"✗ {name}: Not running")
            all_ok = False
    
    if not all_ok:
        print("\n❌ Some services are not running. Please start them first.")
        return
    
    print("\n✅ All services ready!\n")
    
    # Run processor
    processor = LiveProcessor()
    await processor.run(duration_seconds=60)


if __name__ == "__main__":
    asyncio.run(main())
