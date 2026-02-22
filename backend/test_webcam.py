"""Test script - use laptop webcam to simulate drone and build world graph.

This lets you test the world graph system without an iPhone.
Move your laptop around the room and watch nodes accumulate.
"""

import cv2
import base64
import time
import requests
from PIL import Image
import io

# Configuration
API_URL = "http://localhost:8001"
FAKE_GPS_START = (37.4275, -122.1697)  # Stanford coordinates
GPS_INCREMENT = 0.00001  # Small increment per frame
ALTITUDE = 1.5  # meters (typical handheld height)
CAPTURE_INTERVAL = 1.0  # seconds between captures

def capture_and_send():
    """Capture from webcam and send to graph API."""
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open webcam")
        return
    
    print("=" * 70)
    print("🎥 Webcam Test - World Graph Builder")
    print("=" * 70)
    print("\nControls:")
    print("  - Move laptop around to simulate drone movement")
    print("  - Press 'q' to quit")
    print("  - Press 's' to view stats")
    print("  - Press 'c' to clear graph")
    print("\nStarting in 3 seconds...\n")
    time.sleep(3)
    
    # Fake GPS position (increments to simulate movement)
    lat, lon = FAKE_GPS_START
    yaw = 0.0
    frame_count = 0
    last_capture = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to capture frame")
                break
            
            # Display frame
            cv2.putText(frame, f"Nodes: {frame_count}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"GPS: ({lat:.6f}, {lon:.6f})", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.imshow('Webcam Test', frame)
            
            # Capture and send at intervals
            current_time = time.time()
            if current_time - last_capture >= CAPTURE_INTERVAL:
                last_capture = current_time
                
                # Convert frame to JPEG
                pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                buffer = io.BytesIO()
                pil_image.save(buffer, format="JPEG", quality=85)
                image_b64 = base64.b64encode(buffer.getvalue()).decode("ascii")
                
                # Send to API
                try:
                    response = requests.post(
                        f"{API_URL}/ingest_frame",
                        json={
                            "gps_lat": lat,
                            "gps_lon": lon,
                            "altitude_m": ALTITUDE,
                            "yaw_deg": yaw,
                            "image_b64": image_b64,
                        },
                        timeout=5,
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        if data["node_added"]:
                            frame_count += 1
                            print(f"✓ Node {data['node_id']} added - {data['detections_count']} detections")
                        else:
                            print(f"○ Position too close, not added - {data['detections_count']} detections")
                    else:
                        print(f"❌ API error: {response.status_code}")
                        
                except requests.exceptions.ConnectionError:
                    print("❌ Cannot connect to API. Is it running on port 8001?")
                except Exception as e:
                    print(f"❌ Error: {e}")
                
                # Increment fake GPS (simulate movement)
                lat += GPS_INCREMENT
                lon += GPS_INCREMENT
                yaw = (yaw + 15) % 360  # Slowly rotate
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Show stats
                try:
                    stats = requests.get(f"{API_URL}/stats").json()
                    print("\n" + "=" * 50)
                    print("📊 Graph Statistics:")
                    print(f"  Nodes: {stats['node_count']}")
                    print(f"  Total Detections: {stats['total_detections']}")
                    print(f"  Coverage: {stats['coverage_m2']}m² ({stats['coverage_percentage']}%)")
                    print(f"  Categories: {stats['category_counts']}")
                    print("=" * 50 + "\n")
                except Exception as e:
                    print(f"❌ Error fetching stats: {e}")
            elif key == ord('c'):
                # Clear graph
                try:
                    requests.post(f"{API_URL}/clear")
                    frame_count = 0
                    print("🗑️  Graph cleared")
                except Exception as e:
                    print(f"❌ Error clearing graph: {e}")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\n✅ Test complete")
        
        # Final stats
        try:
            stats = requests.get(f"{API_URL}/stats").json()
            print("\n" + "=" * 70)
            print("📊 Final Statistics:")
            print("=" * 70)
            print(f"Total Nodes: {stats['node_count']}")
            print(f"Total Detections: {stats['total_detections']}")
            print(f"Coverage: {stats['coverage_m2']}m²")
            print(f"\nDetections by Category:")
            for cat, count in stats['category_counts'].items():
                if count > 0:
                    print(f"  {cat}: {count}")
            print("=" * 70)
        except:
            pass


if __name__ == "__main__":
    print("\n⚠️  Make sure the Graph API is running:")
    print("    cd local_backend")
    print("    python3 graph_api.py")
    print("\nPress Enter to start webcam test...")
    input()
    
    capture_and_send()
