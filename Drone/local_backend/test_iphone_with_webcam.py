"""Test iPhone stream server using Mac webcam.

This simulates an iPhone sending frames to the server.
Run this to test the system before connecting your actual iPhone.
"""

import cv2
import base64
import requests
import time

SERVER_URL = "http://localhost:8002/upload_frame"

def test_stream():
    print("=" * 70)
    print("Testing iPhone Stream with Mac Webcam")
    print("=" * 70)
    print("\nStarting webcam capture...")
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Cannot open webcam")
        return
    
    print("✅ Webcam opened")
    print(f"📡 Sending frames to: {SERVER_URL}")
    print("\nPress Ctrl+C to stop\n")
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("❌ Failed to capture frame")
                break
            
            # Encode as JPEG
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            img_b64 = base64.b64encode(buffer).decode()
            
            # Send to server
            try:
                response = requests.post(SERVER_URL, json={
                    'image_b64': img_b64,
                    'timestamp': time.time()
                }, timeout=1)
                
                if response.status_code == 200:
                    frame_count += 1
                    elapsed = time.time() - start_time
                    fps = frame_count / elapsed if elapsed > 0 else 0
                    print(f"✓ Frame {frame_count} sent ({fps:.1f} FPS)", end='\r')
                else:
                    print(f"❌ Server error: {response.status_code}")
                    
            except requests.exceptions.ConnectionError:
                print("\n❌ Cannot connect to server. Is it running on port 8002?")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
            
            # ~10 FPS
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\n\n✅ Stopped")
        elapsed = time.time() - start_time
        print(f"\nStats:")
        print(f"  Frames sent: {frame_count}")
        print(f"  Duration: {elapsed:.1f}s")
        print(f"  Average FPS: {frame_count/elapsed:.1f}")
    finally:
        cap.release()
        print("\n" + "=" * 70)


if __name__ == "__main__":
    test_stream()
