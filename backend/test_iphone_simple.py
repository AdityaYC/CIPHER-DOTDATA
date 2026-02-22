"""Simple iPhone stream test - sends a test image to verify the server works.

This doesn't require webcam access - just tests the server endpoints.
"""

import requests
import base64
from PIL import Image
import io
import time

SERVER_URL = "http://localhost:8002"

def test_server():
    print("=" * 70)
    print("Testing iPhone Stream Server")
    print("=" * 70)
    
    # Check server health
    print("\n1. Checking server health...")
    try:
        response = requests.get(f"{SERVER_URL}/health")
        if response.status_code == 200:
            print("   ✅ Server is running")
        else:
            print(f"   ❌ Server returned {response.status_code}")
            return
    except requests.exceptions.ConnectionError:
        print("   ❌ Cannot connect to server. Is it running on port 8002?")
        print("   Run: python3 iphone_stream.py")
        return
    
    # Create a test image
    print("\n2. Creating test image...")
    img = Image.new('RGB', (640, 480), color=(73, 109, 137))
    
    # Add some text to make it interesting
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(img)
    draw.text((200, 200), "Test Frame", fill=(255, 255, 255))
    
    # Convert to base64
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG', quality=70)
    img_b64 = base64.b64encode(buffer.getvalue()).decode()
    print(f"   ✅ Test image created ({len(img_b64)} bytes)")
    
    # Send to server
    print("\n3. Uploading test frame...")
    try:
        response = requests.post(f"{SERVER_URL}/upload_frame", json={
            'image_b64': img_b64,
            'timestamp': time.time()
        })
        
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Frame uploaded successfully")
            print(f"      Timestamp: {data['timestamp']}")
            print(f"      Size: {data['size']} bytes")
        else:
            print(f"   ❌ Upload failed: {response.status_code}")
            print(f"      {response.text}")
            return
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return
    
    # Check status
    print("\n4. Checking server status...")
    response = requests.get(f"{SERVER_URL}/status")
    if response.status_code == 200:
        status = response.json()
        print(f"   ✅ Server status:")
        print(f"      Has frame: {status['has_frame']}")
        print(f"      Frame size: {status['frame_size']} bytes")
        print(f"      Age: {status['age_seconds']:.2f}s")
    
    # Get latest frame
    print("\n5. Retrieving latest frame...")
    response = requests.get(f"{SERVER_URL}/latest_frame")
    if response.status_code == 200:
        print(f"   ✅ Retrieved frame ({len(response.content)} bytes)")
        
        # Verify it's a valid image
        try:
            retrieved_img = Image.open(io.BytesIO(response.content))
            print(f"      Image size: {retrieved_img.size}")
            print(f"      Format: {retrieved_img.format}")
        except Exception as e:
            print(f"   ❌ Invalid image: {e}")
    
    print("\n" + "=" * 70)
    print("✅ All tests passed!")
    print("=" * 70)
    print("\nNext steps:")
    print("1. View stream in browser: http://localhost:8002/stream")
    print("2. Check status: http://localhost:8002/status")
    print("3. Connect your iPhone using one of the methods in IPHONE_SETUP.md")
    print("=" * 70)


if __name__ == "__main__":
    test_server()
