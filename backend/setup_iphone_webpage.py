#!/usr/bin/env python3
"""
Setup script to create a custom iPhone camera webpage.

This generates an HTML file with your laptop's IP address embedded,
so you can just open it on your iPhone's browser.
"""

import socket
import subprocess
from pathlib import Path

def get_local_ip():
    """Get the local IP address of this machine."""
    try:
        # Create a socket to determine local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        # Fallback: try ifconfig
        try:
            result = subprocess.run(
                ["ifconfig"], 
                capture_output=True, 
                text=True
            )
            for line in result.stdout.split('\n'):
                if 'inet ' in line and '127.0.0.1' not in line:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == 'inet' and i + 1 < len(parts):
                            ip = parts[i + 1]
                            if not ip.startswith('127.'):
                                return ip
        except:
            pass
    
    return "YOUR_LAPTOP_IP"

def setup_iphone_page():
    """Create a customized iPhone camera page with the laptop's IP."""
    
    print("=" * 70)
    print("iPhone Camera Webpage Setup")
    print("=" * 70)
    
    # Get laptop IP
    laptop_ip = get_local_ip()
    server_url = f"http://{laptop_ip}:8002"
    
    print(f"\n✓ Detected laptop IP: {laptop_ip}")
    
    # Read template
    template_path = Path(__file__).parent / "IPHONE_SIMPLE.html"
    
    if not template_path.exists():
        print(f"\n❌ Template not found: {template_path}")
        return
    
    with open(template_path, 'r') as f:
        html_content = f.read()
    
    # Replace placeholder with actual server URL
    html_content = html_content.replace('SERVER_URL_PLACEHOLDER', server_url)
    
    # Save customized version
    output_path = Path(__file__).parent / "iphone_camera.html"
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    print(f"✓ Created: {output_path}")
    
    # Create QR code URL (optional - requires qrcode library)
    try:
        import qrcode
        qr_url = f"http://{laptop_ip}:8002/iphone_camera.html"
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(qr_url)
        qr.make(fit=True)
        
        print("\n" + "=" * 70)
        print("QR Code (scan with iPhone camera):")
        print("=" * 70)
        qr.print_ascii(invert=True)
    except ImportError:
        pass
    
    print("\n" + "=" * 70)
    print("✅ Setup Complete!")
    print("=" * 70)
    print(f"\n📱 On your iPhone:")
    print(f"   1. Open Safari")
    print(f"   2. Go to: http://{laptop_ip}:8002/iphone_camera.html")
    print(f"   3. Tap 'Start Camera'")
    print(f"   4. Allow camera access")
    print(f"\n💻 On your laptop:")
    print(f"   View stream at: http://localhost:8002/stream")
    print(f"   Or in Drone Vision frontend: http://localhost:5173/manual")
    print("\n" + "=" * 70)
    
    # Also serve the file via the iPhone stream server
    print("\n💡 Tip: The iPhone stream server will serve this file automatically")
    print(f"   Just visit: http://{laptop_ip}:8002/iphone_camera.html")
    print("=" * 70)

if __name__ == "__main__":
    setup_iphone_page()
