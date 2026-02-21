"""Test script to verify Llama Vision access and functionality."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from app import models, image_db
from PIL import Image

def test_llama_vision():
    """Test Llama Vision model loading and inference."""
    
    print("=" * 60)
    print("Testing Llama 3.2 Vision Access")
    print("=" * 60)
    
    # Check if we have frames
    if not image_db.db:
        print("\n❌ No frames loaded. Loading database...")
        image_db.load()
    
    print(f"\n✓ Database loaded: {len(image_db.db)} frames")
    
    # Get a test image
    if image_db.db:
        test_frame = image_db.db[0]
        print(f"✓ Using test frame: {test_frame['filename']}")
        
        # Load image
        image = Image.open(test_frame['path'])
        print(f"✓ Image loaded: {image.size}")
        
        # Test Llama Vision
        print("\n🔄 Loading Llama Vision model...")
        print("   (This will download ~22GB on first run)")
        print("   Please wait...")
        
        try:
            models.load_llama_vision()
            print("\n✅ Llama Vision loaded successfully!")
            
            # Test inference
            print("\n🔄 Testing inference...")
            prompt = "Describe what you see in this image in one sentence."
            response = models.infer_llama(image, prompt)
            
            print("\n✅ Inference successful!")
            print(f"\nResponse: {response}")
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("\nTroubleshooting:")
            print("1. Make sure you ran: hf auth login")
            print("2. Check you have access to: meta-llama/Llama-3.2-11B-Vision-Instruct")
            print("3. Verify you have enough disk space (~22GB)")
            return False
    
    print("\n" + "=" * 60)
    print("✅ All tests passed! Llama Vision is ready to use.")
    print("=" * 60)
    return True

if __name__ == "__main__":
    test_llama_vision()
