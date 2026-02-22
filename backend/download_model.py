"""Download Llama 3.2 Vision model from Hugging Face.

This script downloads the model to your local cache so it's ready to use.
"""

from huggingface_hub import snapshot_download
import os

MODEL_ID = "meta-llama/Llama-3.2-11B-Vision-Instruct"

print("=" * 70)
print("Downloading Llama 3.2 Vision (11B) Model")
print("=" * 70)
print(f"\nModel: {MODEL_ID}")
print("Size: ~22GB")
print("This will take 5-15 minutes depending on your internet speed.\n")

try:
    print("Starting download...")
    print("(Files will be cached in ~/.cache/huggingface/hub/)\n")
    
    model_path = snapshot_download(
        repo_id=MODEL_ID,
        cache_dir=None,  # Uses default cache location
        resume_download=True,  # Resume if interrupted
    )
    
    print("\n" + "=" * 70)
    print("✅ Download Complete!")
    print("=" * 70)
    print(f"\nModel cached at: {model_path}")
    print("\nYou can now use the agent system!")
    print("The model will load automatically when you run a query.\n")
    
except Exception as e:
    print("\n" + "=" * 70)
    print("❌ Download Failed")
    print("=" * 70)
    print(f"\nError: {e}\n")
    print("Troubleshooting:")
    print("1. Make sure you ran: hf auth login")
    print("2. Verify access at: https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct")
    print("3. Check your internet connection")
    print("4. Ensure you have ~25GB free disk space\n")
