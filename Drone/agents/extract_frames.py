"""Extract all video frames at 0.1s intervals and save locally.

Runs ffmpeg on Modal (since ffmpeg may not be installed locally),
then downloads the JPEGs to data/image_samples/.

Usage:
    modal run agents/extract_frames.py
"""

import subprocess
from pathlib import Path

import modal

_HERE = Path(__file__).resolve().parent
_PROJECT = _HERE.parent
_DATA = _PROJECT / "data"

app = modal.App("vision-extract")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .add_local_file(str(_DATA / "video.MP4"), "/data/video.MP4", copy=True)
)


@app.function(image=image, timeout=300)
def extract_frames() -> dict[str, bytes]:
    """Extract frames at 10fps and return as {filename: jpeg_bytes}."""
    import os

    os.makedirs("/tmp/frames", exist_ok=True)
    result = subprocess.run(
        [
            "ffmpeg", "-i", "/data/video.MP4",
            "-vf", "fps=10",
            "-q:v", "2",
            "/tmp/frames/frame_%05d.jpg",
        ],
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr.decode()[:500]}")

    frames = {}
    for f in sorted(Path("/tmp/frames").glob("frame_*.jpg")):
        frames[f.name] = f.read_bytes()

    print(f"Extracted {len(frames)} frames")
    return frames


@app.local_entrypoint()
def main():
    out_dir = Path(__file__).resolve().parent.parent / "data" / "image_samples"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Extracting frames on Modal...")
    frames = extract_frames.remote()
    print(f"Got {len(frames)} frames, saving to {out_dir}")

    for name, data in frames.items():
        (out_dir / name).write_bytes(data)

    print(f"Done. Saved {len(frames)} JPEGs to {out_dir}")
