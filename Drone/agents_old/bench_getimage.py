"""Temporary benchmark: time each step of getImage pipeline.

Runs inside the same container image as GetImage so it has access to
the video, trajectory CSV, etc.

Usage:
    modal run agents/bench_getimage.py
"""

import csv
import math
import subprocess
import time
from pathlib import Path

import modal
import numpy as np

_HERE = Path(__file__).resolve().parent
_PROJECT = _HERE.parent
_DATA = _PROJECT / "data"

app = modal.App("vision-bench")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install("numpy<2")
    .add_local_file(str(_DATA / "trajectory_postprocessed.csv"), "/data/trajectory.csv", copy=True)
    .add_local_file(str(_DATA / "video.MP4"), "/data/video.MP4", copy=True)
)


@app.function(image=image, timeout=300)
def bench():
    results = {}

    # ---- 1. Load trajectory DB ----
    t0 = time.perf_counter()
    db = []
    with open("/data/trajectory.csv") as f:
        for row in csv.DictReader(f):
            db.append({
                "timestamp_s": float(row["timestamp_s"]),
                "x": float(row["x_m"]),
                "y": float(row["y_m"]),
                "z": float(row["z_m"]),
                "pitch": float(row["pitch_deg"]),
                "yaw": float(row["yaw_deg"]),
                "roll": float(row["roll_deg"]),
            })
    results["1_load_db"] = time.perf_counter() - t0

    # ---- 2. _find_best (linear scan) ----
    x, y, z, yaw = 0.0, 0.0, 0.0, 0.0
    t0 = time.perf_counter()
    best_score, best_idx = float("inf"), 0
    for i, e in enumerate(db):
        dx, dy, dz = e["x"] - x, e["y"] - y, e["z"] - z
        pos_d = math.sqrt(dx**2 + dy**2 + dz**2)
        d = (e["yaw"] - yaw) % 360
        dyaw = abs(d - 360 if d > 180 else d)
        ang_d = math.sqrt(dyaw**2 + e["pitch"]**2 + e["roll"]**2)
        score = pos_d + 0.05 * ang_d
        if score < best_score:
            best_score, best_idx = score, i
    src = db[best_idx]
    results["2_find_best"] = time.perf_counter() - t0

    # ---- 3. Extract frame via ffmpeg -ss (fast keyframe seek) ----
    t0 = time.perf_counter()
    ts = src["timestamp_s"]
    h = int(ts // 3600)
    m = int((ts % 3600) // 60)
    s = ts % 60
    ss = f"{h:02d}:{m:02d}:{s:06.3f}"

    result = subprocess.run(
        [
            "ffmpeg", "-ss", ss, "-i", "/data/video.MP4",
            "-frames:v", "1", "-q:v", "2", "-f", "image2", "-c:v", "mjpeg",
            "pipe:1",
        ],
        capture_output=True,
    )
    jpeg_bytes = result.stdout
    results["3_ffmpeg_extract"] = time.perf_counter() - t0

    # ---- Print results ----
    total = sum(results.values())
    print(f"\n{'='*55}")
    print(f"{'Step':<30} {'Time':>10}")
    print(f"{'='*55}")
    for k, v in results.items():
        bar = '#' * int(v / total * 40) if total > 0 else ''
        print(f"  {k:<28} {v:>8.4f}s  {bar}")
    print(f"{'='*55}")
    print(f"  {'TOTAL':<28} {total:>8.4f}s")
    print(f"  {'JPEG size':<28} {len(jpeg_bytes):>8} bytes")
    print(f"{'='*55}")

    return results


@app.local_entrypoint()
def main():
    results = bench.remote()
