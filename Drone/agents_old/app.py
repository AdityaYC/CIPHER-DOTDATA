"""Drone Vision – visual search agent for building exploration.

Usage (from project root):
    modal serve agents/app.py   # dev mode (hot-reload)
    modal deploy agents/app.py  # production
"""

import csv
import math
from pathlib import Path

import modal
import numpy as np

# ---------------------------------------------------------------------------
# Paths (resolved relative to this file)
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_PROJECT = _HERE.parent
_DEPTH_PRO = _PROJECT / "ml-depth-pro"
_DATA = _PROJECT / "data"

# ---------------------------------------------------------------------------
# Modal app + container image
# ---------------------------------------------------------------------------
app = modal.App("vision-old")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0", "ffmpeg")
    .pip_install(
        "torch",
        "torchvision",
        "timm",
        "numpy<2",
        "pillow_heif",
        "matplotlib",
        "opencv-python-headless",
        "pillow",
        "fastapi[standard]",
    )
    # depth_pro package
    .add_local_dir(
        str(_DEPTH_PRO / "src" / "depth_pro"),
        "/root/src/depth_pro",
        copy=True,
    )
    .add_local_file(
        str(_DEPTH_PRO / "pyproject.toml"),
        "/root/pyproject.toml",
        copy=True,
    )
    .run_commands("cd /root && pip install -e .")
    # Bake data files — large/stable files first, small/changing files last
    .add_local_file(
        str(_DATA / "video.MP4"),
        "/data/video.MP4",
        copy=True,
    )
    .add_local_file(
        str(_DATA / "gyro.csv"),
        "/data/gyro.csv",
        copy=True,
    )
    .add_local_file(
        str(_DATA / "timestamp_coordinates.csv"),
        "/data/timestamp_coordinates.csv",
        copy=True,
    )
)

checkpoint_vol = modal.Volume.from_name(
    "depth-pro-checkpoints", create_if_missing=True
)
CKPT_PATH = "/checkpoints/depth_pro.pt"

# ---------------------------------------------------------------------------
# Geometry helpers (same as view_transform_modal.py)
# ---------------------------------------------------------------------------


def rot_x(deg: float) -> np.ndarray:
    th = math.radians(deg)
    c, s = math.cos(th), math.sin(th)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float32)


def rot_y(deg: float) -> np.ndarray:
    th = math.radians(deg)
    c, s = math.cos(th), math.sin(th)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)


def rot_z(deg: float) -> np.ndarray:
    th = math.radians(deg)
    c, s = math.cos(th), math.sin(th)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)


def build_K(W: int, H: int, f_px: float) -> np.ndarray:
    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    return np.array(
        [[f_px, 0, cx], [0, f_px, cy], [0, 0, 1]], dtype=np.float32
    )


def angle_diff(a: float, b: float) -> float:
    """Shortest signed angular difference (a - b), in [-180, 180]."""
    d = (a - b) % 360
    return d - 360 if d > 180 else d


def reproject_novel_view(
    I_bgr: np.ndarray,
    D: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    inpaint_radius: int = 3,
):
    """Reproject source image to a novel viewpoint using GPU acceleration."""
    import cv2
    import torch

    H, W = D.shape
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Move matrices to GPU
    K_t = torch.from_numpy(K).to(device)
    Kinv_t = torch.inverse(K_t)
    R_t = torch.from_numpy(R).to(device)
    t_t = torch.from_numpy(t).to(device).reshape(1, 3)

    # Build pixel grid on GPU
    uu, vv = torch.meshgrid(
        torch.arange(W, dtype=torch.float32, device=device),
        torch.arange(H, dtype=torch.float32, device=device),
        indexing="xy",
    )
    pix = torch.stack([uu, vv, torch.ones_like(uu)], dim=-1).reshape(-1, 3)
    z = torch.from_numpy(D.astype(np.float32)).to(device).reshape(-1)

    valid0 = torch.isfinite(z) & (z > 1e-6)
    idx0 = torch.where(valid0)[0]
    pix0, z0 = pix[valid0], z[valid0]

    # Back-project, transform, forward-project — all on GPU
    X = (pix0 @ Kinv_t.T) * z0[:, None]
    Xp = (X - t_t) @ R_t.T
    zp = Xp[:, 2]
    valid1 = zp > 1e-6
    Xp, zp, src_idx = Xp[valid1], zp[valid1], idx0[valid1]

    proj = Xp @ K_t.T
    up = proj[:, 0] / (proj[:, 2] + 1e-8)
    vp = proj[:, 1] / (proj[:, 2] + 1e-8)
    ui = torch.round(up).to(torch.int64)
    vi = torch.round(vp).to(torch.int64)

    valid2 = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    ui, vi, zp, src_idx = ui[valid2], vi[valid2], zp[valid2], src_idx[valid2]

    lin = vi * W + ui

    # Z-buffer resolve: sort by depth, first occurrence per pixel = closest
    order = torch.argsort(zp)
    lin_s = lin[order]
    src_s = src_idx[order]
    mask = torch.ones(lin_s.shape[0], dtype=torch.bool, device=device)
    mask[1:] = lin_s[1:] != lin_s[:-1]
    unique_lin = lin_s[mask].cpu().numpy()
    chosen_src = src_s[mask].cpu().numpy()

    # Scatter to output (CPU — needed for cv2.inpaint anyway)
    out = np.zeros_like(I_bgr)
    filled = np.zeros(H * W, dtype=bool)
    out.reshape(-1, 3)[unique_lin] = I_bgr.reshape(-1, 3)[chosen_src]
    filled[unique_lin] = True

    holes = (~filled).reshape(H, W).astype(np.uint8) * 255
    if inpaint_radius > 0:
        out = cv2.inpaint(
            out, holes, inpaintRadius=inpaint_radius, flags=cv2.INPAINT_TELEA
        )
    return out, holes


# ---------------------------------------------------------------------------
# Image database loader (runs inside the container)
# ---------------------------------------------------------------------------


def _load_database() -> list[dict]:
    """Build trajectory database by merging timestamp_coordinates.csv and gyro.csv.

    timestamp_coordinates.csv provides sparse (x, y, z) waypoints at key timestamps.
    gyro.csv provides dense orientation (yaw, pitch, roll) at ~1ms intervals.

    For each gyro row whose timestamp falls within the range of the coordinate
    waypoints, we linearly interpolate x/y/z between the two surrounding
    waypoints and combine with the gyro orientation.

    Returns a list of dicts with keys: x, y, z, yaw, pitch, roll, timestamp_s.
    """
    import bisect

    # 1. Load sparse position waypoints
    waypoint_ts: list[float] = []
    waypoint_x: list[float] = []
    waypoint_y: list[float] = []
    waypoint_z: list[float] = []
    with open("/data/timestamp_coordinates.csv") as f:
        for row in csv.DictReader(f):
            waypoint_ts.append(float(row["timestamp"]))
            waypoint_x.append(float(row["x"]))
            waypoint_y.append(float(row["y"]))
            waypoint_z.append(float(row["z"]))

    wp_min, wp_max = waypoint_ts[0], waypoint_ts[-1]

    # 2. Load gyro orientation data, sampling every ~10ms
    SAMPLE_INTERVAL = 0.010  # 10 ms
    next_sample_t = wp_min
    traj: list[dict] = []
    with open("/data/gyro.csv") as f:
        for row in csv.DictReader(f):
            t = float(row["timestamp_s"])

            if t < wp_min or t > wp_max:
                continue
            if t < next_sample_t:
                continue
            next_sample_t = t + SAMPLE_INTERVAL

            # Find surrounding waypoints for interpolation
            i = bisect.bisect_right(waypoint_ts, t) - 1
            i = max(0, min(i, len(waypoint_ts) - 2))

            t0, t1 = waypoint_ts[i], waypoint_ts[i + 1]
            dt = t1 - t0
            alpha = (t - t0) / dt if dt > 0 else 0.0
            alpha = max(0.0, min(1.0, alpha))

            x = waypoint_x[i] + alpha * (waypoint_x[i + 1] - waypoint_x[i])
            y = waypoint_y[i] + alpha * (waypoint_y[i + 1] - waypoint_y[i])
            z = waypoint_z[i] + alpha * (waypoint_z[i + 1] - waypoint_z[i])

            traj.append({
                "timestamp_s": t,
                "x": x,
                "y": y,
                "z": z,
                "yaw": float(row["yaw_deg"]),
                "pitch": float(row["pitch_deg"]),
                "roll": float(row["roll_deg"]),
            })

    return traj


def _extract_frame_jpeg(timestamp_s: float, video_path: str = "/data/video.MP4") -> bytes:
    """Extract a single frame as JPEG bytes using ffmpeg -ss (fast keyframe seek).

    Returns raw JPEG bytes.
    """
    import subprocess

    h = int(timestamp_s // 3600)
    m = int((timestamp_s % 3600) // 60)
    s = timestamp_s % 60
    ss = f"{h:02d}:{m:02d}:{s:06.3f}"

    result = subprocess.run(
        [
            "ffmpeg", "-ss", ss, "-i", video_path,
            "-frames:v", "1", "-q:v", "2", "-f", "image2", "-c:v", "mjpeg",
            "pipe:1",
        ],
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed at {ss}: {result.stderr.decode()}")
    return result.stdout


# Keep old cv2-based extraction for reference
# def _extract_frame(cap, timestamp_s: float) -> np.ndarray:
#     """Extract a single frame from the video at the given timestamp.
#     Returns the frame as a BGR numpy array.
#     """
#     import cv2
#     cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_s * 1000)
#     ret, frame = cap.read()
#     if not ret:
#         raise RuntimeError(f"Failed to read frame at {timestamp_s:.3f}s from video")
#     return frame


# ---------------------------------------------------------------------------
# GetImage – persistent container class
# ---------------------------------------------------------------------------


# GPU + volume commented out while depth/reprojection is disabled
@app.cls(
    image=image,
    # gpu="H200",
    # volumes={"/checkpoints": checkpoint_vol},
    timeout=600,
    scaledown_window=300,
)
class GetImage:
    """Persistent container hosting the DepthPro model and image database.

    Image selection criteria
    ========================
    Given a query pose (x, y, z, yaw) with pitch=0 and roll=0, the best
    source image is chosen by a linear scan over all 797 images.  Each
    image is scored as:

        score = pos_distance + 0.05 * ang_distance

    where pos_distance is the Euclidean distance between positions and
    ang_distance combines yaw, pitch, and roll differences.  The image
    with the lowest score is selected.

    After selecting the source image, the endpoint:
      - Runs DepthPro to estimate a dense depth map for I.
      - Computes the exact relative rotation R and translation t between
        the source camera pose and the query pose.
      - Calls reproject_novel_view(I, depth, K, R, t) to warp I into the
        requested viewpoint, with inpainting to fill disoccluded holes.
    """

    @modal.enter()
    def setup(self):
        # import cv2
        # import torch
        #
        # import depth_pro
        # from depth_pro.depth_pro import (
        #     DEFAULT_MONODEPTH_CONFIG_DICT,
        #     DepthProConfig,
        # )
        # # Load DepthPro (once per container lifetime)
        # config = DepthProConfig(
        #     patch_encoder_preset=DEFAULT_MONODEPTH_CONFIG_DICT.patch_encoder_preset,
        #     image_encoder_preset=DEFAULT_MONODEPTH_CONFIG_DICT.image_encoder_preset,
        #     decoder_features=DEFAULT_MONODEPTH_CONFIG_DICT.decoder_features,
        #     use_fov_head=DEFAULT_MONODEPTH_CONFIG_DICT.use_fov_head,
        #     fov_encoder_preset=DEFAULT_MONODEPTH_CONFIG_DICT.fov_encoder_preset,
        #     checkpoint_uri=CKPT_PATH,
        # )
        # self.model, self.transform = depth_pro.create_model_and_transforms(
        #     config=config
        # )
        # self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # self.model = self.model.to(self.device).half().eval()

        # # Open video capture (kept open for the container lifetime)
        # self.cap = cv2.VideoCapture("/data/video.MP4")
        # if not self.cap.isOpened():
        #     raise RuntimeError("Failed to open /data/video.MP4")

        # Build image database
        self.db = _load_database()

    def _find_best(self, x: float, y: float, z: float, yaw: float) -> int:
        """Return index of the best matching source image (linear scan)."""
        best_score, best_idx = float("inf"), 0
        best_pos_distance, best_ang_distance = 0.0, 0.0
        for i, e in enumerate(self.db):
            dx = e["x"] - x
            dy = e["y"] - y
            dz = e["z"] - z
            pos_distance = math.sqrt(dx**2 + dy**2 + dz**2)
            dyaw = abs(angle_diff(e["yaw"], yaw))
            dpitch = abs(e["pitch"])
            droll = abs(e["roll"])
            ang_distance = math.sqrt(dyaw**2 + dpitch**2 + droll**2)
            score = pos_distance + 0.05 * ang_distance
            if i % 50 == 0:
                print(f"  [{i}] t={e['timestamp_s']:.3f}s  pos_dist={pos_distance:.6f}  ang_dist={ang_distance:.6f}  score={score:.6f}")
            if score < best_score:
                best_score = score
                best_idx = i
                best_pos_distance = pos_distance
                best_ang_distance = ang_distance
        print(f"  Best: [{best_idx}] pos_dist={best_pos_distance:.6f}  ang_dist={best_ang_distance:.6f}  score={best_score:.6f}")
        return best_idx

    @modal.fastapi_endpoint()
    def getImage(self, x: float, y: float, z: float, yaw: float):
        """Return the closest recorded frame to (x, y, z, yaw).

        Returns JSON with a base64-encoded JPEG.
        """
        import base64

        # 1. Pick the best source image
        idx = self._find_best(x, y, z, yaw)
        src = self.db[idx]
        print(f"Selected source frame at t={src['timestamp_s']:.3f}s")

        # 2. Extract frame via ffmpeg (fast keyframe seek)
        jpeg_bytes = _extract_frame_jpeg(src["timestamp_s"])

        # # --- Depth estimation + reprojection (commented out for speed) ---
        # import cv2
        # import torch
        #
        # frame_bgr = cv2.imdecode(
        #     np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR
        # )
        # img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        #
        # # 3. Depth estimation
        # image_t = self.transform(img_rgb).half().to(self.device)
        # with torch.no_grad():
        #     pred = self.model.infer(image_t, f_px=None)
        # depth_m = pred["depth"].detach().float().cpu().numpy()
        # focal = float(pred["focallength_px"])
        #
        # # 4. Relative pose: source → target
        # R_src = rot_y(src["yaw"]) @ rot_x(src["pitch"]) @ rot_z(src["roll"])
        # R_tgt = rot_y(yaw)
        # R = R_tgt @ R_src.T
        # dp = np.array(
        #     [x - src["x"], y - src["y"], z - src["z"]], dtype=np.float32
        # )
        # t = dp @ R_src.T
        #
        # # 5. Reproject to novel view
        # H, W = depth_m.shape
        # K = build_K(W, H, focal)
        # out_bgr, _ = reproject_novel_view(frame_bgr, depth_m, K, R, t)
        #
        # _, buf = cv2.imencode(".png", out_bgr)
        # jpeg_bytes = buf.tobytes()
        # --- End depth estimation + reprojection ---

        b64 = base64.b64encode(jpeg_bytes).decode("ascii")
        return {"image_base64": b64}

    @modal.method()
    def getImageRemote(
        self, x: float, y: float, z: float, yaw: float
    ) -> dict:
        """Return the closest recorded frame to (x, y, z, yaw).

        Returns: {"image_jpeg": bytes, "source_idx": int, "source_timestamp_s": float}
        """
        idx = self._find_best(x, y, z, yaw)
        src = self.db[idx]
        print(f"Selected source frame at t={src['timestamp_s']:.3f}s")

        # Extract frame via ffmpeg (fast keyframe seek)
        jpeg_bytes = _extract_frame_jpeg(src["timestamp_s"])

        # # --- Depth estimation + reprojection (commented out for speed) ---
        # import cv2
        # import torch
        #
        # frame_bgr = cv2.imdecode(
        #     np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR
        # )
        # img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        #
        # image_t = self.transform(img_rgb).half().to(self.device)
        # with torch.no_grad():
        #     pred = self.model.infer(image_t, f_px=None)
        # depth_m = pred["depth"].detach().float().cpu().numpy()
        # focal = float(pred["focallength_px"])
        #
        # R_src = rot_y(src["yaw"]) @ rot_x(src["pitch"]) @ rot_z(src["roll"])
        # R_tgt = rot_y(yaw)
        # R = R_tgt @ R_src.T
        # dp = np.array(
        #     [x - src["x"], y - src["y"], z - src["z"]], dtype=np.float32
        # )
        # t = dp @ R_src.T
        #
        # H, W = depth_m.shape
        # K = build_K(W, H, focal)
        # out_bgr, _ = reproject_novel_view(frame_bgr, depth_m, K, R, t)
        #
        # _, buf = cv2.imencode(".png", out_bgr)
        # jpeg_bytes = buf.tobytes()
        # --- End depth estimation + reprojection ---

        return {
            "image_jpeg": jpeg_bytes,
            "source_idx": idx,
            "source_timestamp_s": src["timestamp_s"],
        }
