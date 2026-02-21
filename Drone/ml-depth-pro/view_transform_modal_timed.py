#!/usr/bin/env python3
"""Run view_transform on Modal with a GPU — WITH TIMING INSTRUMENTATION."""

import math
from pathlib import Path

import modal

app = modal.App("depth-pro-view-transform")

# Build a container image with all dependencies + the depth_pro source package.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")  # needed by cv2
    .pip_install(
        "torch",
        "torchvision",
        "timm",
        "numpy<2",
        "pillow_heif",
        "matplotlib",
        "opencv-python-headless",
        "pillow",
    )
    # Install the depth_pro package from the local repo source.
    .add_local_dir("src/depth_pro", "/root/src/depth_pro", copy=True)
    .add_local_file("pyproject.toml", "/root/pyproject.toml", copy=True)
    .run_commands("cd /root && pip install -e .")
)

# Persist the model checkpoint across runs so it only uploads once.
volume = modal.Volume.from_name("depth-pro-checkpoints", create_if_missing=True)
CHECKPOINT_REMOTE_PATH = "/checkpoints/depth_pro.pt"

# ---------------------------------------------------------------------------
# Pure-Python helpers (same as original, no imports needed at module level)
# ---------------------------------------------------------------------------

import numpy as np


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
    cx = (W - 1) / 2.0
    cy = (H - 1) / 2.0
    return np.array(
        [[f_px, 0.0, cx], [0.0, f_px, cy], [0.0, 0.0, 1.0]], dtype=np.float32
    )


def depth_to_vis(depth_m: np.ndarray) -> np.ndarray:
    import cv2

    d = np.clip(depth_m.copy(), 1e-6, None)
    inv = 1.0 / d
    lo, hi = np.percentile(inv, [2, 98])
    inv = np.clip(inv, lo, hi)
    inv = (inv - lo) / (hi - lo + 1e-8)
    vis = (inv * 255.0).astype(np.uint8)
    return cv2.applyColorMap(vis, cv2.COLORMAP_TURBO)


def reproject_novel_view(
    I_bgr: np.ndarray,
    D: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    inpaint_radius: int = 3,
):
    import cv2

    H, W = D.shape
    Kinv = np.linalg.inv(K).astype(np.float32)

    uu, vv = np.meshgrid(
        np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32)
    )
    ones = np.ones_like(uu, dtype=np.float32)

    pix = np.stack([uu, vv, ones], axis=-1).reshape(-1, 3)
    z = D.reshape(-1).astype(np.float32)

    valid0 = np.isfinite(z) & (z > 1e-6)
    idx0 = np.where(valid0)[0]
    pix0 = pix[valid0]
    z0 = z[valid0]

    X = (pix0 @ Kinv.T) * z0[:, None]
    Xp = (X @ R.T) + t.reshape(1, 3)
    zp = Xp[:, 2]
    valid1 = zp > 1e-6
    Xp = Xp[valid1]
    zp = zp[valid1]
    src_idx = idx0[valid1]

    proj = Xp @ K.T
    up = proj[:, 0] / (proj[:, 2] + 1e-8)
    vp = proj[:, 1] / (proj[:, 2] + 1e-8)

    ui = np.round(up).astype(np.int32)
    vi = np.round(vp).astype(np.int32)

    valid2 = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    ui, vi, zp, src_idx = ui[valid2], vi[valid2], zp[valid2], src_idx[valid2]

    lin = vi * W + ui
    order = np.argsort(zp)
    lin_s = lin[order]
    src_s = src_idx[order]
    unique_lin, first_pos = np.unique(lin_s, return_index=True)
    chosen_src = src_s[first_pos]

    out = np.zeros_like(I_bgr)
    filled = np.zeros((H * W,), dtype=bool)
    out_flat = out.reshape(-1, 3)
    I_flat = I_bgr.reshape(-1, 3)
    out_flat[unique_lin] = I_flat[chosen_src]
    filled[unique_lin] = True

    holes = (~filled).reshape(H, W).astype(np.uint8) * 255
    if inpaint_radius > 0:
        out = cv2.inpaint(out, holes, inpaintRadius=inpaint_radius, flags=cv2.INPAINT_TELEA)

    return out, holes


# ---------------------------------------------------------------------------
# Modal remote function
# ---------------------------------------------------------------------------


@app.function(
    image=image,
    gpu="H200",
    volumes={"/checkpoints": volume},
    timeout=600,
)
def run_view_transform(
    input_image_bytes: bytes,
    yaw_deg: float = 10.0,
    pitch_deg: float = 0.0,
    roll_deg: float = 0.0,
    back_ft: float = 2.0,
    right_ft: float = 0.0,
    down_ft: float = 0.0,
    inpaint_radius: int = 3,
    save_depth_vis: bool = False,
    save_holes: bool = False,
) -> dict:
    """Run depth estimation + novel-view reprojection on Modal.

    Returns a dict with keys: "output" (PNG bytes), and optionally
    "depth_vis" and "holes" (also PNG bytes).
    """
    import io
    import time

    import cv2
    import torch
    from PIL import Image

    t0 = time.perf_counter()

    import depth_pro

    t_import = time.perf_counter()
    print(f"[TIMING] imports: {t_import - t0:.2f}s")

    # ---- Load model ----
    from depth_pro.depth_pro import DEFAULT_MONODEPTH_CONFIG_DICT, DepthProConfig

    config = DepthProConfig(
        patch_encoder_preset=DEFAULT_MONODEPTH_CONFIG_DICT.patch_encoder_preset,
        image_encoder_preset=DEFAULT_MONODEPTH_CONFIG_DICT.image_encoder_preset,
        decoder_features=DEFAULT_MONODEPTH_CONFIG_DICT.decoder_features,
        use_fov_head=DEFAULT_MONODEPTH_CONFIG_DICT.use_fov_head,
        fov_encoder_preset=DEFAULT_MONODEPTH_CONFIG_DICT.fov_encoder_preset,
        checkpoint_uri=CHECKPOINT_REMOTE_PATH,
    )
    model, transform = depth_pro.create_model_and_transforms(config=config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).half().eval()
    t_model = time.perf_counter()
    print(f"[TIMING] model load: {t_model - t_import:.2f}s")

    # ---- Decode input image ----
    pil_img = Image.open(io.BytesIO(input_image_bytes)).convert("RGB")
    img_np = np.array(pil_img)

    # depth_pro.load_rgb expects a file path; write to a temp file.
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        pil_img.save(tmp, format="PNG")
        tmp_path = tmp.name

    image, _, f_px = depth_pro.load_rgb(tmp_path)
    image_t = transform(image).half().to(device)
    t_preprocess = time.perf_counter()
    print(f"[TIMING] preprocess: {t_preprocess - t_model:.2f}s")

    # ---- Inference ----
    with torch.no_grad():
        pred = model.infer(image_t, f_px=f_px)
    depth = pred["depth"]
    focallength_px = float(pred["focallength_px"])

    if isinstance(depth, torch.Tensor):
        depth_m = depth.detach().float().cpu().numpy()
    else:
        depth_m = np.asarray(depth, dtype=np.float32)
    t_inference = time.perf_counter()
    print(f"[TIMING] inference: {t_inference - t_preprocess:.2f}s")

    # ---- Reproject ----
    I_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    H, W = depth_m.shape
    K = build_K(W, H, focallength_px)

    R = rot_y(yaw_deg) @ rot_x(pitch_deg) @ rot_z(roll_deg)
    ft2m = 0.3048
    t = np.array([right_ft * ft2m, down_ft * ft2m, back_ft * ft2m], dtype=np.float32)

    out_bgr, holes_mask = reproject_novel_view(
        I_bgr=I_bgr, D=depth_m, K=K, R=R, t=t, inpaint_radius=inpaint_radius
    )

    t_reproject = time.perf_counter()
    print(f"[TIMING] reproject: {t_reproject - t_inference:.2f}s")

    # ---- Encode results ----
    result: dict = {}
    _, buf = cv2.imencode(".png", out_bgr)
    result["output"] = buf.tobytes()
    result["focallength_px"] = focallength_px

    if save_depth_vis:
        _, buf = cv2.imencode(".png", depth_to_vis(depth_m))
        result["depth_vis"] = buf.tobytes()

    if save_holes:
        _, buf = cv2.imencode(".png", holes_mask)
        result["holes"] = buf.tobytes()

    t_encode = time.perf_counter()
    print(f"[TIMING] encode: {t_encode - t_reproject:.2f}s")
    print(f"[TIMING] TOTAL (in-container): {t_encode - t0:.2f}s")

    return result


# ---------------------------------------------------------------------------
# Local entrypoint – mirrors the original CLI interface
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def main(
    input: str,
    output: str,
    yaw_deg: float = 10.0,
    pitch_deg: float = 0.0,
    roll_deg: float = 0.0,
    back_ft: float = 2.0,
    right_ft: float = 0.0,
    down_ft: float = 0.0,
    inpaint_radius: int = 3,
    save_depth_vis: bool = False,
    save_holes: bool = False,
    upload_checkpoint: bool = False,
):
    """Local CLI that sends the image to Modal for processing.

    First run: pass --upload-checkpoint to push checkpoints/depth_pro.pt to
    the Modal volume. Subsequent runs skip the upload.

    Usage:
        modal run view_transform_modal_timed.py --input input.png --output out.png
        modal run view_transform_modal_timed.py --input input.png --output out.png --yaw-deg 15 --back-ft 3
    """
    import time as _time

    inp = Path(input)
    outp = Path(output)
    outp.parent.mkdir(parents=True, exist_ok=True)

    # Optionally upload checkpoint on first use.
    if upload_checkpoint:
        local_ckpt = Path("checkpoints/depth_pro.pt")
        if not local_ckpt.exists():
            raise FileNotFoundError(f"Checkpoint not found at {local_ckpt}")
        print(f"Uploading {local_ckpt} to Modal volume …")
        import subprocess
        subprocess.run(
            ["modal", "volume", "put", "depth-pro-checkpoints",
             str(local_ckpt), "depth_pro.pt"],
            check=True,
        )
        print("Upload complete.")

    input_bytes = inp.read_bytes()

    print("Sending to Modal …")
    _t_start = _time.perf_counter()
    result = run_view_transform.remote(
        input_image_bytes=input_bytes,
        yaw_deg=yaw_deg,
        pitch_deg=pitch_deg,
        roll_deg=roll_deg,
        back_ft=back_ft,
        right_ft=right_ft,
        down_ft=down_ft,
        inpaint_radius=inpaint_radius,
        save_depth_vis=save_depth_vis,
        save_holes=save_holes,
    )

    _t_end = _time.perf_counter()
    print(f"[TIMING] local: modal round-trip: {_t_end - _t_start:.2f}s")

    outp.write_bytes(result["output"])
    print(f"Wrote: {outp}")
    print(f"DepthPro f_px estimate: {result['focallength_px']}")

    if save_depth_vis and "depth_vis" in result:
        p = outp.with_suffix(".depth_vis.png")
        p.write_bytes(result["depth_vis"])
        print(f"Wrote: {p}")

    if save_holes and "holes" in result:
        p = outp.with_suffix(".holes.png")
        p.write_bytes(result["holes"])
        print(f"Wrote: {p}")
