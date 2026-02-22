"""Drone Vision – image server for building exploration.

Pre-extracted JPEG frames (every 0.1s) are baked into the container image.
getImage returns the closest frame to a query (x, y, z, yaw) along with
the actual position metadata and which cardinal directions are navigable.

Usage:
    modal serve agents/app.py
    modal deploy agents/app.py
"""

import bisect
import csv
import math
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_PROJECT = _HERE.parent
_DATA = _PROJECT / "data"

# ---------------------------------------------------------------------------
# Modal app + container image
# ---------------------------------------------------------------------------
app = modal.App("drone-vision")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("fastapi[standard]")
    # Pre-extracted frames (large, stable) first for layer caching
    .add_local_dir(
        str(_DATA / "image_samples"),
        "/data/image_samples",
        copy=True,
    )
    # Small CSV files last (change often)
    .add_local_file(str(_DATA / "gyro.csv"), "/data/gyro.csv", copy=True)
    .add_local_file(
        str(_DATA / "timestamp_coordinates.csv"),
        "/data/timestamp_coordinates.csv",
        copy=True,
    )
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FRAME_INTERVAL = 0.1  # seconds between pre-extracted frames
STEP_SIZE = 0.1       # meters per movement step
ALLOWED_THRESHOLD = 0.15  # max distance to consider a direction navigable

# Yaw=0 → facing +x.  Adjust this table if the coordinate convention differs.
# Maps (yaw) → {direction: (dx, dy)}
_DIRECTION_OFFSETS = {
    0:   {"forward": (1, 0),  "backward": (-1, 0),  "left": (0, -1), "right": (0, 1)},
    90:  {"forward": (0, 1),  "backward": (0, -1),  "left": (1, 0),  "right": (-1, 0)},
    180: {"forward": (-1, 0), "backward": (1, 0),   "left": (0, 1),  "right": (0, -1)},
    270: {"forward": (0, -1), "backward": (0, 1),   "left": (-1, 0), "right": (1, 0)},
}


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_waypoints() -> tuple[list[float], list[float], list[float], list[float]]:
    """Load sparse (timestamp, x, y, z) waypoints from CSV."""
    ts, xs, ys, zs = [], [], [], []
    with open("/data/timestamp_coordinates.csv") as f:
        for row in csv.DictReader(f):
            ts.append(float(row["timestamp"]))
            xs.append(float(row["x"]))
            ys.append(float(row["y"]))
            zs.append(float(row["z"]))
    return ts, xs, ys, zs


def _load_gyro() -> tuple[list[float], list[float], list[float], list[float]]:
    """Load dense gyro data. Returns parallel lists: (timestamps, yaws, pitches, rolls)."""
    ts, yaws, pitches, rolls = [], [], [], []
    with open("/data/gyro.csv") as f:
        for row in csv.DictReader(f):
            ts.append(float(row["timestamp_s"]))
            yaws.append(float(row["yaw_deg"]))
            pitches.append(float(row["pitch_deg"]))
            rolls.append(float(row["roll_deg"]))
    return ts, yaws, pitches, rolls


def _interpolate_position(
    t: float,
    wp_ts: list[float],
    wp_xs: list[float],
    wp_ys: list[float],
    wp_zs: list[float],
) -> tuple[float, float, float]:
    """Linearly interpolate x/y/z at time t from sparse waypoints."""
    i = bisect.bisect_right(wp_ts, t) - 1
    i = max(0, min(i, len(wp_ts) - 2))
    t0, t1 = wp_ts[i], wp_ts[i + 1]
    dt = t1 - t0
    alpha = (t - t0) / dt if dt > 0 else 0.0
    alpha = max(0.0, min(1.0, alpha))
    x = wp_xs[i] + alpha * (wp_xs[i + 1] - wp_xs[i])
    y = wp_ys[i] + alpha * (wp_ys[i + 1] - wp_ys[i])
    z = wp_zs[i] + alpha * (wp_zs[i + 1] - wp_zs[i])
    return x, y, z


def _closest_gyro(
    t: float,
    gyro_ts: list[float],
    gyro_yaws: list[float],
    gyro_pitches: list[float],
    gyro_rolls: list[float],
) -> tuple[float, float, float]:
    """Find the gyro reading closest to time t."""
    i = bisect.bisect_right(gyro_ts, t) - 1
    i = max(0, min(i, len(gyro_ts) - 1))
    if i + 1 < len(gyro_ts) and abs(gyro_ts[i + 1] - t) < abs(gyro_ts[i] - t):
        i += 1
    return gyro_yaws[i], gyro_pitches[i], gyro_rolls[i]


def _angle_diff(a: float, b: float) -> float:
    """Shortest angular difference in [-180, 180]."""
    d = (a - b) % 360
    return d - 360 if d > 180 else d


# ---------------------------------------------------------------------------
# GetImage – persistent container class
# ---------------------------------------------------------------------------


@app.cls(image=image, timeout=600, scaledown_window=300)
class ImageServer:
    """Loads pre-extracted frames from disk. getImage is a fast in-memory lookup."""

    @modal.enter()
    def setup(self):
        print("Loading waypoints and gyro data...")
        wp_ts, wp_xs, wp_ys, wp_zs = _load_waypoints()
        gyro_ts, gyro_yaws, gyro_pitches, gyro_rolls = _load_gyro()

        max_t = wp_ts[-1]
        print(f"Trajectory spans 0 – {max_t:.1f}s")

        # Load pre-extracted JPEG frames from disk
        print("Loading pre-extracted frames...")
        frame_dir = Path("/data/image_samples")
        frame_files = sorted(frame_dir.glob("frame_*.jpg"))
        print(f"Found {len(frame_files)} frame files")

        # Build database: one entry per frame
        self.db: list[dict] = []
        for i, fpath in enumerate(frame_files):
            t = i * FRAME_INTERVAL
            if t > max_t:
                break
            x, y, z = _interpolate_position(t, wp_ts, wp_xs, wp_ys, wp_zs)
            yaw, pitch, roll = _closest_gyro(
                t, gyro_ts, gyro_yaws, gyro_pitches, gyro_rolls
            )
            self.db.append({
                "t": t,
                "x": x,
                "y": y,
                "z": z,
                "yaw": yaw,
                "pitch": pitch,
                "roll": roll,
                "filename": fpath.name,
                "jpeg": fpath.read_bytes(),
            })

        print(f"Database ready: {len(self.db)} entries")
        if self.db:
            first, last = self.db[0], self.db[-1]
            print(f"  First: t={first['t']:.1f}s pos=({first['x']:.1f}, {first['y']:.1f}, {first['z']:.1f}) yaw={first['yaw']:.1f}")
            print(f"  Last:  t={last['t']:.1f}s pos=({last['x']:.1f}, {last['y']:.1f}, {last['z']:.1f}) yaw={last['yaw']:.1f}")

    def _find_best(self, x: float, y: float, z: float, yaw: float) -> int:
        """Return index of the closest matching frame."""
        best_score, best_idx = float("inf"), 0
        for i, e in enumerate(self.db):
            dx = e["x"] - x
            dy = e["y"] - y
            dz = e["z"] - z
            pos_dist = math.sqrt(dx**2 + dy**2 + dz**2)
            dyaw = abs(_angle_diff(e["yaw"], yaw))
            ang_dist = dyaw
            score = pos_dist + 0.05 * ang_dist
            if score < best_score:
                best_score = score
                best_idx = i
        return best_idx

    def _check_allowed(self, x: float, y: float, z: float, yaw: float) -> dict:
        """Check which movement and turn directions are available.

        Movement (forward/backward/left/right): checks if a DB entry exists
        within ALLOWED_THRESHOLD of the target position after a STEP_SIZE move.

        Turns (turnLeft/turnRight): checks if a DB entry exists at the same
        position with the rotated yaw (±90°).
        """
        yaw_key = int(round(yaw)) % 360
        offsets = _DIRECTION_OFFSETS.get(yaw_key, _DIRECTION_OFFSETS[0])

        allowed = {}
        # Movement directions — require both position AND yaw match
        for direction, (dx, dy) in offsets.items():
            tx = x + dx * STEP_SIZE
            ty = y + dy * STEP_SIZE
            found = False
            for e in self.db:
                dist = math.sqrt((e["x"] - tx)**2 + (e["y"] - ty)**2 + (e["z"] - z)**2)
                if dist < ALLOWED_THRESHOLD:
                    yaw_diff = abs(_angle_diff(e["yaw"], yaw_key))
                    if yaw_diff < 45:
                        found = True
                        break
            allowed[direction] = found

        # Turn directions: same position, rotated yaw
        for turn_name, turn_yaw in [("turnLeft", (yaw_key - 90) % 360),
                                     ("turnRight", (yaw_key + 90) % 360)]:
            found = False
            for e in self.db:
                pos_dist = math.sqrt((e["x"] - x)**2 + (e["y"] - y)**2 + (e["z"] - z)**2)
                if pos_dist < ALLOWED_THRESHOLD:
                    yaw_diff = abs(_angle_diff(e["yaw"], turn_yaw))
                    if yaw_diff < 45:
                        found = True
                        break
            allowed[turn_name] = found

        return allowed

    @modal.fastapi_endpoint()
    def getImage(self, x: float, y: float, z: float, yaw: float):
        """Return the closest frame and navigation metadata.

        Returns JSON:
        {
            "image": <base64 JPEG>,
            "x": <actual x>,
            "y": <actual y>,
            "z": <actual z>,
            "yaw": <actual yaw>,
            "allowed": {"forward": bool, "backward": bool, "left": bool, "right": bool}
        }
        """
        import base64

        idx = self._find_best(x, y, z, yaw)
        src = self.db[idx]
        allowed = self._check_allowed(src["x"], src["y"], src["z"], yaw)
        b64 = base64.b64encode(src["jpeg"]).decode("ascii")
        return {
            "image": b64,
            "x": src["x"],
            "y": src["y"],
            "z": src["z"],
            "yaw": src["yaw"],
            "filename": src["filename"],
            "allowed": allowed,
        }

    @modal.method()
    def getImageRemote(
        self, x: float, y: float, z: float, yaw: float
    ) -> dict:
        """Same as getImage but for programmatic (Modal-to-Modal) callers.

        Returns the same fields as getImage, with "image" as raw JPEG bytes.
        """
        idx = self._find_best(x, y, z, yaw)
        src = self.db[idx]
        allowed = self._check_allowed(src["x"], src["y"], src["z"], yaw)
        return {
            "image": src["jpeg"],
            "x": src["x"],
            "y": src["y"],
            "z": src["z"],
            "yaw": src["yaw"],
            "filename": src["filename"],
            "allowed": allowed,
        }
