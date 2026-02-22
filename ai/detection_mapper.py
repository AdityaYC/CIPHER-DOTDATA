"""
PHANTOM CODE — Map YOLO pixel detections to zones on the tactical map.
"""

import config


def map_detections(
    drone_id: str,
    detections: list[dict],
    frame_width: int,
    frame_height: int,
) -> list[dict]:
    """
    Map bounding box centers from pixel coords to the drone's zone on the map.
    Returns list of detection dicts with added "map_x" and "map_y".
    """
    zone = config.CAMERA_ZONES.get(drone_id)
    if not zone or not detections:
        return []

    x_min = zone["x_min"]
    y_min = zone["y_min"]
    x_max = zone["x_max"]
    y_max = zone["y_max"]
    zone_w = x_max - x_min
    zone_h = y_max - y_min

    if frame_width <= 0 or frame_height <= 0:
        return []

    result = []
    for d in detections:
        center = d.get("center", [0, 0])
        cx, cy = center[0], center[1]
        norm_x = cx / frame_width
        norm_y = cy / frame_height
        norm_x = max(0, min(1, norm_x))
        norm_y = max(0, min(1, norm_y))
        map_x = x_min + norm_x * zone_w
        map_y = y_min + norm_y * zone_h
        out = dict(d)
        out["map_x"] = round(map_x, 1)
        out["map_y"] = round(map_y, 1)
        result.append(out)
    return result


def _position_label(map_x: float, map_y: float, x_min: float, y_min: float, x_max: float, y_max: float) -> str:
    """Describe position within zone as northwest/northeast/center/etc."""
    zone_w = x_max - x_min
    zone_h = y_max - y_min
    if zone_w <= 0 or zone_h <= 0:
        return "center"
    nx = (map_x - x_min) / zone_w
    ny = (map_y - y_min) / zone_h
    nx = max(0, min(1, nx))
    ny = max(0, min(1, ny))
    vert = "north" if ny < 0.4 else ("south" if ny > 0.6 else "center")
    horz = "west" if nx < 0.4 else ("east" if nx > 0.6 else "center")
    if vert == "center" and horz == "center":
        return "center"
    if vert == "center":
        return horz + " area"
    if horz == "center":
        return vert + " area"
    return vert + horz + " area"


def get_detection_summary(all_mapped_detections: dict[str, list[dict]]) -> str:
    """
    Produce plain text summary for the LLM from {drone_id: [mapped detections]}.
    Uses ZONE_LABELS and describes positions within each zone.
    """
    lines = []
    for drone_id, dets in all_mapped_detections.items():
        label = config.ZONE_LABELS.get(drone_id, drone_id)
        zone = config.CAMERA_ZONES.get(drone_id, {})
        x_min = zone.get("x_min", 0)
        y_min = zone.get("y_min", 0)
        x_max = zone.get("x_max", 800)
        y_max = zone.get("y_max", 600)

        if not dets:
            lines.append(f"{drone_id} ({label}): No objects detected.")
            continue

        # Group by class and position description
        parts = []
        for d in dets:
            cls = d.get("class", "object")
            conf = d.get("confidence", 0)
            mx = d.get("map_x", 0)
            my = d.get("map_y", 0)
            pos = _position_label(mx, my, x_min, y_min, x_max, y_max)
            parts.append(f"{cls} in {pos} (conf {conf:.0%})")
        lines.append(f"{drone_id} ({label}): " + "; ".join(parts) + ".")
    return "\n".join(lines)
