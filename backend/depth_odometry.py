"""Depth odometry / pose estimation for world graph.

Provides local (x, y, z) pose from GPS (lat, lon, alt) relative to an origin.
When depth maps are available per frame, this module can be extended to
accumulate pose from depth-based odometry; for now uses GPS-derived positions.
"""

import math
from typing import Optional, Tuple


def lat_lon_alt_to_local(
    lat: float,
    lon: float,
    alt: float,
    origin_lat: float,
    origin_lon: float,
    origin_alt: float,
) -> Tuple[float, float, float]:
    """Convert GPS (lat, lon, alt) to local (x, y, z) in meters.
    x = east, y = north, z = up. Origin is (origin_lat, origin_lon, origin_alt)."""
    x = (lon - origin_lon) * 111320 * math.cos(math.radians(origin_lat))
    y = (lat - origin_lat) * 110540
    z = alt - origin_alt
    return (x, y, z)
