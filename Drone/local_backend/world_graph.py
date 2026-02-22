"""World Graph - Spatial graph of detected objects with GPS positions.

Each node represents a location where the drone captured a frame.
Nodes store: GPS position, YOLO detections, distances, and categories.
"""

import math
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum


class ObjectCategory(str, Enum):
    """Categories for detected objects."""
    SURVIVOR = "survivor"
    HAZARD = "hazard"
    EXIT = "exit"
    CLEAR = "clear"
    STRUCTURAL = "structural"
    UNKNOWN = "unknown"


@dataclass
class Detection:
    """Single YOLO detection with metadata."""
    class_name: str
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2]
    distance_meters: Optional[float] = None
    category: ObjectCategory = ObjectCategory.UNKNOWN
    
    def to_dict(self):
        return {
            "class_name": self.class_name,
            "confidence": self.confidence,
            "bbox": self.bbox,
            "distance_meters": self.distance_meters,
            "category": self.category.value,
        }


@dataclass
class GraphNode:
    """Node in the world graph representing a drone position."""
    node_id: str
    timestamp: float
    gps_lat: float
    gps_lon: float
    altitude_m: float
    yaw_deg: float
    detections: List[Detection]
    image_b64: Optional[str] = None  # Optional: store thumbnail
    source: str = "live"  # "live" | "imported"
    depth_b64: Optional[str] = None  # Optional: depth map image (e.g. for imported)
    local_x: Optional[float] = None  # When set (e.g. imported), used as pose instead of gps-derived
    local_y: Optional[float] = None
    local_z: Optional[float] = None
    
    def to_dict(self):
        out = {
            "node_id": self.node_id,
            "timestamp": self.timestamp,
            "gps_lat": self.gps_lat,
            "gps_lon": self.gps_lon,
            "altitude_m": self.altitude_m,
            "yaw_deg": self.yaw_deg,
            "detections": [d.to_dict() for d in self.detections],
            "image_b64": self.image_b64,
            "source": self.source,
        }
        if self.depth_b64 is not None:
            out["depth_b64"] = self.depth_b64
        return out


class WorldGraph:
    """Spatial graph of drone observations.
    
    Only adds nodes when drone moves beyond MIN_DISTANCE threshold
    to avoid duplicate positions.
    """
    
    MIN_DISTANCE = 0.5  # meters - minimum distance to create new node
    
    def __init__(self):
        self.nodes: Dict[str, GraphNode] = {}
        self.node_count = 0
        self.last_position: Optional[Tuple[float, float, float]] = None
        # Imported video: explicit next/prev edges (CLIP-based; only 0.3 <= sim < 0.95 get edge)
        self._imported_next: Dict[str, str] = {}
        self._imported_prev: Dict[str, str] = {}

        # Category mappings for YOLO classes
        self.category_map = {
            # Survivors
            "person": ObjectCategory.SURVIVOR,
            "dog": ObjectCategory.SURVIVOR,
            "cat": ObjectCategory.SURVIVOR,
            
            # Hazards
            "fire": ObjectCategory.HAZARD,
            "smoke": ObjectCategory.HAZARD,
            "knife": ObjectCategory.HAZARD,
            "scissors": ObjectCategory.HAZARD,
            
            # Exits
            "door": ObjectCategory.EXIT,
            "window": ObjectCategory.EXIT,
            
            # Clear (safe objects)
            "chair": ObjectCategory.CLEAR,
            "couch": ObjectCategory.CLEAR,
            "table": ObjectCategory.CLEAR,
            "bed": ObjectCategory.CLEAR,
            "tv": ObjectCategory.CLEAR,
            "laptop": ObjectCategory.CLEAR,
            "book": ObjectCategory.CLEAR,
        }
    
    def should_add_node(self, lat: float, lon: float, alt: float) -> bool:
        """Check if drone has moved far enough to warrant a new node."""
        if self.last_position is None:
            return True
        
        last_lat, last_lon, last_alt = self.last_position
        distance = self._calculate_distance(lat, lon, alt, last_lat, last_lon, last_alt)
        
        return distance >= self.MIN_DISTANCE
    
    def add_node(
        self,
        lat: float,
        lon: float,
        alt: float,
        yaw: float,
        yolo_detections: List[Dict],
        image_b64: Optional[str] = None,
    ) -> Optional[GraphNode]:
        """Add a new node to the graph if position is far enough from last.
        
        Args:
            lat: GPS latitude
            lon: GPS longitude
            alt: Altitude in meters
            yaw: Heading in degrees
            yolo_detections: List of YOLO detection dicts
            image_b64: Optional base64 encoded thumbnail
            
        Returns:
            GraphNode if added, None if too close to last position
        """
        if not self.should_add_node(lat, lon, alt):
            return None
        
        # Create node
        node_id = f"node_{self.node_count:04d}"
        self.node_count += 1
        
        # Process detections
        detections = []
        for det in yolo_detections:
            class_name = det.get("class", det.get("class_name", "unknown"))
            confidence = det.get("confidence", 0.0)
            bbox = det.get("bbox", [0, 0, 0, 0])
            
            # Categorize object
            category = self.category_map.get(class_name, ObjectCategory.UNKNOWN)
            
            # Estimate distance (placeholder - would use depth estimation)
            distance = self._estimate_distance(bbox, alt)
            
            detection = Detection(
                class_name=class_name,
                confidence=confidence,
                bbox=bbox,
                distance_meters=distance,
                category=category,
            )
            detections.append(detection)
        
        # Create and store node
        node = GraphNode(
            node_id=node_id,
            timestamp=time.time(),
            gps_lat=lat,
            gps_lon=lon,
            altitude_m=alt,
            yaw_deg=yaw,
            detections=detections,
            image_b64=image_b64,
        )
        
        self.nodes[node_id] = node
        self.last_position = (lat, lon, alt)
        
        return node
    
    def add_imported_node(
        self,
        local_x: float,
        local_y: float,
        local_z: float,
        yaw: float,
        yolo_detections: List[Dict],
        image_b64: Optional[str] = None,
        depth_b64: Optional[str] = None,
    ) -> GraphNode:
        """Add a node from imported footage (local pose, no GPS)."""
        node_id = f"node_{self.node_count:04d}"
        self.node_count += 1
        detections = []
        for det in yolo_detections:
            class_name = det.get("class", det.get("class_name", "unknown"))
            confidence = det.get("confidence", 0.0)
            bbox = det.get("bbox", [0, 0, 0, 0])
            category = self.category_map.get(class_name, ObjectCategory.UNKNOWN)
            distance = self._estimate_distance(bbox, max(0.1, local_z))
            detections.append(Detection(
                class_name=class_name,
                confidence=confidence,
                bbox=bbox,
                distance_meters=distance,
                category=category,
            ))
        node = GraphNode(
            node_id=node_id,
            timestamp=time.time(),
            gps_lat=0.0,
            gps_lon=0.0,
            altitude_m=0.0,
            yaw_deg=yaw,
            detections=detections,
            image_b64=image_b64,
            source="imported",
            depth_b64=depth_b64,
            local_x=local_x,
            local_y=local_y,
            local_z=local_z,
        )
        self.nodes[node_id] = node
        return node
    
    def get_graph(self) -> Dict:
        """Return the entire graph as a dict."""
        return {
            "nodes": [node.to_dict() for node in self.nodes.values()],
            "node_count": self.node_count,
            "last_updated": time.time(),
        }
    
    def get_stats(self) -> Dict:
        """Return statistics about the graph."""
        total_detections = sum(len(node.detections) for node in self.nodes.values())
        
        # Count by category
        category_counts = {cat.value: 0 for cat in ObjectCategory}
        for node in self.nodes.values():
            for det in node.detections:
                category_counts[det.category.value] += 1
        
        # Calculate coverage (simplified - would use actual area)
        coverage_m2 = self.node_count * (self.MIN_DISTANCE ** 2) * math.pi
        
        return {
            "node_count": self.node_count,
            "total_detections": total_detections,
            "category_counts": category_counts,
            "coverage_m2": round(coverage_m2, 2),
            "coverage_percentage": min(100, round(coverage_m2 / 100, 2)),  # Assume 100m² target area
        }
    
    def get_nodes_by_category(self, category: ObjectCategory) -> List[GraphNode]:
        """Get all nodes that detected objects in a specific category."""
        result = []
        for node in self.nodes.values():
            if any(det.category == category for det in node.detections):
                result.append(node)
        return result
    
    def _origin_pose(self) -> Optional[Tuple[float, float, float]]:
        """First node's (lat, lon, alt) as origin for local coordinates."""
        ordered = sorted(self.nodes.keys())
        if not ordered:
            return None
        n = self.nodes[ordered[0]]
        return (n.gps_lat, n.gps_lon, n.altitude_m)
    
    def get_pose_at_node(self, node_id: str) -> Optional[Tuple[float, float, float]]:
        """Return (x, y, z) position in local meters for the given node.
        Origin is the first node (by id order). Uses local_x/y/z when set (imported nodes)."""
        if node_id not in self.nodes:
            return None
        n = self.nodes[node_id]
        if n.local_x is not None and n.local_y is not None and n.local_z is not None:
            return (n.local_x, n.local_y, n.local_z)
        origin = self._origin_pose()
        if origin is None:
            return None
        lat0, lon0, alt0 = origin
        x = (n.gps_lon - lon0) * 111320 * math.cos(math.radians(lat0))
        y = (n.gps_lat - lat0) * 110540
        z = n.altitude_m - alt0
        return (x, y, z)
    
    def get_neighbor_direction(self, node_id: str, direction: str) -> Optional[str]:
        """Return node_id of the neighbor most in the given direction (forward, back, left, right).
        Uses yaw at current node. Returns None if no neighbor in that direction."""
        if node_id not in self.nodes or direction not in ("forward", "back", "left", "right"):
            return None
        pos = self.get_pose_at_node(node_id)
        if pos is None:
            return None
        cx, cy, cz = pos
        n = self.nodes[node_id]
        yaw_rad = math.radians(n.yaw_deg)
        # Forward = direction of yaw (x = east, y = north; yaw 0 = +x)
        fx = math.cos(yaw_rad)
        fy = math.sin(yaw_rad)
        if direction == "forward":
            dx, dy = fx, fy
        elif direction == "back":
            dx, dy = -fx, -fy
        elif direction == "left":
            dx, dy = -fy, fx
        else:  # right
            dx, dy = fy, -fx
        best_id: Optional[str] = None
        best_dot = -2.0
        for other_id, other_node in self.nodes.items():
            if other_id == node_id:
                continue
            op = self.get_pose_at_node(other_id)
            if op is None:
                continue
            ox, oy, oz = op
            vx = ox - cx
            vy = oy - cy
            dist_xy = math.sqrt(vx * vx + vy * vy)
            if dist_xy < 0.01:
                continue
            vx /= dist_xy
            vy /= dist_xy
            dot = vx * dx + vy * dy
            if dot > 0.25 and dot > best_dot:
                best_dot = dot
                best_id = other_id
        return best_id

    def link_imported(self, prev_id: str, next_id: str) -> None:
        """Record navigable edge between two imported nodes (used when CLIP similarity in [0.3, 0.95))."""
        if prev_id in self.nodes and next_id in self.nodes:
            self._imported_next[prev_id] = next_id
            self._imported_prev[next_id] = prev_id

    def get_neighbor_by_order(self, node_id: str, direction: str) -> Optional[str]:
        """Return next or previous node in visit order (for video-style navigation). direction is 'next' or 'prev'."""
        if node_id not in self.nodes or direction not in ("next", "prev"):
            return None
        if direction == "next" and node_id in self._imported_next:
            return self._imported_next[node_id]
        if direction == "prev" and node_id in self._imported_prev:
            return self._imported_prev[node_id]
        ordered = sorted(self.nodes.keys())
        try:
            i = ordered.index(node_id)
        except ValueError:
            return None
        if direction == "next" and i + 1 < len(ordered):
            return ordered[i + 1]
        if direction == "prev" and i - 1 >= 0:
            return ordered[i - 1]
        return None
    
    def get_path(self, start_id: str, end_id: str) -> List[str]:
        """BFS path as list of node_ids. Graph is linear (visit order), so path is the ordered segment."""
        ordered = sorted(self.nodes.keys())
        if not ordered or start_id not in self.nodes or end_id not in self.nodes:
            return []
        try:
            i = ordered.index(start_id)
            j = ordered.index(end_id)
        except ValueError:
            return []
        if i <= j:
            return ordered[i : j + 1]
        return ordered[j : i + 1][::-1]
    
    def to_3d_pointcloud(self) -> List[Tuple[float, float, float, float, float, float]]:
        """Return list of (x, y, z, r, g, b) in local meters; RGB 0..1. One point per node, color by category."""
        out: List[Tuple[float, float, float, float, float, float]] = []
        origin = self._origin_pose()
        if origin is None:
            return out
        lat0, lon0, alt0 = origin
        # Category to RGB (0-1)
        def cat_color(cat: ObjectCategory) -> Tuple[float, float, float]:
            if cat == ObjectCategory.SURVIVOR:
                return (0.0, 1.0, 0.4)
            if cat == ObjectCategory.HAZARD:
                return (1.0, 0.2, 0.2)
            if cat == ObjectCategory.EXIT:
                return (0.2, 0.5, 1.0)
            if cat == ObjectCategory.STRUCTURAL:
                return (1.0, 0.5, 0.0)
            if cat == ObjectCategory.CLEAR:
                return (0.5, 0.5, 0.5)
            return (0.4, 0.4, 0.45)
        for node_id in sorted(self.nodes.keys()):
            n = self.nodes[node_id]
            pose = self.get_pose_at_node(node_id)
            if pose is None:
                continue
            x, y, z = pose
            if n.source == "imported":
                r, g, b = (1.0, 1.0, 1.0)
            else:
                cats = [d.category for d in n.detections]
                if cats:
                    cat = max(set(cats), key=cats.count)
                    r, g, b = cat_color(cat)
                else:
                    r, g, b = (0.45, 0.45, 0.5)
            out.append((x, y, z, r, g, b))
        return out
    
    def clear(self):
        """Clear all nodes from the graph."""
        self.nodes.clear()
        self.node_count = 0
        self.last_position = None
        self._imported_next.clear()
        self._imported_prev.clear()
    
    @staticmethod
    def _calculate_distance(
        lat1: float, lon1: float, alt1: float,
        lat2: float, lon2: float, alt2: float,
    ) -> float:
        """Calculate 3D distance between two GPS positions.
        
        For small distances, we can use simple Euclidean distance.
        For real GPS, would use Haversine formula.
        """
        # Simplified: treat as meters (works for small areas)
        dx = (lon2 - lon1) * 111320 * math.cos(math.radians(lat1))  # meters
        dy = (lat2 - lat1) * 110540  # meters
        dz = alt2 - alt1  # meters
        
        return math.sqrt(dx**2 + dy**2 + dz**2)
    
    @staticmethod
    def _estimate_distance(bbox: List[float], altitude: float) -> float:
        """Estimate distance to object based on bbox size and altitude.
        
        This is a simplified heuristic. In production, would use:
        - Depth estimation model (like Depth-Pro in ml-depth-pro/)
        - Camera intrinsics
        - Known object sizes
        """
        # Bbox size as proxy for distance
        x1, y1, x2, y2 = bbox
        bbox_height = abs(y2 - y1)
        
        # Larger bbox = closer object
        # Assume normalized coordinates [0, 1]
        if bbox_height > 0.5:
            return altitude * 0.5  # Very close
        elif bbox_height > 0.3:
            return altitude * 1.0  # Close
        elif bbox_height > 0.1:
            return altitude * 2.0  # Medium
        else:
            return altitude * 3.0  # Far
