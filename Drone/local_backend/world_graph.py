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
    
    def to_dict(self):
        return {
            "node_id": self.node_id,
            "timestamp": self.timestamp,
            "gps_lat": self.gps_lat,
            "gps_lon": self.gps_lon,
            "altitude_m": self.altitude_m,
            "yaw_deg": self.yaw_deg,
            "detections": [d.to_dict() for d in self.detections],
            "image_b64": self.image_b64,
        }


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
    
    def clear(self):
        """Clear all nodes from the graph."""
        self.nodes.clear()
        self.node_count = 0
        self.last_position = None
    
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
