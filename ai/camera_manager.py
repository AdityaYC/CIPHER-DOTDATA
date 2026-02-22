"""
PHANTOM CODE — Camera Manager
Manages multiple IP Webcam feeds via OpenCV.
"""

import cv2
import logging

logger = logging.getLogger(__name__)

# Target frame size for consistency
TARGET_WIDTH = 640
TARGET_HEIGHT = 480


class CameraManager:
    """Open and maintain OpenCV VideoCapture connections to each phone feed."""

    def __init__(self, feeds: dict):
        """
        Args:
            feeds: Dict of {drone_id: url} e.g. from config.CAMERA_FEEDS
        """
        self.feeds = feeds
        self.captures: dict[str, cv2.VideoCapture | None] = {}

    def open_all(self) -> None:
        """Open VideoCapture for each feed. Accepts URL string or camera index (int, e.g. 0 for built-in webcam).
        If two drones share the same source (e.g. both 0 on Mac), one capture is opened and reused."""
        source_to_cap: dict = {}
        for drone_id, source in self.feeds.items():
            key = source
            if key not in source_to_cap:
                if isinstance(source, int):
                    cap = cv2.VideoCapture(source)
                else:
                    cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
                if not cap.isOpened():
                    logger.warning(f"[{drone_id}] Failed to open feed: {source}")
                    self.captures[drone_id] = None
                    continue
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                source_to_cap[key] = cap
            self.captures[drone_id] = source_to_cap[key]
            logger.info(f"[{drone_id}] Feed opened: {source}")

    def grab_frame(self, drone_id: str):
        """
        Return latest BGR frame from that feed, or None if unavailable.
        Frames are resized to TARGET_WIDTH x TARGET_HEIGHT if needed.
        Never raises — returns None on any error so the app doesn't crash.
        """
        try:
            cap = self.captures.get(drone_id)
            if cap is None:
                return None
            if not cap.isOpened():
                return None
            ret, frame = cap.read()
            if not ret or frame is None:
                return None
            if frame.size == 0:
                return None
            h, w = frame.shape[:2]
            if w != TARGET_WIDTH or h != TARGET_HEIGHT:
                frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))
            return frame
        except Exception as e:
            logger.debug(f"[{drone_id}] grab_frame: {e}")
            return None

    def grab_all_frames(self) -> dict:
        """Return dict of {drone_id: frame or None} for all feeds."""
        return {drone_id: self.grab_frame(drone_id) for drone_id in self.feeds}

    def release_all(self) -> None:
        """Release all VideoCapture resources (each unique capture only once)."""
        seen = set()
        for drone_id, cap in self.captures.items():
            if cap is not None and id(cap) not in seen:
                seen.add(id(cap))
                cap.release()
                logger.info(f"[{drone_id}] Released")
        self.captures.clear()
