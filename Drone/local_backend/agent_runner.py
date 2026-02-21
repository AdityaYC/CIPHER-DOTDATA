"""Agent navigation logic using Llama Vision + YOLO.

This module handles the exploration logic for autonomous agents.
"""

import json
import math
from typing import Dict, List, Optional, Tuple
from PIL import Image

SYSTEM_PROMPT = """\
You are a building-exploration agent. You navigate a discrete grid by choosing movement actions.

## Coordinate system
- The world uses (x, y, z) coordinates with step size 0.1 meters.
- **x and y** are the horizontal axes (the floor plane). **z** is the vertical axis (floor level).
- **yaw** is your facing direction in degrees. Only use: 0, 90, 180, 270.
  - yaw=0 → facing +x direction
  - yaw=90 → facing +y direction
  - yaw=180 → facing -x direction
  - yaw=270 → facing -y direction

## Movement rules
Each turn you will see which directions are **allowed** (true/false):
  forward, backward, left, right, turnLeft, turnRight

You MUST only move in allowed directions. To move:
- **forward**: advance 0.1m in your facing direction
- **backward**: opposite of forward
- **left/right**: strafe perpendicular to facing direction
- **turnLeft**: yaw -= 90 (position stays the same)
- **turnRight**: yaw += 90 (position stays the same)

## Your task
The user asked: "{query}"
Explore the building to find what they asked for.

## How to respond
Output **only** a JSON object (no markdown, no extra text):

If you have NOT found the target:
{{"action": "move", "x": <float>, "y": <float>, "z": <float>, "yaw": <float>, "reasoning": "<1-2 sentences>"}}

If you CAN SEE the target in the current image:
{{"action": "found", "description": "<what and where you see it>", "confidence": "<low|medium|high>", "evidence": ["<visual cue 1>", "<visual cue 2>"]}}

Use "found" only when confidence is HIGH based on direct visual evidence.
"""


class AgentRunner:
    """Runs exploration agent with Llama Vision + YOLO."""
    
    def __init__(self, model_manager, image_db):
        self.models = model_manager
        self.image_db = image_db
    
    def run_agent(
        self,
        query: str,
        start_x: float,
        start_y: float,
        start_z: float,
        start_yaw: float,
        agent_id: int,
        max_steps: int = 15,
    ):
        """Run one exploration agent. Yields step events."""
        x, y, z, yaw = start_x, start_y, start_z, start_yaw
        trajectory = []
        visited_positions = set()
        
        for step in range(max_steps):
            # Get current frame
            idx = self.image_db.find_best(x, y, z, yaw)
            frame_data = self.image_db.db[idx]
            
            # Load image
            image = Image.open(frame_data["path"])
            
            # Get allowed directions
            allowed = self.image_db.check_allowed(
                frame_data["x"], frame_data["y"], frame_data["z"], yaw
            )
            
            # Run YOLO detection (helps with object recognition)
            detections = self.models.detect_objects(image)
            detected_objects = [d["class"] for d in detections if d["confidence"] > 0.5]
            
            # Build prompt for Llama Vision
            prompt = self._build_prompt(
                query, step, max_steps, x, y, z, yaw, allowed, detected_objects, visited_positions
            )
            
            # Run Llama Vision inference
            response = self.models.infer_llama(image, prompt)
            
            # Parse action
            action = self._parse_action(response)
            
            # Prepare step event
            import base64
            from io import BytesIO
            
            # Downscale image for streaming
            img_small = image.resize((256, 256), Image.LANCZOS)
            buf = BytesIO()
            img_small.save(buf, format="JPEG", quality=85)
            img_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            
            step_event = {
                "type": "agent_step",
                "agent_id": agent_id,
                "step": step,
                "total_steps": max_steps,
                "pose": {"x": x, "y": y, "z": z, "yaw": yaw},
                "image_b64": img_b64,
                "reasoning": action.get("reasoning", "") if action else "Exploring...",
                "action": action.get("action", "move") if action else "move",
            }
            
            yield step_event
            
            # Check if found
            if action and action.get("action") == "found":
                # Full resolution image for final result
                buf_full = BytesIO()
                image.save(buf_full, format="JPEG", quality=95)
                img_full_b64 = base64.b64encode(buf_full.getvalue()).decode("ascii")
                
                found_event = {
                    "type": "agent_found",
                    "agent_id": agent_id,
                    "description": action.get("description", "Target found"),
                    "final_image_b64": img_full_b64,
                    "steps": step + 1,
                    "trajectory": trajectory,
                }
                yield found_event
                return
            
            # Apply movement
            if action and action.get("action") == "move":
                new_x = self._clamp(float(action.get("x", x)), -200, 200)
                new_y = self._clamp(float(action.get("y", y)), -200, 200)
                new_z = self._clamp(float(action.get("z", z)), -100, 100)
                new_yaw = float(action.get("yaw", yaw)) % 360
                
                # Validate movement is allowed
                if self._is_valid_move(x, y, z, yaw, new_x, new_y, new_z, new_yaw, allowed):
                    x, y, z, yaw = new_x, new_y, new_z, new_yaw
                    visited_positions.add((round(x, 1), round(y, 1), round(z, 1)))
                else:
                    # Invalid move, try turning
                    yaw = (yaw + 90) % 360
            
            trajectory.append({"x": x, "y": y, "z": z, "yaw": yaw, "step": step})
        
        # Max steps reached
        done_event = {
            "type": "agent_done",
            "agent_id": agent_id,
            "found": False,
            "steps": max_steps,
            "trajectory": trajectory,
        }
        yield done_event
    
    def _build_prompt(
        self,
        query: str,
        step: int,
        max_steps: int,
        x: float,
        y: float,
        z: float,
        yaw: float,
        allowed: Dict,
        detected_objects: List[str],
        visited: set,
    ) -> str:
        """Build prompt for Llama Vision."""
        system = SYSTEM_PROMPT.format(query=query)
        
        context = f"""
Current position: ({x:.2f}, {y:.2f}, {z:.2f}), yaw={yaw:.1f}°
Step {step + 1}/{max_steps}

Allowed movements: {', '.join([k for k, v in allowed.items() if v])}

Objects detected by YOLO: {', '.join(detected_objects) if detected_objects else 'none'}

Visited positions: {len(visited)} locations

What do you see in this image? Should you move forward, turn, or have you found the target?
Respond with JSON only.
"""
        
        return system + context
    
    def _parse_action(self, response: str) -> Optional[Dict]:
        """Parse JSON action from Llama Vision response."""
        try:
            # Extract JSON from response
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                json_str = response[start:end]
                return json.loads(json_str)
        except Exception as e:
            print(f"Failed to parse action: {e}")
        return None
    
    def _is_valid_move(
        self,
        x: float, y: float, z: float, yaw: float,
        new_x: float, new_y: float, new_z: float, new_yaw: float,
        allowed: Dict,
    ) -> bool:
        """Check if a move is valid based on allowed directions."""
        dx = new_x - x
        dy = new_y - y
        dz = new_z - z
        dyaw = new_yaw - yaw
        
        # Check if it's a turn
        if abs(dx) < 0.01 and abs(dy) < 0.01 and abs(dz) < 0.01:
            if abs(dyaw - 90) < 5 or abs(dyaw + 270) < 5:
                return allowed.get("turnRight", False)
            if abs(dyaw + 90) < 5 or abs(dyaw - 270) < 5:
                return allowed.get("turnLeft", False)
        
        # Check movement direction
        dist = math.sqrt(dx**2 + dy**2)
        if dist > 0.01:
            # Determine direction
            angle = math.atan2(dy, dx) * 180 / math.pi
            relative_angle = (angle - yaw) % 360
            
            if relative_angle < 45 or relative_angle > 315:
                return allowed.get("forward", False)
            elif 135 < relative_angle < 225:
                return allowed.get("backward", False)
            elif 45 < relative_angle < 135:
                return allowed.get("right", False)
            elif 225 < relative_angle < 315:
                return allowed.get("left", False)
        
        return True
    
    @staticmethod
    def _clamp(value: float, min_val: float, max_val: float) -> float:
        return max(min_val, min(value, max_val))
