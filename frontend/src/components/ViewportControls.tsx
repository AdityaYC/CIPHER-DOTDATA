import { RotateCcw, RotateCw, ArrowUp, ArrowDown, ArrowLeft, ArrowRight } from "lucide-react";
import type { AllowedMoves } from "../api/images";

type ViewportControlsProps = {
  onForward: () => void;
  onBackward: () => void;
  onLeft: () => void;
  onRight: () => void;
  onTurnLeft: () => void;
  onTurnRight: () => void;
  allowed?: AllowedMoves;
};

export function ViewportControls({
  onForward,
  onBackward,
  onLeft,
  onRight,
  onTurnLeft,
  onTurnRight,
  allowed,
}: ViewportControlsProps) {
  const fwd = allowed?.forward ?? true;
  const bwd = allowed?.backward ?? true;
  const lft = allowed?.left ?? true;
  const rgt = allowed?.right ?? true;
  const tl = allowed?.turnLeft ?? true;
  const tr = allowed?.turnRight ?? true;

  return (
    <div className="overlay-controls">
      {/* Rotation controls - bottom corners with circular arrows */}
      <button type="button" className="nav-arrow rotate-left" aria-label="Rotate counter-clockwise" onClick={onTurnLeft} disabled={!tl} title="Turn Left (CCW)">
        <RotateCcw size={26} strokeWidth={2.5} />
        <span className="nav-label">CCW</span>
      </button>
      <button
        type="button"
        className="nav-arrow rotate-right"
        aria-label="Rotate clockwise"
        onClick={onTurnRight}
        disabled={!tr}
        title="Turn Right (CW)"
      >
        <RotateCw size={26} strokeWidth={2.5} />
        <span className="nav-label">CW</span>
      </button>

      {/* Movement controls - center bottom (WASD-style layout) */}
      <button
        type="button"
        className="nav-arrow move-forward"
        aria-label="Move forward"
        onClick={onForward}
        disabled={!fwd}
        title="Move Forward"
      >
        <ArrowUp size={28} strokeWidth={3} />
        <span className="nav-label">FWD</span>
      </button>
      <button
        type="button"
        className="nav-arrow move-left"
        aria-label="Strafe left"
        onClick={onLeft}
        disabled={!lft}
        title="Strafe Left"
      >
        <ArrowLeft size={28} strokeWidth={3} />
        <span className="nav-label">LEFT</span>
      </button>
      <button
        type="button"
        className="nav-arrow move-backward"
        aria-label="Move backward"
        onClick={onBackward}
        disabled={!bwd}
        title="Move Backward"
      >
        <ArrowDown size={28} strokeWidth={3} />
        <span className="nav-label">BACK</span>
      </button>
      <button
        type="button"
        className="nav-arrow move-right"
        aria-label="Strafe right"
        onClick={onRight}
        disabled={!rgt}
        title="Strafe Right"
      >
        <ArrowRight size={28} strokeWidth={3} />
        <span className="nav-label">RIGHT</span>
      </button>
    </div>
  );
}
