import { NavLink } from "react-router-dom";

export function ModeSwitch() {
  return (
    <nav className="mode-switch" aria-label="Mode selector">
      <NavLink
        to="/agent"
        className={({ isActive }) =>
          `mode-switch-item ${isActive ? "active" : ""}`
        }
      >
        Agent
      </NavLink>
      <NavLink
        to="/manual"
        className={({ isActive }) =>
          `mode-switch-item ${isActive ? "active" : ""}`
        }
      >
        Manual
      </NavLink>
      <NavLink
        to="/replay"
        className={({ isActive }) =>
          `mode-switch-item ${isActive ? "active" : ""}`
        }
      >
        Replay
      </NavLink>
    </nav>
  );
}

