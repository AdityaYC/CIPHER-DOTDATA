// In dev (Vite on 5173, 5174, etc.) use backend on 8000 so SSE and feed work
const isDevOrigin =
  typeof window !== "undefined" &&
  window.location.hostname === "localhost" &&
  window.location.port !== "" &&
  window.location.port !== "80" &&
  window.location.port !== "443";
export const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL ??
  (isDevOrigin ? "http://localhost:8000" : typeof window !== "undefined" ? "" : "http://localhost:8000");

/** When set (including ""), Manual page uses this backend's /api/feed/Drone-1/processed instead of iPhone stream */
export const BACKEND_FEED_BASE =
  import.meta.env.VITE_BACKEND_FEED_BASE ?? "";

export const AGENT_STREAM_URL =
  import.meta.env.VITE_AGENT_STREAM_URL ??
  (isDevOrigin ? "http://localhost:8000/stream_agents" : typeof window !== "undefined" ? "/stream_agents" : "http://localhost:8000/stream_agents");

export const AGENT_API_URL =
  import.meta.env.VITE_AGENT_API_URL ??
  (isDevOrigin ? "http://localhost:8000" : typeof window !== "undefined" ? window.location.origin : "http://localhost:8000");

export const MOVE_STEP = 0.5;
export const VERTICAL_STEP = 0.5;
export const YAW_STEP_DEGREES = 90;

export const DEFAULT_AGENT_COUNT = 2;
export const MAX_AGENT_STEPS = 20;

// iPhone camera stream
export const IPHONE_STREAM_URL =
  import.meta.env.VITE_IPHONE_STREAM_URL ?? "http://localhost:8002/stream";

