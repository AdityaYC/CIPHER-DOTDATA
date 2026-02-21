export const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL ??
  (typeof window !== "undefined" ? "" : "http://localhost:8000");

/** When set (including ""), Manual page uses this backend's /api/feed/Drone-1/processed instead of iPhone stream */
export const BACKEND_FEED_BASE =
  import.meta.env.VITE_BACKEND_FEED_BASE ?? "";

export const AGENT_STREAM_URL =
  import.meta.env.VITE_AGENT_STREAM_URL ??
  (typeof window !== "undefined" ? "/stream_agents" : "http://localhost:8000/stream_agents");

export const AGENT_API_URL =
  import.meta.env.VITE_AGENT_API_URL ??
  (typeof window !== "undefined" ? window.location.origin : "http://localhost:8000");

export const MOVE_STEP = 0.5;
export const VERTICAL_STEP = 0.5;
export const YAW_STEP_DEGREES = 90;

export const DEFAULT_AGENT_COUNT = 2;
export const MAX_AGENT_STEPS = 15;

// iPhone camera stream
export const IPHONE_STREAM_URL =
  import.meta.env.VITE_IPHONE_STREAM_URL ?? "http://localhost:8002/stream";

