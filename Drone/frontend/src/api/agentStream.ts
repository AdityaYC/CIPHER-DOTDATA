import { AGENT_API_URL, AGENT_STREAM_URL } from "../config";

// ---------------------------------------------------------------------------
// SSE Event types
// ---------------------------------------------------------------------------

export type Pose = { x: number; y: number; z: number; yaw: number };

export type AgentStartedEvent = {
  type: "agent_started";
  agent_id: number;
  start_pose: Pose;
};

export type AgentStepEvent = {
  type: "agent_step";
  agent_id: number;
  step: number;
  total_steps: number;
  pose: Pose;
  image_b64: string;
  reasoning: string;
  action: "move" | "found";
};

export type AgentFoundEvent = {
  type: "agent_found";
  agent_id: number;
  description: string;
  final_image_b64: string;
  steps: number;
  trajectory: Array<Pose & { step: number }>;
};

export type AgentDoneEvent = {
  type: "agent_done";
  agent_id: number;
  found: false;
  steps: number;
  trajectory: Array<Pose & { step: number }>;
};

export type SessionCompleteEvent = {
  type: "session_complete";
  winner_agent_id: number | null;
  description: string;
};

export type AgentErrorEvent = {
  type: "error";
  agent_id?: number;
  message: string;
};

export type AgentEvent =
  | AgentStartedEvent
  | AgentStepEvent
  | AgentFoundEvent
  | AgentDoneEvent
  | SessionCompleteEvent
  | AgentErrorEvent;

// ---------------------------------------------------------------------------
// SSE parsing helper
// ---------------------------------------------------------------------------

type SSECallbacks = {
  onEvent: (event: AgentEvent) => void;
  onError: (error: Error) => void;
  onComplete: () => void;
};

async function consumeSSE(
  res: Response,
  callbacks: SSECallbacks,
) {
  if (!res.ok) {
    throw new Error(`Agent stream failed (${res.status})`);
  }
  const reader = res.body!.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";

    for (const line of lines) {
      const trimmed = line.trim();
      if (trimmed.startsWith("data: ")) {
        const jsonStr = trimmed.slice(6);
        if (jsonStr === "[DONE]") continue;
        try {
          const event = JSON.parse(jsonStr) as AgentEvent;
          callbacks.onEvent(event);
        } catch {
          // skip malformed lines
        }
      }
    }
  }
  callbacks.onComplete();
}

// ---------------------------------------------------------------------------
// Create a new session (POST /sessions)
// ---------------------------------------------------------------------------

export async function createSession(opts: {
  query: string;
  startX?: number;
  startY?: number;
  startZ?: number;
  startYaw?: number;
  numAgents?: number;
}): Promise<string> {
  const res = await fetch(`${AGENT_API_URL}/sessions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      query: opts.query,
      start_x: opts.startX ?? 0,
      start_y: opts.startY ?? 0,
      start_z: opts.startZ ?? 0,
      start_yaw: opts.startYaw ?? 0,
      num_agents: opts.numAgents ?? 2,
    }),
  });
  if (!res.ok) {
    throw new Error(`Failed to create session (${res.status})`);
  }
  const data = await res.json();
  return data.session_id;
}

// ---------------------------------------------------------------------------
// Observe an existing session (GET /sessions/:id/stream)
// ---------------------------------------------------------------------------

export function observeAgentStream(opts: {
  sessionId: string;
  onEvent: (event: AgentEvent) => void;
  onError: (error: Error) => void;
  onComplete: () => void;
}): AbortController {
  const controller = new AbortController();

  fetch(`${AGENT_API_URL}/sessions/${opts.sessionId}/stream`, {
    method: "GET",
    headers: { Accept: "text/event-stream" },
    signal: controller.signal,
  })
    .then((res) => consumeSSE(res, opts))
    .catch((err) => {
      if (err instanceof DOMException && err.name === "AbortError") return;
      opts.onError(err instanceof Error ? err : new Error(String(err)));
    });

  return controller;
}

// ---------------------------------------------------------------------------
// Legacy: start stream via single POST (backwards compat / fallback)
// ---------------------------------------------------------------------------

export type AgentStreamOptions = {
  query: string;
  startX?: number;
  startY?: number;
  startZ?: number;
  startYaw?: number;
  numAgents?: number;
  onEvent: (event: AgentEvent) => void;
  onError: (error: Error) => void;
  onComplete: () => void;
};

export function startAgentStream(opts: AgentStreamOptions): AbortController {
  const controller = new AbortController();

  const body = JSON.stringify({
    query: opts.query,
    start_x: opts.startX ?? 0,
    start_y: opts.startY ?? 0,
    start_z: opts.startZ ?? 0,
    start_yaw: opts.startYaw ?? 0,
    num_agents: opts.numAgents ?? 2,
  });

  fetch(AGENT_STREAM_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body,
    signal: controller.signal,
  })
    .then((res) => consumeSSE(res, opts))
    .catch((err) => {
      if (err instanceof DOMException && err.name === "AbortError") return;
      opts.onError(err instanceof Error ? err : new Error(String(err)));
    });

  return controller;
}
