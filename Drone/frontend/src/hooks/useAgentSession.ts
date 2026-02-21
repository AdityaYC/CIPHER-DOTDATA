import { useCallback, useEffect, useRef, useState } from "react";
import {
  createSession,
  observeAgentStream,
  startAgentStream,
} from "../api/agentStream";
import { fetchImageForPose } from "../api/images";
import { DEFAULT_AGENT_COUNT, MAX_AGENT_STEPS } from "../config";
import type { AgentState, AgentStep, SessionStatus } from "../types/agent";
import type { AgentEvent } from "../api/agentStream";

const MOCK_MODE = import.meta.env.VITE_AGENT_MOCK === "true";

export function useAgentSession() {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [sessionStatus, setSessionStatus] = useState<SessionStatus>("idle");
  const [agents, setAgents] = useState<Map<number, AgentState>>(new Map());
  const [winnerAgentId, setWinnerAgentId] = useState<number | null>(null);
  const [error, setError] = useState("");
  const [selectedAgentId, setSelectedAgentId] = useState<number | null>(null);

  const abortRef = useRef<AbortController | null>(null);
  const mockTimersRef = useRef<number[]>([]);

  // Clean up on unmount
  useEffect(() => {
    return () => {
      abortRef.current?.abort();
      mockTimersRef.current.forEach(clearTimeout);
    };
  }, []);

  const processEvent = useCallback((event: AgentEvent) => {
    console.log("[SSE event]", event);
    switch (event.type) {
      case "agent_started":
        setAgents((prev) => {
          const next = new Map(prev);
          next.set(event.agent_id, {
            agentId: event.agent_id,
            status: "exploring",
            startPose: event.start_pose,
            steps: [],
            trajectory: [],
          });
          return next;
        });
        // Auto-select first agent
        setSelectedAgentId((cur) => cur ?? event.agent_id);
        break;

      case "agent_step":
        setAgents((prev) => {
          const next = new Map(prev);
          const agent = next.get(event.agent_id);
          if (!agent) return prev;

          const step: AgentStep = {
            step: event.step,
            pose: event.pose,
            imageSrc: `data:image/png;base64,${event.image_b64}`,
            reasoning: event.reasoning,
            action: event.action,
          };

          next.set(event.agent_id, {
            ...agent,
            steps: [...agent.steps, step],
            trajectory: [
              ...agent.trajectory,
              { ...event.pose, step: event.step },
            ],
          });
          return next;
        });
        break;

      case "agent_found":
        setAgents((prev) => {
          const next = new Map(prev);
          const agent = next.get(event.agent_id);
          if (!agent) return prev;
          next.set(event.agent_id, {
            ...agent,
            status: "found",
            description: event.description,
            trajectory: event.trajectory,
          });
          return next;
        });
        setWinnerAgentId(event.agent_id);
        break;

      case "agent_done":
        setAgents((prev) => {
          const next = new Map(prev);
          const agent = next.get(event.agent_id);
          if (!agent) return prev;
          next.set(event.agent_id, {
            ...agent,
            status: "done",
            trajectory: event.trajectory,
          });
          return next;
        });
        break;

      case "session_complete":
        setSessionStatus("complete");
        if (event.winner_agent_id !== null) {
          setWinnerAgentId(event.winner_agent_id);
          setSelectedAgentId(event.winner_agent_id);
        }
        break;

      case "error":
        setError(event.message);
        setSessionStatus("error");
        break;
    }
  }, []);

  // -------------------------------------------------------------------
  // Join an existing session by ID (GET /sessions/:id/stream)
  // -------------------------------------------------------------------
  const joinSession = useCallback(
    (id: string) => {
      setSessionStatus("running");
      setAgents(new Map());
      setWinnerAgentId(null);
      setError("");
      setSelectedAgentId(null);
      setSessionId(id);
      abortRef.current?.abort();

      const controller = observeAgentStream({
        sessionId: id,
        onEvent: processEvent,
        onError: (err) => {
          setError(err.message);
          setSessionStatus("error");
        },
        onComplete: () => {
          setSessionStatus((s) => (s === "running" ? "complete" : s));
        },
      });
      abortRef.current = controller;
    },
    [processEvent],
  );

  // -------------------------------------------------------------------
  // Start a new session (POST /sessions then observe)
  // -------------------------------------------------------------------
  const startSession = useCallback(
    async (query: string, numAgents = DEFAULT_AGENT_COUNT) => {
      // Reset state
      setSessionStatus("running");
      setAgents(new Map());
      setWinnerAgentId(null);
      setError("");
      setSelectedAgentId(null);
      setSessionId(null);
      abortRef.current?.abort();
      mockTimersRef.current.forEach(clearTimeout);
      mockTimersRef.current = [];

      if (MOCK_MODE) {
        runMockSession(query, numAgents, processEvent, mockTimersRef, () =>
          setSessionStatus("complete"),
        );
        return;
      }

      // Use single-POST stream (backend exposes POST /stream_agents with SSE body)
      const controller = startAgentStream({
        query,
        numAgents,
        onEvent: processEvent,
        onError: (e) => {
          setError(e.message);
          setSessionStatus("error");
        },
        onComplete: () => {
          setSessionStatus((s) => (s === "running" ? "complete" : s));
        },
      });
      abortRef.current = controller;
    },
    [processEvent],
  );

  const cancelSession = useCallback(() => {
    abortRef.current?.abort();
    mockTimersRef.current.forEach(clearTimeout);
    mockTimersRef.current = [];
    setSessionStatus("complete");
  }, []);

  const selectAgent = useCallback((id: number) => {
    setSelectedAgentId(id);
  }, []);

  return {
    sessionId,
    sessionStatus,
    agents,
    winnerAgentId,
    error,
    selectedAgentId,
    startSession,
    joinSession,
    cancelSession,
    selectAgent,
  };
}

// ---------------------------------------------------------------------------
// Mock mode — simulates agent events using real image fetching
// ---------------------------------------------------------------------------

function runMockSession(
  query: string,
  numAgents: number,
  onEvent: (e: AgentEvent) => void,
  timersRef: React.RefObject<number[]>,
  onComplete: () => void,
) {
  const mockSteps = 6;
  const winnerAgent = 0;
  const winnerStep = mockSteps - 1;
  let delay = 0;

  // Start events
  for (let i = 0; i < numAgents; i++) {
    const yaw = (i * 180) % 360;
    delay += 300;
    const t = window.setTimeout(() => {
      onEvent({
        type: "agent_started",
        agent_id: i,
        start_pose: { x: 0, y: 0, z: 0, yaw },
      });
    }, delay);
    timersRef.current!.push(t);
  }

  // Step events
  for (let i = 0; i < numAgents; i++) {
    for (let s = 0; s < mockSteps; s++) {
      delay += 1200;
      const agentId = i;
      const step = s;
      const yawBase = (i * 180) % 360;
      const pose = {
        x: (step + 1) * 0.5 * (i === 0 ? 1 : -1),
        y: (step + 1) * 0.3,
        z: 0,
        yaw: (yawBase + step * 30) % 360,
      };

      const t = window.setTimeout(async () => {
        let imageSrc = "";
        try {
          imageSrc = await fetchImageForPose(pose);
          // Strip the data:image/...;base64, prefix to get raw b64
          const commaIdx = imageSrc.indexOf(",");
          imageSrc = commaIdx >= 0 ? imageSrc.slice(commaIdx + 1) : imageSrc;
        } catch {
          imageSrc = "";
        }

        onEvent({
          type: "agent_step",
          agent_id: agentId,
          step,
          total_steps: MAX_AGENT_STEPS,
          pose,
          image_b64: imageSrc,
          reasoning:
            step === winnerStep && agentId === winnerAgent
              ? `I can see what appears to be related to "${query}".`
              : `Exploring area at (${pose.x.toFixed(1)}, ${pose.y.toFixed(1)}). Moving to get a better view.`,
          action:
            step === winnerStep && agentId === winnerAgent ? "found" : "move",
        });

        // Found / done events
        if (step === winnerStep) {
          if (agentId === winnerAgent) {
            const trajectory = Array.from({ length: step + 1 }, (_, s) => ({
              x: (s + 1) * 0.5,
              y: (s + 1) * 0.3,
              z: 0,
              yaw: (s * 30) % 360,
              step: s,
            }));
            onEvent({
              type: "agent_found",
              agent_id: agentId,
              description: `Found what appears to be "${query}" in the current view.`,
              final_image_b64: imageSrc,
              steps: step + 1,
              trajectory,
            });
          } else {
            const trajectory = Array.from({ length: step + 1 }, (_, s) => ({
              x: (s + 1) * -0.5,
              y: (s + 1) * 0.3,
              z: 0,
              yaw: (180 + s * 30) % 360,
              step: s,
            }));
            onEvent({
              type: "agent_done",
              agent_id: agentId,
              found: false,
              steps: step + 1,
              trajectory,
            });
          }
        }
      }, delay);
      timersRef.current!.push(t);
    }
  }

  // Session complete
  delay += 500;
  const finalT = window.setTimeout(() => {
    onEvent({
      type: "session_complete",
      winner_agent_id: winnerAgent,
      description: `Agent ${winnerAgent} found the target.`,
    });
    onComplete();
  }, delay);
  timersRef.current!.push(finalT);
}
