import { useEffect, useMemo, useState, useCallback } from "react";
import { useParams } from "react-router-dom";
import { useAgentSession } from "../hooks/useAgentSession";
import { QueryInput } from "../components/QueryInput";
import { API_BASE_URL, MAX_AGENT_STEPS } from "../config";

const STATUS_LABELS: Record<string, string> = {
  idle: "READY",
  running: "SEARCHING...",
  complete: "COMPLETE",
  error: "ERROR",
};

const AGENT_STATUS_LABELS: Record<string, string> = {
  waiting: "WAITING",
  exploring: "EXPLORING",
  found: "FOUND",
  done: "DONE",
  error: "ERROR",
};

/** Base URL for API: use relative path in browser so Vite proxy can forward to backend */
const agentApiBase = () =>
  (typeof window !== "undefined" && (!API_BASE_URL || API_BASE_URL === ""))
    ? ""
    : (API_BASE_URL || "http://localhost:8000");

export function AgentPage() {
  const { sessionId: urlSessionId } = useParams<{ sessionId?: string }>();

  const {
    sessionId,
    sessionStatus,
    agents,
    winnerAgentId,
    error,
    startSession,
    joinSession,
    cancelSession,
  } = useAgentSession();

  // Agent mode: one search bar powers tactical query (manuals + map), answer shown below
  const [agentAnswer, setAgentAnswer] = useState("");
  const [agentNodeIds, setAgentNodeIds] = useState<string[]>([]);
  const [agentLoading, setAgentLoading] = useState(false);
  const [agentError, setAgentError] = useState("");

  // Auto-join session from URL on mount
  useEffect(() => {
    if (urlSessionId && sessionStatus === "idle") {
      joinSession(urlSessionId);
    }
  }, [urlSessionId, sessionStatus, joinSession]);

  // Update URL when a new session is created from the query form
  useEffect(() => {
    if (sessionId && !urlSessionId) {
      window.history.replaceState(null, "", `/agent/${sessionId}`);
    }
  }, [sessionId, urlSessionId]);

  const agentList = useMemo(() => Array.from(agents.values()), [agents]);

  // Main Agent search: do BOTH (1) tactical query → show answer, (2) visual agent stream → feed grid
  const handleAgentQuery = useCallback(
    async (query: string, numAgents: number) => {
      const text = (query || "").trim();
      if (!text) return;

      // 1) Start visual agent search (stream) so "SEARCHING..." and agent feed grid run
      startSession(text, numAgents);

      // 2) In parallel, run tactical query (manuals + map) and show answer below
      setAgentLoading(true);
      setAgentAnswer("");
      setAgentNodeIds([]);
      setAgentError("");
      try {
        const base = agentApiBase();
        const res = await fetch(`${base}/api/voice_query`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text }),
        });
        if (!res.ok) throw new Error(res.statusText);
        const data = await res.json();
        setAgentAnswer(data.answer ?? "");
        setAgentNodeIds(Array.isArray(data.node_ids) ? data.node_ids : []);
      } catch (e) {
        setAgentError("Tactical answer unavailable. Backend on port 8000?");
        setAgentAnswer("");
      } finally {
        setAgentLoading(false);
      }
    },
    [startSession],
  );

  // Hide query form when viewing a shared session link
  const showQueryForm =
    !urlSessionId &&
    (sessionStatus === "idle" ||
      sessionStatus === "complete" ||
      sessionStatus === "error");

  // Determine grid columns based on agent count
  const gridCols =
    agentList.length <= 1 ? 1 : agentList.length <= 4 ? 2 : 3;

  return (
    <section className="agent-page">
      {/* Status bar */}
      <div className="status-bar">
        <span className="pose">AGENT MODE</span>
        <span className={`status ${sessionStatus === "error" ? "error" : ""}`}>
          {error || STATUS_LABELS[sessionStatus] || sessionStatus.toUpperCase()}
        </span>
        {sessionStatus === "running" && !urlSessionId && (
          <button className="replay-btn" onClick={cancelSession}>
            CANCEL
          </button>
        )}
        {sessionStatus === "complete" && winnerAgentId !== null && (
          <span className="agent-success-label">
            AGENT {winnerAgentId} FOUND TARGET
          </span>
        )}
        {sessionStatus === "complete" && winnerAgentId === null && (
          <span className="agent-success-label" style={{ color: "var(--swiss-black)" }}>
            NO TARGET FOUND
          </span>
        )}
      </div>

      {/* Agent query: one search bar → tactical query (manuals + map), answer below */}
      {showQueryForm && (
        <QueryInput onSubmit={handleAgentQuery} disabled={agentLoading} />
      )}

      {/* Answer from tactical query (shown below search when loading or has result) */}
      {(agentLoading || agentAnswer || agentError) && (
        <div className="agent-answer-panel">
          {agentLoading && (
            <span className="agent-answer-loading">Querying...</span>
          )}
          {!agentLoading && agentError && (
            <p className="agent-answer-error">{agentError}</p>
          )}
          {!agentLoading && agentAnswer && (
            <>
              <p className="agent-answer-text">{agentAnswer}</p>
              {agentNodeIds.length > 0 && (
                <p className="agent-answer-nodes">Nodes: {agentNodeIds.join(", ")}</p>
              )}
            </>
          )}
        </div>
      )}

      {/* Agent feed grid — visible when session has started */}
      {sessionStatus !== "idle" && (
        <div
          className="agent-feed-grid"
          style={{ "--grid-cols": gridCols } as React.CSSProperties}
        >
          {agentList.length === 0 && (
            <div className="agent-feed-loading">
              <div className="loading-spinner" />
              <p className="loading-message">Agents initializing...</p>
            </div>
          )}
          {agentList.map((agent) => {
            const latestStep = agent.steps[agent.steps.length - 1];
            const stepCount = agent.steps.length;
            const progress = (stepCount / MAX_AGENT_STEPS) * 100;
            const isWinner = agent.agentId === winnerAgentId;

            return (
              <div
                key={agent.agentId}
                className={`agent-feed-panel${isWinner ? " winner" : ""}`}
              >
                {/* Header */}
                <div className="agent-feed-header">
                  <span className="agent-feed-id">AGENT {agent.agentId}</span>
                  <span className={`agent-feed-status ${agent.status}`}>
                    {AGENT_STATUS_LABELS[agent.status] ?? agent.status.toUpperCase()}
                  </span>
                  <span className="agent-feed-steps">
                    {stepCount}/{MAX_AGENT_STEPS}
                  </span>
                </div>

                {/* Progress bar */}
                <div className="agent-feed-progress">
                  <div
                    className="agent-feed-progress-fill"
                    style={{ width: `${progress}%` }}
                  />
                </div>

                {/* Camera feed */}
                <div className="agent-feed-image">
                  {latestStep ? (
                    <>
                      <img
                        src={latestStep.imageSrc}
                        alt={`Agent ${agent.agentId} view`}
                      />
                      {/* Reasoning overlay */}
                      <div className="agent-feed-reasoning">
                        {latestStep.reasoning}
                      </div>
                      {/* Found banner */}
                      {agent.status === "found" && (
                        <div className="agent-found-banner">TARGET FOUND</div>
                      )}
                    </>
                  ) : (
                    <div className="agent-feed-waiting">
                      <div className="loading-spinner" />
                      <span>WAITING FOR FEED...</span>
                    </div>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      )}

    </section>
  );
}
