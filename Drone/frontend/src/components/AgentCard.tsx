import type { AgentState } from "../types/agent";
import { MAX_AGENT_STEPS } from "../config";

type AgentCardProps = {
  agent: AgentState;
  isWinner: boolean;
  isSelected: boolean;
  onClick: () => void;
};

const STATUS_LABELS: Record<string, string> = {
  waiting: "WAITING",
  exploring: "EXPLORING",
  found: "FOUND",
  done: "DONE",
  error: "ERROR",
};

export function AgentCard({
  agent,
  isWinner,
  isSelected,
  onClick,
}: AgentCardProps) {
  const latestStep = agent.steps[agent.steps.length - 1];
  const stepCount = agent.steps.length;
  const progress = (stepCount / MAX_AGENT_STEPS) * 100;

  const classes = [
    "agent-card",
    isSelected && "selected",
    isWinner && "winner",
  ]
    .filter(Boolean)
    .join(" ");

  return (
    <div className={classes} onClick={onClick}>
      <div className="agent-card-header">
        <span>AGENT {agent.agentId}</span>
        <span className={`agent-card-status ${agent.status}`}>
          {STATUS_LABELS[agent.status] ?? agent.status.toUpperCase()}
        </span>
        <span>
          {stepCount}/{MAX_AGENT_STEPS}
        </span>
      </div>

      <div className="agent-card-image">
        {latestStep ? (
          <img src={latestStep.imageSrc} alt={`Agent ${agent.agentId} view`} />
        ) : (
          <div
            style={{
              height: "100%",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              color: "rgba(255,255,255,0.4)",
              fontSize: "0.75rem",
              fontWeight: 700,
              textTransform: "uppercase",
              letterSpacing: "0.05em",
            }}
          >
            WAITING...
          </div>
        )}
      </div>

      <div className="agent-step-progress">
        <div
          className="agent-step-progress-fill"
          style={{ width: `${progress}%` }}
        />
      </div>

      <div className="agent-card-reasoning">
        {latestStep?.reasoning ?? "\u00A0"}
      </div>
    </div>
  );
}
