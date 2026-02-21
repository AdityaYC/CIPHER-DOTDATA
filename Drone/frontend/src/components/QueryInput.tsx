import { useState } from "react";
import { Search } from "lucide-react";
import { DEFAULT_AGENT_COUNT } from "../config";

type QueryInputProps = {
  onSubmit: (query: string, numAgents: number) => void;
  disabled: boolean;
};

export function QueryInput({ onSubmit, disabled }: QueryInputProps) {
  const [value, setValue] = useState("");
  const [numAgents, setNumAgents] = useState(DEFAULT_AGENT_COUNT);
  const [agentInputValue, setAgentInputValue] = useState(String(DEFAULT_AGENT_COUNT));

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const trimmed = value.trim();
    if (trimmed && !disabled) {
      onSubmit(trimmed, numAgents);
    }
  };

  return (
    <form className="agent-query-bar" onSubmit={handleSubmit}>
      <input
        className="agent-query-input"
        type="text"
        value={value}
        onChange={(e) => setValue(e.target.value)}
        placeholder="DESCRIBE WHAT TO FIND..."
        disabled={disabled}
      />
      <input
        className="agent-count-input"
        type="number"
        min={1}
        max={8}
        value={agentInputValue}
        onChange={(e) => setAgentInputValue(e.target.value)}
        onBlur={() => {
          const n = parseInt(agentInputValue, 10);
          const clamped = isNaN(n) ? DEFAULT_AGENT_COUNT : Math.max(1, Math.min(8, n));
          setNumAgents(clamped);
          setAgentInputValue(String(clamped));
        }}
        disabled={disabled}
        title="Number of agents"
      />
      <button
        className="agent-query-submit"
        type="submit"
        disabled={disabled || !value.trim()}
      >
        <Search size={16} strokeWidth={2.5} style={{ marginRight: 6, verticalAlign: "middle" }} />
        SEARCH
      </button>
    </form>
  );
}
