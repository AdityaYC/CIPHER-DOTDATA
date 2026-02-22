import type { Pose } from "./pose";

export type AgentStatus = "waiting" | "exploring" | "found" | "done" | "error";

export type AgentStep = {
  step: number;
  pose: Pose;
  imageSrc: string;
  reasoning: string;
  action: "move" | "found";
};

export type AgentState = {
  agentId: number;
  status: AgentStatus;
  startPose: Pose;
  steps: AgentStep[];
  description?: string;
  trajectory: Array<Pose & { step: number }>;
};

export type SessionStatus = "idle" | "running" | "complete" | "error";
