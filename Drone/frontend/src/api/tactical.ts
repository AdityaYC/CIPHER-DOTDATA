/**
 * Drone2 / PHANTOM CODE APIs: status, detections, advisory, mission, feed
 */

const TACTICAL_BASE =
  import.meta.env.VITE_TACTICAL_API ?? "";

export type TacticalStatus = {
  feeds: Record<string, boolean>;
  npu_provider: string;
  yolo_latency_ms: number;
  camera_ready?: boolean;
  yolo_loaded?: boolean;
  yolo_error?: string | null;
};

export type MappedDetection = {
  class: string;
  confidence: number;
  map_x: number;
  map_y: number;
};

export type TacticalDetections = Record<string, MappedDetection[]>;

export type TacticalAdvisory = {
  text: string;
  mission: string;
  timestamp: string;
};

const MISSIONS = [
  { id: "search_rescue", label: "Search & Rescue" },
  { id: "perimeter", label: "Perimeter Surveillance" },
  { id: "threat_detection", label: "Threat Detection" },
  { id: "damage_assessment", label: "Damage Assessment" },
] as const;

export { MISSIONS };

export async function fetchTacticalStatus(): Promise<TacticalStatus | null> {
  try {
    const r = await fetch(`${TACTICAL_BASE}/api/status`);
    return r.ok ? await r.json() : null;
  } catch {
    return null;
  }
}

export async function fetchTacticalDetections(): Promise<TacticalDetections | null> {
  try {
    const r = await fetch(`${TACTICAL_BASE}/api/detections`);
    return r.ok ? await r.json() : null;
  } catch {
    return null;
  }
}

export async function fetchTacticalAdvisory(): Promise<TacticalAdvisory | null> {
  try {
    const r = await fetch(`${TACTICAL_BASE}/api/advisory`);
    return r.ok ? await r.json() : null;
  } catch {
    return null;
  }
}

export async function setMission(mission: string): Promise<boolean> {
  try {
    const r = await fetch(`${TACTICAL_BASE}/api/mission`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mission }),
    });
    return r.ok && (await r.json()).ok === true;
  } catch {
    return false;
  }
}

export function feedImageUrl(droneId: string, processed = false): string {
  const path = processed ? `/api/feed/${droneId}/processed` : `/api/feed/${droneId}`;
  return `${TACTICAL_BASE}${path}`;
}

export function liveStreamUrl(): string {
  return `${TACTICAL_BASE}/live`;
}
