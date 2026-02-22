import { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import {
  fetchTacticalStatus,
  fetchTacticalDetections,
  fetchTacticalAdvisory,
  setMission,
  feedImageUrl,
  liveStreamUrl,
  MISSIONS,
  type TacticalStatus,
  type TacticalDetections,
  type TacticalAdvisory,
} from "../api/tactical";

const MAP_WIDTH = 800;
const MAP_HEIGHT = 600;

const CAMERA_ZONES: Record<string, { x_min: number; y_min: number; x_max: number; y_max: number }> = {
  "Drone-1": { x_min: 50, y_min: 50, x_max: 370, y_max: 550 },
  "Drone-2": { x_min: 430, y_min: 50, x_max: 750, y_max: 550 },
};

const ZONE_LABELS: Record<string, string> = {
  "Drone-1": "SECTOR ALPHA",
  "Drone-2": "SECTOR BRAVO",
};

const CLASS_COLORS: Record<string, string> = {
  person: "#00ff66",
  bicycle: "#4488ff",
  car: "#4488ff",
  motorcycle: "#4488ff",
  bus: "#4488ff",
  truck: "#4488ff",
  default: "#888888",
};

function getClassColor(className: string): string {
  return CLASS_COLORS[className] ?? CLASS_COLORS.default;
}

export function TacticalPage() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [status, setStatus] = useState<TacticalStatus | null>(null);
  const [detections, setDetections] = useState<TacticalDetections | null>(null);
  const [advisory, setAdvisory] = useState<TacticalAdvisory | null>(null);
  const [selectedMission, setSelectedMission] = useState("search_rescue");
  const [feedTick, setFeedTick] = useState(0);
  const pulseRef = useRef<Record<string, number>>({});

  useEffect(() => {
    const t = setInterval(() => setFeedTick((n) => n + 1), 500);
    return () => clearInterval(t);
  }, []);

  // Poll status and detections
  useEffect(() => {
    const t = setInterval(async () => {
      const [s, d] = await Promise.all([
        fetchTacticalStatus(),
        fetchTacticalDetections(),
      ]);
      if (s) setStatus(s);
      if (d) setDetections(d);
    }, 500);
    return () => clearInterval(t);
  }, []);

  useEffect(() => {
    const t = setInterval(async () => {
      const a = await fetchTacticalAdvisory();
      if (a) setAdvisory(a);
    }, 3000);
    return () => clearInterval(t);
  }, []);

  // Draw tactical map
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const feeds = status?.feeds ?? {};
    const det = detections ?? {};

    ctx.fillStyle = "#0a0f19";
    ctx.fillRect(0, 0, MAP_WIDTH, MAP_HEIGHT);

    // Grid
    ctx.strokeStyle = "#0d1520";
    ctx.lineWidth = 1;
    for (let x = 0; x <= MAP_WIDTH; x += 40) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, MAP_HEIGHT);
      ctx.stroke();
    }
    for (let y = 0; y <= MAP_HEIGHT; y += 40) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(MAP_WIDTH, y);
      ctx.stroke();
    }

    // Zones
    (["Drone-1", "Drone-2"] as const).forEach((droneId) => {
      const zone = CAMERA_ZONES[droneId];
      if (!zone) return;
      const active = feeds[droneId];
      ctx.strokeStyle = active ? "#00ff66" : "#ff4444";
      ctx.setLineDash([6, 4]);
      ctx.lineWidth = 2;
      ctx.strokeRect(
        zone.x_min,
        zone.y_min,
        zone.x_max - zone.x_min,
        zone.y_max - zone.y_min
      );
      ctx.setLineDash([]);
      const label = ZONE_LABELS[droneId] ?? droneId;
      ctx.font = "11px Courier New";
      ctx.fillStyle = active ? "#00ff66" : "#ff4444";
      ctx.fillText(label, zone.x_min + 4, zone.y_min + 14);
      if (!active) {
        ctx.fillStyle = "#ff4444";
        ctx.font = "12px Courier New";
        ctx.fillText(
          "FEED LOST",
          zone.x_min + (zone.x_max - zone.x_min) / 2 - 30,
          zone.y_min + (zone.y_max - zone.y_min) / 2 - 4
        );
      }
    });

    // Detections
    (["Drone-1", "Drone-2"] as const).forEach((droneId) => {
      const list = det[droneId] ?? [];
      const zone = CAMERA_ZONES[droneId];
      if (!zone) return;
      list.forEach((d, i) => {
        const key = `${droneId}-${i}-${d.map_x}-${d.map_y}`;
        if (!(key in pulseRef.current)) pulseRef.current[key] = 0;
        pulseRef.current[key] += 0.08;
        const pulse = 0.7 + 0.3 * Math.sin(pulseRef.current[key]);
        const r = 6 * pulse;
        ctx.fillStyle = getClassColor(d.class);
        ctx.beginPath();
        ctx.arc(d.map_x, d.map_y, r, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = "#0a0f19";
        ctx.lineWidth = 1;
        ctx.stroke();
        ctx.fillStyle = "#c0c8d4";
        ctx.font = "10px Courier New";
        ctx.fillText(
          d.class,
          d.map_x - 20,
          d.map_y + r + 12
        );
      });
    });

    // No contacts
    (["Drone-1", "Drone-2"] as const).forEach((droneId) => {
      const zone = CAMERA_ZONES[droneId];
      if (!zone || !feeds[droneId]) return;
      const list = det[droneId] ?? [];
      if (list.length === 0) {
        ctx.fillStyle = "rgba(192, 200, 212, 0.25)";
        ctx.font = "11px Courier New";
        ctx.fillText(
          "NO CONTACTS",
          zone.x_min + (zone.x_max - zone.x_min) / 2 - 35,
          zone.y_min + (zone.y_max - zone.y_min) / 2
        );
      }
    });
  }, [status, detections]);

  const handleMissionClick = async (id: string) => {
    setSelectedMission(id);
    await setMission(id);
  };

  return (
    <section className="tactical-page" style={{ display: "flex", flexDirection: "column", height: "100%", background: "#0a0f19", color: "#c0c8d4" }}>
      <div className="status-bar" style={{ display: "flex", gap: "1rem", padding: "0.5rem 1rem", borderBottom: "1px solid rgba(255,255,255,0.1)", flexWrap: "wrap", alignItems: "center" }}>
        <span>NPU: {status?.npu_provider ?? "—"}</span>
        <span>YOLO: {status?.yolo_latency_ms != null ? `${Math.round(status.yolo_latency_ms)}ms` : "—"}</span>
        <span style={{ color: status?.feeds?.["Drone-1"] ? "#00ff66" : "#ff4444" }}>
          Drone-1: {status?.feeds?.["Drone-1"] ? "ACTIVE" : "FEED LOST"}
        </span>
        <span style={{ color: status?.feeds?.["Drone-2"] ? "#00ff66" : "#ff4444" }}>
          Drone-2: {status?.feeds?.["Drone-2"] ? "ACTIVE" : "FEED LOST"}
        </span>
        <Link to={liveStreamUrl()} target="_blank" rel="noopener noreferrer" style={{ color: "var(--swiss-accent)", fontWeight: 700 }}>
          Live stream
        </Link>
      </div>

      <div style={{ display: "flex", flex: 1, minHeight: 0 }}>
        <div style={{ flex: 1, padding: "var(--space-4)", overflow: "auto" }}>
          <canvas
            ref={canvasRef}
            width={MAP_WIDTH}
            height={MAP_HEIGHT}
            style={{ maxWidth: "100%", height: "auto", display: "block" }}
          />
        </div>

        <aside style={{ width: 320, display: "flex", flexDirection: "column", borderLeft: "1px solid rgba(255,255,255,0.1)", padding: "var(--space-4)" }}>
          <div className="advisory-panel" style={{ marginBottom: "var(--space-6)" }}>
            <div className="ai-panel-label">TACTICAL ADVISORY</div>
            <div style={{ fontSize: "0.875rem", lineHeight: 1.5, marginTop: "0.25rem", minHeight: 60 }}>
              {advisory?.text || "Waiting for analysis..."}
            </div>
            <div style={{ fontSize: "0.7rem", color: "rgba(255,255,255,0.5)", marginTop: "0.25rem" }}>
              Mission: {advisory?.mission?.replace(/_/g, " ") ?? "—"} | {advisory?.timestamp ?? "—"}
            </div>
          </div>

          <div className="mission-selector" style={{ marginBottom: "var(--space-6)" }}>
            <div className="ai-panel-label">MISSION</div>
            <div style={{ display: "flex", flexDirection: "column", gap: "0.5rem", marginTop: "0.5rem" }}>
              {MISSIONS.map((m) => (
                <button
                  key={m.id}
                  type="button"
                  className="replay-btn"
                  style={{
                    background: selectedMission === m.id ? "var(--swiss-accent)" : "transparent",
                    color: selectedMission === m.id ? "#fff" : "inherit",
                    border: "2px solid var(--swiss-border)",
                    padding: "0.5rem 0.75rem",
                    cursor: "pointer",
                    fontWeight: 700,
                    fontSize: "0.8rem",
                  }}
                  onClick={() => handleMissionClick(m.id)}
                >
                  {m.label}
                </button>
              ))}
            </div>
          </div>

          <div className="feed-panels" style={{ marginTop: "auto" }}>
            <div className="ai-panel-label">FEEDS</div>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.5rem", marginTop: "0.5rem" }}>
              <div>
                <div style={{ fontSize: "0.7rem", marginBottom: "0.25rem" }}>Drone-1</div>
                <img
                  src={feedImageUrl("Drone-1", true) + "?t=" + feedTick}
                  alt="Drone 1"
                  style={{ width: "100%", aspectRatio: "4/3", objectFit: "cover", background: "#000" }}
                />
              </div>
              <div>
                <div style={{ fontSize: "0.7rem", marginBottom: "0.25rem" }}>Drone-2</div>
                <img
                  src={feedImageUrl("Drone-2", true) + "?t=" + feedTick}
                  alt="Drone 2"
                  style={{ width: "100%", aspectRatio: "4/3", objectFit: "cover", background: "#000" }}
                />
              </div>
            </div>
          </div>
        </aside>
      </div>
    </section>
  );
}
