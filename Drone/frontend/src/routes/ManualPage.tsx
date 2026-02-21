import { useEffect, useMemo, useState, useCallback, useRef } from "react";
import type { AllowedMoves } from "../api/images";
import { MOVE_STEP, YAW_STEP_DEGREES, IPHONE_STREAM_URL, API_BASE_URL, BACKEND_FEED_BASE } from "../config";
import { ViewportControls } from "../components/ViewportControls";
import type { Pose } from "../types/pose";
import { fetchTacticalStatus, fetchTacticalAdvisory, fetchTacticalDetections, setMission, MISSIONS, type TacticalStatus, type TacticalAdvisory, type TacticalDetections } from "../api/tactical";

const MAIN_BACKEND = API_BASE_URL || (typeof window !== "undefined" ? window.location.origin : "http://localhost:8000");
const USE_BACKEND_FEED = BACKEND_FEED_BASE !== undefined && BACKEND_FEED_BASE !== null;

// Drone2 tactical map (from frontend/app.js): zone layout and scale for panel
const TACTICAL_MAP_WIDTH = 800;
const TACTICAL_MAP_HEIGHT = 600;
const TACTICAL_MAP_SCALE = 0.5; // panel canvas 400x300
const CAMERA_ZONES: Record<string, { x_min: number; y_min: number; x_max: number; y_max: number }> = {
  "Drone-1": { x_min: 50, y_min: 50, x_max: 370, y_max: 550 },
  "Drone-2": { x_min: 430, y_min: 50, x_max: 750, y_max: 550 },
};
const ZONE_LABELS: Record<string, string> = { "Drone-1": "SECTOR ALPHA", "Drone-2": "SECTOR BRAVO" };
const CLASS_COLORS_MAP: Record<string, string> = {
  person: "#00ff66",
  bicycle: "#4488ff",
  car: "#4488ff",
  motorcycle: "#4488ff",
  bus: "#4488ff",
  truck: "#4488ff",
  default: "#888888",
};

interface Detection {
  class: string;
  confidence: number;
  bbox: [number, number, number, number]; // x1,y1,x2,y2 in pixels
}

const initialPose: Pose = {
  x: 0,
  y: 0,
  z: 0,
  yaw: 0,
};

function normalizeYaw(yaw: number): number {
  const normalized = yaw % 360;
  return normalized < 0 ? normalized + 360 : normalized;
}

export function ManualPage() {
  const [pose, setPose] = useState<Pose>(initialPose);
  const [allowed] = useState<AllowedMoves>({
    forward: true, backward: true, left: true, right: true, turnLeft: true, turnRight: true,
  });
  const allowedRef = useRef(allowed);
  allowedRef.current = allowed;

  // YOLO + Llama state
  const [detections, setDetections] = useState<Detection[]>([]);
  const [llamaText, setLlamaText] = useState<string>("");
  const [aiStatus, setAiStatus] = useState<"idle" | "connecting" | "live" | "waiting" | "error">("idle");
  const [runLlama, setRunLlama] = useState(false);
  const imgRef = useRef<HTMLImageElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const sseRef = useRef<EventSource | null>(null);

  // Drone2 tactical map canvas
  const mapCanvasRef = useRef<HTMLCanvasElement>(null);

  // Backend feed tick: ~4 FPS to reduce flicker (was 100ms)
  const [feedTick, setFeedTick] = useState(0);
  useEffect(() => {
    if (!USE_BACKEND_FEED) return;
    const t = setInterval(() => setFeedTick((n) => n + 1), 250);
    return () => clearInterval(t);
  }, []);

  // Drone2 tactical features (advisory, mission, status, detections for map) — poll when on Manual
  const [tacticalStatus, setTacticalStatus] = useState<TacticalStatus | null>(null);
  const [tacticalDetections, setTacticalDetections] = useState<TacticalDetections | null>(null);
  const [advisory, setAdvisory] = useState<TacticalAdvisory | null>(null);
  const [selectedMission, setSelectedMission] = useState("search_rescue");
  useEffect(() => {
    const poll = async () => {
      const [s, a, d] = await Promise.all([fetchTacticalStatus(), fetchTacticalAdvisory(), fetchTacticalDetections()]);
      if (s) setTacticalStatus(s);
      if (a) setAdvisory(a);
      if (d) setTacticalDetections(d);
    };
    poll();
    const t = setInterval(poll, 1500);
    return () => clearInterval(t);
  }, []);
  const handleMissionClick = useCallback(async (id: string) => {
    setSelectedMission(id);
    await setMission(id);
  }, []);

  // Draw YOLO boxes on canvas overlay
  useEffect(() => {
    const img = imgRef.current;
    const canvas = canvasRef.current;
    if (!canvas || !img) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    canvas.width = img.clientWidth;
    canvas.height = img.clientHeight;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (detections.length === 0) return;

    detections.forEach((det) => {
      const [x1, y1, x2, y2] = det.bbox;
      // bbox is in pixels from YOLO — scale to canvas size
      const scaleX = canvas.width / (img.naturalWidth || canvas.width);
      const scaleY = canvas.height / (img.naturalHeight || canvas.height);
      const bx = x1 * scaleX;
      const by = y1 * scaleY;
      const bw = (x2 - x1) * scaleX;
      const bh = (y2 - y1) * scaleY;

      const conf = Math.round(det.confidence * 100);
      const label = `${det.class} ${conf}%`;

      // Box
      ctx.strokeStyle = "#FF3000";
      ctx.lineWidth = 2;
      ctx.strokeRect(bx, by, bw, bh);

      // Label background
      ctx.font = "bold 13px Inter, sans-serif";
      const textW = ctx.measureText(label).width + 8;
      ctx.fillStyle = "#FF3000";
      ctx.fillRect(bx, by - 20, textW, 20);

      // Label text
      ctx.fillStyle = "#fff";
      ctx.fillText(label, bx + 4, by - 5);
    });
  }, [detections]);

  // SSE connection to /live_detections
  const startDetections = useCallback(() => {
    if (sseRef.current) {
      sseRef.current.close();
    }
    setAiStatus("connecting");
    const url = `${MAIN_BACKEND}/live_detections?run_llama=${runLlama}`;
    const sse = new EventSource(url);
    sseRef.current = sse;

    sse.onmessage = (e) => {
      try {
        const data = JSON.parse(e.data);
        if (data.type === "detections") {
          setDetections(data.detections ?? []);
          if (data.llama_description) setLlamaText(data.llama_description);
          setAiStatus("live");
        } else if (data.type === "waiting") {
          setAiStatus("waiting");
        } else if (data.type === "error") {
          setAiStatus("error");
        }
      } catch {}
    };

    sse.onerror = () => setAiStatus("error");
  }, [runLlama]);

  const stopDetections = useCallback(() => {
    sseRef.current?.close();
    sseRef.current = null;
    setDetections([]);
    setLlamaText("");
    setAiStatus("idle");
  }, []);

  useEffect(() => () => { sseRef.current?.close(); }, []);

  // Draw Drone2 tactical map (zones + detections at map_x, map_y) — from Drone2 frontend/app.js
  useEffect(() => {
    const canvas = mapCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const W = TACTICAL_MAP_WIDTH * TACTICAL_MAP_SCALE;
    const H = TACTICAL_MAP_HEIGHT * TACTICAL_MAP_SCALE;
    const S = TACTICAL_MAP_SCALE;
    canvas.width = W;
    canvas.height = H;
    ctx.fillStyle = "#0a0f19";
    ctx.fillRect(0, 0, W, H);

    // Grid
    ctx.strokeStyle = "#0d1520";
    ctx.lineWidth = 1;
    for (let x = 0; x <= W; x += 40 * S) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke(); }
    for (let y = 0; y <= H; y += 40 * S) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke(); }

    const feeds = tacticalStatus?.feeds ?? {};

    // Zones (Drone-1, Drone-2)
    (["Drone-1", "Drone-2"] as const).forEach((droneId) => {
      const zone = CAMERA_ZONES[droneId];
      if (!zone) return;
      const active = feeds[droneId];
      ctx.strokeStyle = active ? "#00ff66" : "#ff4444";
      ctx.setLineDash([6, 4]);
      ctx.lineWidth = 2;
      ctx.strokeRect(zone.x_min * S, zone.y_min * S, (zone.x_max - zone.x_min) * S, (zone.y_max - zone.y_min) * S);
      ctx.setLineDash([]);
      const label = ZONE_LABELS[droneId] ?? droneId;
      ctx.font = "11px Courier New";
      ctx.fillStyle = active ? "#00ff66" : "#ff4444";
      ctx.fillText(label, zone.x_min * S + 4, zone.y_min * S + 14);
      if (!active) {
        ctx.fillStyle = "#ff4444";
        ctx.font = "12px Courier New";
        const zW = (zone.x_max - zone.x_min) * S, zH = (zone.y_max - zone.y_min) * S;
        ctx.fillText("FEED LOST", zone.x_min * S + zW / 2 - 30, zone.y_min * S + zH / 2 - 4);
      }
    });

    // Detections (map_x, map_y from /api/detections)
    const detections = tacticalDetections ?? {};
    (["Drone-1", "Drone-2"] as const).forEach((droneId) => {
      const list = detections[droneId] ?? [];
      const zone = CAMERA_ZONES[droneId];
      if (!zone || !feeds[droneId]) return;
      if (list.length === 0) {
        ctx.fillStyle = "rgba(192, 200, 212, 0.25)";
        ctx.font = "11px Courier New";
        const zW = (zone.x_max - zone.x_min) * S, zH = (zone.y_max - zone.y_min) * S;
        ctx.fillText("NO CONTACTS", zone.x_min * S + zW / 2 - 35, zone.y_min * S + zH / 2);
        return;
      }
      list.forEach((d) => {
        const mx = d.map_x * S, my = d.map_y * S;
        const color = CLASS_COLORS_MAP[d.class] ?? CLASS_COLORS_MAP.default;
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(mx, my, 6 * S, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = "#0a0f19";
        ctx.lineWidth = 1;
        ctx.stroke();
        ctx.fillStyle = "#c0c8d4";
        ctx.font = "10px Courier New";
        ctx.fillText(`${d.class} ${(d.confidence * 100).toFixed(0)}%`, mx - 20, my + 6 * S + 12);
      });
    });
  }, [tacticalStatus, tacticalDetections]);

  const poseLabel = useMemo(
    () =>
      `x: ${pose.x.toFixed(2)} | y: ${pose.y.toFixed(2)} | z: ${pose.z.toFixed(2)} | yaw: ${pose.yaw.toFixed(0)}°`,
    [pose],
  );

  const moveByYaw = useCallback((step: number) => {
    setPose((prev) => {
      const radians = (prev.yaw * Math.PI) / 180;
      return {
        ...prev,
        x: prev.x + Math.cos(radians) * step,
        y: prev.y + Math.sin(radians) * step,
      };
    });
  }, []);

  // Strafe left/right (perpendicular to yaw direction)
  const strafeByYaw = useCallback((step: number) => {
    setPose((prev) => {
      const radians = (prev.yaw * Math.PI) / 180;
      // Perpendicular direction: rotate 90 degrees
      return {
        ...prev,
        x: prev.x + Math.cos(radians + Math.PI / 2) * step,
        y: prev.y + Math.sin(radians + Math.PI / 2) * step,
      };
    });
  }, []);

  const rotateYaw = useCallback((delta: number) => {
    setPose((prev) => ({
      ...prev,
      yaw: normalizeYaw(prev.yaw + delta),
    }));
  }, []);

  // Keyboard controls
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Ignore if user is typing in an input
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) {
        return;
      }

      const a = allowedRef.current;
      switch (e.key.toLowerCase()) {
        case 'w':
          e.preventDefault();
          if (a.forward) moveByYaw(MOVE_STEP);
          break;
        case 's':
          e.preventDefault();
          if (a.backward) moveByYaw(-MOVE_STEP);
          break;
        case 'a':
          e.preventDefault();
          if (a.left) strafeByYaw(MOVE_STEP);
          break;
        case 'd':
          e.preventDefault();
          if (a.right) strafeByYaw(-MOVE_STEP);
          break;
        case 'q':
          e.preventDefault();
          if (a.turnLeft) rotateYaw(-YAW_STEP_DEGREES);
          break;
        case 'e':
          e.preventDefault();
          if (a.turnRight) rotateYaw(YAW_STEP_DEGREES);
          break;
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [moveByYaw, strafeByYaw, rotateYaw]);

  return (
    <section className="manual-page">
      <div className="status-bar">
        <span className="pose">{poseLabel}</span>
        <span className={`status ${
          aiStatus === "live" ? "" :
          aiStatus === "error" ? "error" : ""
        }`}>
          {aiStatus === "idle" && "AI OFF"}
          {aiStatus === "connecting" && "CONNECTING..."}
          {aiStatus === "live" && `LIVE · ${detections.length} objects`}
          {aiStatus === "waiting" && "WAITING FOR CAMERA..."}
          {aiStatus === "error" && "AI ERROR"}
        </span>
        <div style={{ display: "flex", gap: "0.5rem", alignItems: "center" }}>
          <label style={{ fontSize: "0.75rem", display: "flex", alignItems: "center", gap: "0.25rem", cursor: "pointer" }}>
            <input
              type="checkbox"
              checked={runLlama}
              onChange={(e) => setRunLlama(e.target.checked)}
              disabled={aiStatus !== "idle"}
            />
            Llama Vision
          </label>
          {aiStatus === "idle" ? (
            <button className="replay-btn" onClick={startDetections}>START AI</button>
          ) : (
            <button className="replay-btn" onClick={stopDetections}>STOP AI</button>
          )}
        </div>
      </div>

      <div className="viewport-card" style={{ position: "relative" }}>
        {/* Camera feed: backend /api/feed or iPhone stream */}
        <img
          ref={imgRef}
          src={USE_BACKEND_FEED ? `${MAIN_BACKEND}/api/feed/Drone-1/processed?t=${feedTick}` : IPHONE_STREAM_URL}
          alt="Camera Feed"
          style={{ width: "100%", height: "100%", objectFit: "contain", display: "block", background: "#000" }}
          onLoad={() => {
            const canvas = canvasRef.current;
            const img = imgRef.current;
            if (canvas && img) {
              canvas.width = img.clientWidth;
              canvas.height = img.clientHeight;
            }
          }}
          onError={() => {
            // Keep showing last frame; avoid clearing img to reduce flicker
          }}
        />
        {/* Webcam & models status overlay */}
        {USE_BACKEND_FEED && (
          <div style={{
            position: "absolute", top: "0.75rem", left: "0.75rem",
            display: "flex", flexDirection: "column", gap: "0.25rem",
            background: "rgba(0,0,0,0.75)", color: "#fff",
            padding: "0.4rem 0.6rem", borderRadius: "6px",
            fontSize: "0.7rem", fontWeight: 600,
          }}>
            <span style={{ color: tacticalStatus?.camera_ready ? "#00ff66" : "#ffaa00" }}>
              Webcam: {tacticalStatus?.camera_ready ? "LIVE" : "Starting…"}
            </span>
            {tacticalStatus?.yolo_error ? (
              <span style={{ color: "#ff6666" }} title={tacticalStatus.yolo_error}>
                YOLO: Error (pip install ultralytics?)
              </span>
            ) : (
              <span style={{ color: (tacticalStatus?.yolo_loaded ?? tacticalStatus?.yolo_latency_ms != null) ? "#00ff66" : "rgba(255,255,255,0.6)" }}>
                YOLO: {tacticalStatus?.yolo_latency_ms != null ? `Running (${Math.round(tacticalStatus.yolo_latency_ms)}ms)` : "—"}
              </span>
            )}
          </div>
        )}
        {/* YOLO detection box overlay */}
        <canvas
          ref={canvasRef}
          style={{
            position: "absolute",
            top: 0, left: 0,
            width: "100%", height: "100%",
            pointerEvents: "none",
          }}
        />
        {/* Live indicator */}
        {aiStatus === "live" && (
          <div style={{
            position: "absolute", top: "0.75rem", right: "0.75rem",
            background: "rgba(0,0,0,0.7)", color: "#fff",
            padding: "0.3rem 0.75rem", borderRadius: "4px",
            fontSize: "0.75rem", fontWeight: 700, letterSpacing: "0.05em",
            display: "flex", alignItems: "center", gap: "0.4rem",
          }}>
            <span style={{ width: 8, height: 8, borderRadius: "50%", background: "#FF3000", display: "inline-block", animation: "pulse 2s infinite" }} />
            LIVE
          </div>
        )}

        <ViewportControls
          onForward={() => moveByYaw(MOVE_STEP)}
          onBackward={() => moveByYaw(-MOVE_STEP)}
          onLeft={() => strafeByYaw(MOVE_STEP)}
          onRight={() => strafeByYaw(-MOVE_STEP)}
          onTurnLeft={() => rotateYaw(-YAW_STEP_DEGREES)}
          onTurnRight={() => rotateYaw(YAW_STEP_DEGREES)}
          allowed={allowed}
        />
      </div>

      {/* Bottom panel: detections + world map + Drone2 tactical */}
      <div className="ai-panel" style={{ display: "grid", gridTemplateColumns: "1fr 260px 280px", gap: 0, padding: 0 }}>
        {/* Left: detections + llama */}
        <div style={{ padding: "0.75rem 1rem", borderRight: "1px solid rgba(255,255,255,0.1)" }}>
          {detections.length > 0 && (
            <div style={{ marginBottom: llamaText ? "0.6rem" : 0 }}>
              <span className="ai-panel-label">YOLO DETECTIONS</span>
              <div className="detection-tags">
                {detections.map((d, i) => (
                  <span key={i} className="detection-tag">
                    {d.class} <span className="det-conf">{Math.round(d.confidence * 100)}%</span>
                  </span>
                ))}
              </div>
            </div>
          )}
          {!detections.length && aiStatus === "idle" && (
            <span style={{ color: "rgba(255,255,255,0.3)", fontSize: "0.8rem" }}>Start AI to see detections</span>
          )}
          {llamaText && (
            <div>
              <span className="ai-panel-label">VISION</span>
              <p className="ai-llama-text">{llamaText}</p>
            </div>
          )}
        </div>

        {/* Middle: Drone2 tactical map (zones + detections from /api/detections) */}
        <div style={{ padding: "0.5rem", display: "flex", flexDirection: "column", gap: "0.3rem", borderRight: "1px solid rgba(255,255,255,0.1)" }}>
          <span className="ai-panel-label" style={{ paddingLeft: "0.25rem" }}>
            TACTICAL MAP
          </span>
          <canvas
            ref={mapCanvasRef}
            width={TACTICAL_MAP_WIDTH * TACTICAL_MAP_SCALE}
            height={TACTICAL_MAP_HEIGHT * TACTICAL_MAP_SCALE}
            style={{ background: "#0a0f19", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 2, width: "100%", maxHeight: 320 }}
          />
          <div style={{ display: "flex", gap: "0.6rem", flexWrap: "wrap", fontSize: "0.65rem" }}>
            {Object.entries(CLASS_COLORS_MAP).slice(0, 4).map(([k, c]) => (
              <span key={k} style={{ display: "flex", alignItems: "center", gap: 3, color: "rgba(255,255,255,0.6)" }}>
                <span style={{ width: 8, height: 8, borderRadius: "50%", background: c, display: "inline-block" }} />
                {k}
              </span>
            ))}
          </div>
        </div>

        {/* Right: Drone2 tactical (advisory, mission, status) */}
        <div style={{ padding: "0.75rem", display: "flex", flexDirection: "column", gap: "0.5rem" }}>
          <span className="ai-panel-label">TACTICAL</span>
          <div style={{ fontSize: "0.8rem", lineHeight: 1.4, color: "rgba(255,255,255,0.9)", minHeight: 48 }}>
            {advisory?.text || "—"}
          </div>
          <div style={{ fontSize: "0.65rem", color: "rgba(255,255,255,0.5)" }}>
            {advisory?.mission?.replace(/_/g, " ")} | {advisory?.timestamp ?? "—"}
          </div>
          <span className="ai-panel-label" style={{ marginTop: "0.25rem" }}>MISSION</span>
          <div style={{ display: "flex", flexWrap: "wrap", gap: "0.35rem" }}>
            {MISSIONS.map((m) => (
              <button
                key={m.id}
                type="button"
                className="replay-btn"
                style={{
                  padding: "0.25rem 0.5rem",
                  fontSize: "0.7rem",
                  background: selectedMission === m.id ? "var(--swiss-accent)" : "transparent",
                  color: selectedMission === m.id ? "#fff" : "inherit",
                  border: "1px solid rgba(255,255,255,0.3)",
                }}
                onClick={() => handleMissionClick(m.id)}
              >
                {m.label}
              </button>
            ))}
          </div>
          <span className="ai-panel-label" style={{ marginTop: "0.25rem" }}>STATUS</span>
          <div style={{ fontSize: "0.7rem", color: "rgba(255,255,255,0.7)", display: "flex", flexDirection: "column", gap: "0.2rem" }}>
            <span>NPU: {tacticalStatus?.npu_provider ?? "—"}</span>
            <span>YOLO: {tacticalStatus?.yolo_latency_ms != null ? `${Math.round(tacticalStatus.yolo_latency_ms)}ms` : "—"}</span>
            <span style={{ color: tacticalStatus?.feeds?.["Drone-1"] ? "#00ff66" : "rgba(255,255,255,0.5)" }}>
              Drone-1: {tacticalStatus?.feeds?.["Drone-1"] ? "ACTIVE" : "—"}
            </span>
            <span style={{ color: tacticalStatus?.feeds?.["Drone-2"] ? "#00ff66" : "rgba(255,255,255,0.5)" }}>
              Drone-2: {tacticalStatus?.feeds?.["Drone-2"] ? "ACTIVE" : "—"}
            </span>
          </div>
        </div>
      </div>
    </section>
  );
}

