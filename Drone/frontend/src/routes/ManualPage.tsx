import { useEffect, useMemo, useState, useCallback, useRef } from "react";
import { IPHONE_STREAM_URL, API_BASE_URL, BACKEND_FEED_BASE } from "../config";
import { fetchTacticalStatus, fetchTacticalDetections, type TacticalStatus, type TacticalDetections } from "../api/tactical";

const MAIN_BACKEND = API_BASE_URL || (typeof window !== "undefined" ? window.location.origin : "http://localhost:8000");
const USE_BACKEND_FEED = BACKEND_FEED_BASE !== undefined && BACKEND_FEED_BASE !== null;

/** Placeholder when backend is down — avoids ERR_CONNECTION_REFUSED spam */
const BACKEND_DOWN_PLACEHOLDER =
  "data:image/svg+xml," + encodeURIComponent(
    '<svg xmlns="http://www.w3.org/2000/svg" width="640" height="480" viewBox="0 0 640 480"><rect fill="#0a0f19" width="640" height="480"/><text x="320" y="220" fill="rgba(255,255,255,0.9)" font-family="sans-serif" font-size="18" text-anchor="middle">Backend not running on port 8000</text><text x="320" y="260" fill="rgba(255,255,255,0.6)" font-family="sans-serif" font-size="14" text-anchor="middle">Start it from repo root:</text><text x="320" y="295" fill="#00ff66" font-family="monospace" font-size="13" text-anchor="middle">.\\run_drone_full.ps1</text><text x="320" y="330" fill="rgba(255,255,255,0.5)" font-family="sans-serif" font-size="12" text-anchor="middle">(or open a terminal and run the backend, then refresh)</text></svg>'
  );

// Semantic map: detection class colors (same as tactical)
const SEMANTIC_CLASS_COLORS: Record<string, string> = {
  person: "#00ff66",
  bicycle: "#4488ff",
  car: "#4488ff",
  motorcycle: "#4488ff",
  bus: "#4488ff",
  truck: "#4488ff",
  default: "#888888",
};
function getSemanticClassColor(className: string): string {
  return SEMANTIC_CLASS_COLORS[className] ?? SEMANTIC_CLASS_COLORS.default;
}

// Backend Drone-1 zone (map_x, map_y range) — we scale to semantic map canvas
const ZONE_DRONE1 = { x_min: 50, y_min: 50, x_max: 370, y_max: 550 };
const SEMANTIC_MAP_W = 320;
const SEMANTIC_MAP_H = 200;

interface Detection {
  class: string;
  confidence: number;
  bbox: [number, number, number, number]; // x1,y1,x2,y2 in pixels
  distance_meters?: number | null;
}

export function ManualPage() {
  const [detections, setDetections] = useState<Detection[]>([]);
  const [aiStatus, setAiStatus] = useState<"idle" | "connecting" | "live" | "waiting" | "error">("idle");
  const imgRef = useRef<HTMLImageElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const sseRef = useRef<EventSource | null>(null);
  const semanticMapRef = useRef<HTMLCanvasElement>(null);
  const [tacticalDetections, setTacticalDetections] = useState<TacticalDetections | null>(null);
  const semanticPulseRef = useRef<Record<string, number>>({});
  const [backendReachable, setBackendReachable] = useState<boolean | null>(null);

  // Poll tactical detections for semantic map (map_x, map_y)
  useEffect(() => {
    if (backendReachable !== true) return;
    const t = setInterval(async () => {
      const d = await fetchTacticalDetections();
      if (d) setTacticalDetections(d);
    }, 500);
    return () => clearInterval(t);
  }, [backendReachable]);

  // Backend reachable: when false, show placeholder and don't hammer 8000 (fixes ERR_CONNECTION_REFUSED spam)
  useEffect(() => {
    if (!USE_BACKEND_FEED || !MAIN_BACKEND) {
      setBackendReachable(true);
      return;
    }
    let cancelled = false;
    const check = async () => {
      try {
        const r = await fetch(`${MAIN_BACKEND}/health`, { signal: AbortSignal.timeout(3000) });
        if (!cancelled) setBackendReachable(r.ok);
      } catch {
        if (!cancelled) setBackendReachable(false);
      }
    };
    check();
    const t = setInterval(() => {
      if (backendReachable === false) check(); // re-check every 5s when down so we recover when user starts backend
    }, 5000);
    return () => {
      cancelled = true;
      clearInterval(t);
    };
  }, [USE_BACKEND_FEED, MAIN_BACKEND, backendReachable]);

  // Backend feed tick: only when backend is reachable to avoid connection refused spam
  const [feedTick, setFeedTick] = useState(0);
  useEffect(() => {
    if (!USE_BACKEND_FEED || backendReachable !== true) return;
    const t = setInterval(() => setFeedTick((n) => n + 1), 100);
    return () => clearInterval(t);
  }, [backendReachable]);

  const [tacticalStatus, setTacticalStatus] = useState<TacticalStatus | null>(null);
  useEffect(() => {
    const poll = async () => {
      const s = await fetchTacticalStatus();
      if (s) {
        setTacticalStatus(s);
        if (backendReachable === false) setBackendReachable(true);
      }
    };
    if (backendReachable === false) {
      const t = setInterval(poll, 5000);
      return () => clearInterval(t);
    }
    if (backendReachable !== true) return;
    poll();
    const t = setInterval(poll, 1500);
    return () => clearInterval(t);
  }, [backendReachable]);

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

      const distStr = det.distance_meters != null ? ` · ${Math.round(det.distance_meters * 100)}cm` : "";
      const label = `${det.class}${distStr}`;

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
    const url = `${MAIN_BACKEND}/live_detections?run_llama=false`;
    const sse = new EventSource(url);
    sseRef.current = sse;

    sse.onmessage = (e) => {
      try {
        const data = JSON.parse(e.data);
        if (data.type === "detections") {
          setDetections(data.detections ?? []);
          setAiStatus("live");
        } else if (data.type === "waiting") {
          setAiStatus("waiting");
        } else if (data.type === "error") {
          setAiStatus("error");
        }
      } catch {}
    };

    sse.onerror = () => setAiStatus("error");
  }, []);

  const stopDetections = useCallback(() => {
    sseRef.current?.close();
    sseRef.current = null;
    setDetections([]);
    setAiStatus("idle");
  }, []);

  useEffect(() => () => { sseRef.current?.close(); }, []);

  // Semantic map: camera zone + detections at map_x, map_y (scaled from backend zone to canvas)
  useEffect(() => {
    const canvas = semanticMapRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const W = (canvas.width = SEMANTIC_MAP_W);
    const H = (canvas.height = SEMANTIC_MAP_H);
    ctx.fillStyle = "#0a0f19";
    ctx.fillRect(0, 0, W, H);
    const zone = ZONE_DRONE1;
    const scaleX = W / (zone.x_max - zone.x_min);
    const scaleY = H / (zone.y_max - zone.y_min);
    const toCanvas = (map_x: number, map_y: number) => ({
      x: (map_x - zone.x_min) * scaleX,
      y: (map_y - zone.y_min) * scaleY,
    });
    ctx.strokeStyle = "#00ff66";
    ctx.setLineDash([4, 3]);
    ctx.lineWidth = 2;
    ctx.strokeRect(0, 0, W, H);
    ctx.setLineDash([]);
    ctx.font = "10px Inter, sans-serif";
    ctx.fillStyle = "#00ff66";
    ctx.fillText("LIVE", 6, 12);
    const det = tacticalDetections ?? {};
    const list = det["Drone-1"] ?? [];
    list.forEach((d, i) => {
      const key = `d1-${i}-${d.map_x}-${d.map_y}`;
      if (!(key in semanticPulseRef.current)) semanticPulseRef.current[key] = 0;
      semanticPulseRef.current[key] += 0.08;
      const pulse = 0.7 + 0.3 * Math.sin(semanticPulseRef.current[key]);
      const r = 5 * pulse;
      const { x, y } = toCanvas(d.map_x, d.map_y);
      ctx.fillStyle = getSemanticClassColor(d.class);
      ctx.beginPath();
      ctx.arc(x, y, r, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = "#0a0f19";
      ctx.lineWidth = 1;
      ctx.stroke();
      ctx.fillStyle = "#c0c8d4";
      ctx.fillText(`${d.class} ${(d.confidence * 100).toFixed(0)}%`, x - 14, y + r + 10);
    });
  }, [tacticalDetections]);

  const poseDisplay = useMemo(() => ({ x: 0, y: 0, z: 0, yaw: 0 }), []);

  return (
    <section className="manual-page">
      {USE_BACKEND_FEED && backendReachable === false && (
        <div
          style={{
            background: "linear-gradient(90deg, #8b0000 0%, #4a0000 100%)",
            color: "#fff",
            padding: "0.6rem 1rem",
            fontSize: "0.85rem",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            gap: "0.75rem",
            flexWrap: "wrap",
          }}
        >
          <span style={{ fontWeight: 600 }}>Backend not running — webcam and YOLO need it.</span>
          <span>From repo root in PowerShell:</span>
          <code style={{ background: "rgba(0,0,0,0.4)", padding: "0.25rem 0.5rem", borderRadius: 4 }}>
            .\run_drone_full.ps1
          </code>
          <span style={{ opacity: 0.85 }}>Then refresh this page.</span>
        </div>
      )}
      <div className="status-bar">
        <span className="pose">
          X {poseDisplay.x.toFixed(2)} · Y {poseDisplay.y.toFixed(2)} · Z {poseDisplay.z.toFixed(2)} · YAW {poseDisplay.yaw.toFixed(0)}°
        </span>
        <span className={`status ${aiStatus === "live" ? "" : aiStatus === "error" ? "error" : ""}`}>
          {aiStatus === "idle" && "AI OFF"}
          {aiStatus === "connecting" && "CONNECTING..."}
          {aiStatus === "live" && `LIVE · ${detections.length} objects`}
          {aiStatus === "waiting" && "WAITING FOR CAMERA..."}
          {aiStatus === "error" && "AI ERROR — is backend running on 8000? Click STOP AI then START AI."}
        </span>
        <div style={{ display: "flex", gap: "0.5rem", alignItems: "center" }}>
          {aiStatus === "idle" ? (
            <button className="replay-btn" onClick={startDetections}>START AI</button>
          ) : (
            <button className="replay-btn" onClick={stopDetections}>STOP AI</button>
          )}
        </div>
      </div>
      {backendReachable === true && aiStatus === "idle" && (
        <p style={{ margin: "0.25rem 0 0", fontSize: "0.75rem", color: "rgba(255,255,255,0.7)" }}>
          Click <strong>START AI</strong> to turn on the webcam feed and live YOLO object detection.
        </p>
      )}

      {/* 65% live feed | 35% semantic map */}
      <div style={{ display: "flex", flex: 1, minHeight: 0, gap: "0.5rem", padding: "0.5rem 0" }}>
        {/* Left 65%: live feed + overlays */}
        <div className="manual-viewport-wrap" style={{ flex: "0 0 65%", minWidth: 0, display: "flex", flexDirection: "column" }}>
          <div className="viewport-card" style={{ position: "relative", width: "100%", flex: 1, minHeight: 0 }}>
            <img
              ref={imgRef}
              src={
                USE_BACKEND_FEED && backendReachable === true
                  ? `${MAIN_BACKEND}/api/feed/Drone-1/processed?t=${feedTick}`
                  : USE_BACKEND_FEED && (backendReachable === false || backendReachable === null)
                    ? BACKEND_DOWN_PLACEHOLDER
                    : IPHONE_STREAM_URL
              }
              alt="Camera Feed"
              style={{ background: "#000", width: "100%", height: "100%", objectFit: "contain" }}
              onLoad={() => {
                const canvas = canvasRef.current;
                const img = imgRef.current;
                if (canvas && img) {
                  canvas.width = img.clientWidth;
                  canvas.height = img.clientHeight;
                }
              }}
              onError={() => {}}
            />
            {/* X Y Z YAW top-left */}
            <div style={{
              position: "absolute", top: "0.5rem", left: "0.5rem",
              background: "rgba(0,0,0,0.75)", color: "#fff",
              padding: "0.3rem 0.5rem", borderRadius: "4px",
              fontFamily: "monospace", fontSize: "0.75rem",
            }}>
              X {poseDisplay.x.toFixed(2)} · Y {poseDisplay.y.toFixed(2)} · Z {poseDisplay.z.toFixed(2)} · YAW {poseDisplay.yaw.toFixed(0)}°
            </div>
            {/* NPU latency top-right */}
            {USE_BACKEND_FEED && tacticalStatus && (
              <div style={{
                position: "absolute", top: "0.5rem", right: "0.5rem",
                background: "rgba(0,0,0,0.75)", color: "#fff",
                padding: "0.3rem 0.5rem", borderRadius: "4px",
                fontSize: "0.7rem",
              }}>
                {tacticalStatus.yolo_latency_ms != null ? `NPU ${Math.round(tacticalStatus.yolo_latency_ms)}ms` : "NPU —"}
              </div>
            )}
            {/* LIVE bottom-left */}
            {aiStatus === "live" && (
              <div style={{
                position: "absolute", bottom: "0.5rem", left: "0.5rem",
                background: "rgba(0,0,0,0.7)", color: "#fff",
                padding: "0.25rem 0.6rem", borderRadius: "4px",
                fontSize: "0.75rem", fontWeight: 700, letterSpacing: "0.05em",
                display: "flex", alignItems: "center", gap: "0.35rem",
              }}>
                <span style={{ width: 6, height: 6, borderRadius: "50%", background: "#FF3000", display: "inline-block", animation: "pulse 2s infinite" }} />
                LIVE
              </div>
            )}
            <canvas
              ref={canvasRef}
              style={{ position: "absolute", top: 0, left: 0, width: "100%", height: "100%", pointerEvents: "none" }}
            />
          </div>
        </div>

        {/* Right 35%: semantic map (detections in camera view) + detection list */}
        <div style={{ flex: "0 0 35%", minWidth: 0, display: "flex", flexDirection: "column", gap: "0.5rem" }}>
          <span className="ai-panel-label">SEMANTIC MAP</span>
          <div style={{ flex: 1, minHeight: 120, background: "#0a0f19", borderRadius: 4, border: "1px solid rgba(255,255,255,0.1)", overflow: "hidden" }}>
            <canvas
              ref={semanticMapRef}
              width={SEMANTIC_MAP_W}
              height={SEMANTIC_MAP_H}
              style={{ width: "100%", height: "100%", objectFit: "contain", display: "block" }}
            />
          </div>
          <div style={{ fontSize: "0.75rem", color: "rgba(255,255,255,0.8)", display: "flex", flexDirection: "column", gap: "0.2rem" }}>
            {detections.length > 0 ? (
              <span>Detections: {detections.map((d) => d.class).join(", ") || "—"}</span>
            ) : (
              <span>Objects appear here when YOLO detects them</span>
            )}
          </div>
        </div>
      </div>
    </section>
  );
}

