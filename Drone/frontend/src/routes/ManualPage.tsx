import { useEffect, useMemo, useState, useCallback, useRef } from "react";
import { IPHONE_STREAM_URL, API_BASE_URL, BACKEND_FEED_BASE } from "../config";
import { fetchTacticalStatus, type TacticalStatus } from "../api/tactical";

const MAIN_BACKEND = API_BASE_URL || (typeof window !== "undefined" ? window.location.origin : "http://localhost:8000");
const USE_BACKEND_FEED = BACKEND_FEED_BASE !== undefined && BACKEND_FEED_BASE !== null;

/** Placeholder when backend is down — avoids ERR_CONNECTION_REFUSED spam */
const BACKEND_DOWN_PLACEHOLDER =
  "data:image/svg+xml," + encodeURIComponent(
    '<svg xmlns="http://www.w3.org/2000/svg" width="640" height="480" viewBox="0 0 640 480"><rect fill="#0a0f19" width="640" height="480"/><text x="320" y="220" fill="rgba(255,255,255,0.9)" font-family="sans-serif" font-size="18" text-anchor="middle">Backend not running on port 8000</text><text x="320" y="260" fill="rgba(255,255,255,0.6)" font-family="sans-serif" font-size="14" text-anchor="middle">Start it from repo root:</text><text x="320" y="295" fill="#00ff66" font-family="monospace" font-size="13" text-anchor="middle">.\\run_drone_full.ps1</text><text x="320" y="330" fill="rgba(255,255,255,0.5)" font-family="sans-serif" font-size="12" text-anchor="middle">(or open a terminal and run the backend, then refresh)</text></svg>'
  );

// Overhead map: world graph nodes. Colors by category
const MAP_NODE_COLORS: Record<string, string> = {
  survivor: "#00ff66",
  hazard: "#ff3333",
  structural: "#ff8800",
  exit: "#3388ff",
  clear: "#888888",
  unknown: "#888888",
};

interface Detection {
  class: string;
  confidence: number;
  bbox: [number, number, number, number]; // x1,y1,x2,y2 in pixels
  depth_m?: number; // optional, show on box when present
}

type GraphNode = { node_id: string; pose: [number, number, number] | null; detections?: Array<{ class_name: string; confidence: number; category?: string }> };

export function ManualPage() {
  const [detections, setDetections] = useState<Detection[]>([]);
  const [aiStatus, setAiStatus] = useState<"idle" | "connecting" | "live" | "waiting" | "error">("idle");
  const imgRef = useRef<HTMLImageElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const sseRef = useRef<EventSource | null>(null);
  const mapCanvasRef = useRef<HTMLCanvasElement>(null);
  const [graphData, setGraphData] = useState<{ nodes: GraphNode[]; path: Array<{ x: number; y: number; z: number }>; stats?: { node_count: number; area_m2: number } } | null>(null);

  const fetchGraph3d = useCallback(async () => {
    try {
      const r = await fetch(`${MAIN_BACKEND}/api/graph_3d`);
      if (!r.ok) return;
      const data = await r.json();
      setGraphData({ nodes: data.nodes ?? [], path: data.path ?? [], stats: data.stats });
    } catch {
      setGraphData(null);
    }
  }, [MAIN_BACKEND]);
  useEffect(() => {
    fetchGraph3d();
    const t = setInterval(fetchGraph3d, 2000);
    return () => clearInterval(t);
  }, [fetchGraph3d]);

  // Backend reachable: when false, show placeholder and don't hammer 8000 (fixes ERR_CONNECTION_REFUSED spam)
  const [backendReachable, setBackendReachable] = useState<boolean | null>(null);
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
    const t = setInterval(() => setFeedTick((n) => n + 1), 250);
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

      const conf = Math.round(det.confidence * 100);
      const depthStr = det.depth_m != null ? ` ${det.depth_m.toFixed(1)}m` : "";
      const label = `${det.class} ${conf}%${depthStr}`;

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

  // Overhead tactical map: world graph nodes as colored dots, path as white line, current as pulsing white
  useEffect(() => {
    const canvas = mapCanvasRef.current;
    if (!canvas || !graphData?.nodes?.length) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const W = canvas.width = 320;
    const H = canvas.height = 200;
    ctx.fillStyle = "#0a0f19";
    ctx.fillRect(0, 0, W, H);
    const path = graphData.path || [];
    const nodes = graphData.nodes;
    const xs = path.length ? path.map((p) => p.x) : nodes.map((n) => n.pose?.[0] ?? 0).filter((v) => typeof v === "number");
    const ys = path.length ? path.map((p) => p.y) : nodes.map((n) => n.pose?.[1] ?? 0).filter((v) => typeof v === "number");
    const minX = Math.min(...xs, 0);
    const maxX = Math.max(...xs, 0);
    const minY = Math.min(...ys, 0);
    const maxY = Math.max(...ys, 0);
    const pad = 20;
    const rangeX = maxX - minX || 1;
    const rangeY = maxY - minY || 1;
    const scale = Math.min((W - pad * 2) / rangeX, (H - pad * 2) / rangeY);
    const cx = (minX + maxX) / 2;
    const cy = (minY + maxY) / 2;
    const toScreen = (x: number, y: number) => ({
      sx: W / 2 + (x - cx) * scale,
      sy: H / 2 - (y - cy) * scale,
    });
    if (path.length >= 2) {
      ctx.strokeStyle = "#fff";
      ctx.lineWidth = 1;
      ctx.beginPath();
      const { sx, sy } = toScreen(path[0].x, path[0].y);
      ctx.moveTo(sx, sy);
      for (let i = 1; i < path.length; i++) {
        const { sx: px, sy: py } = toScreen(path[i].x, path[i].y);
        ctx.lineTo(px, py);
      }
      ctx.stroke();
    }
    const lastId = nodes[nodes.length - 1]?.node_id;
    nodes.forEach((n) => {
      const pos = n.pose;
      if (!pos) return;
      const { sx, sy } = toScreen(pos[0], pos[1]);
      const cat = n.detections?.[0]?.category ?? "unknown";
      const color = MAP_NODE_COLORS[cat] ?? MAP_NODE_COLORS.unknown;
      const isCurrent = n.node_id === lastId;
      ctx.fillStyle = isCurrent ? "#fff" : color;
      ctx.beginPath();
      ctx.arc(sx, sy, isCurrent ? 5 : 3, 0, Math.PI * 2);
      ctx.fill();
      if (isCurrent) {
        ctx.strokeStyle = "#fff";
        ctx.lineWidth = 2;
        ctx.stroke();
      }
    });
  }, [graphData]);

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

      {/* 65% live feed | 35% overhead map */}
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

        {/* Right 35%: overhead tactical map + node count, area, detection summary */}
        <div style={{ flex: "0 0 35%", minWidth: 0, display: "flex", flexDirection: "column", gap: "0.5rem" }}>
          <span className="ai-panel-label">OVERHEAD MAP</span>
          <div style={{ flex: 1, minHeight: 120, background: "#0a0f19", borderRadius: 4, border: "1px solid rgba(255,255,255,0.1)", overflow: "hidden" }}>
            <canvas
              ref={mapCanvasRef}
              width={320}
              height={200}
              style={{ width: "100%", height: "100%", objectFit: "contain", display: "block" }}
            />
          </div>
          <div style={{ fontSize: "0.75rem", color: "rgba(255,255,255,0.8)", display: "flex", flexDirection: "column", gap: "0.2rem" }}>
            <span>Nodes: {graphData?.stats?.node_count ?? graphData?.nodes?.length ?? 0}</span>
            <span>Area: {(graphData?.stats?.area_m2 ?? 0).toFixed(1)} m²</span>
            {detections.length > 0 && (
              <span>Detections: {detections.map((d) => d.class).join(", ") || "—"}</span>
            )}
          </div>
        </div>
      </div>
    </section>
  );
}

