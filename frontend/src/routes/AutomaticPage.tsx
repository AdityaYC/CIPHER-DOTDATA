import { useEffect, useState, useCallback, useRef } from "react";
import { API_BASE_URL } from "../config";

const MAIN_BACKEND = API_BASE_URL || (typeof window !== "undefined" ? window.location.origin : "http://localhost:8000");

const CLASS_COLORS: Record<string, string> = {
  person: "#00ff66",
  bicycle: "#4488ff", car: "#4488ff", motorcycle: "#4488ff",
  bus: "#4488ff", truck: "#4488ff",
  fire: "#ff4444", flame: "#ff4444", smoke: "#ff8800",
  default: "#aaaaaa",
};
const color = (cls: string) => CLASS_COLORS[cls.toLowerCase()] ?? CLASS_COLORS.default;

interface Detection {
  class: string;
  confidence: number;
  bbox: [number, number, number, number];
  distance_meters?: number | null;
}

interface VideoJob {
  job_id: string;
  status: "idle" | "running" | "complete" | "error";
  current: number;
  total: number;
  message: string;
  error?: string;
  video_url?: string;
  fps?: number;
  total_frames?: number;
  summary?: { objects_found?: Record<string, number> };
  detections_by_frame?: Detection[][];
}

export function AutomaticPage() {
  const [jobId, setJobId] = useState<string | null>(null);
  const [job, setJob] = useState<VideoJob | null>(null);
  const [uploading, setUploading] = useState(false);
  const [frame, setFrame] = useState(0);
  const [pdfError, setPdfError] = useState<string | null>(null);

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const totalFrames = job?.status === "complete" && job.total_frames != null
    ? Math.max(1, job.total_frames) : 0;
  const frameIdx = Math.max(0, Math.min(frame, Math.max(0, totalFrames - 1)));

  // Poll for job status
  useEffect(() => {
    if (!jobId) return;
    const poll = async () => {
      try {
        const r = await fetch(`${MAIN_BACKEND}/api/video/analysis/${jobId}`);
        const d: VideoJob = await r.json();
        setJob(d);
        if (d.status === "complete" || d.status === "error") {
          clearInterval(pollRef.current!); pollRef.current = null;
        }
      } catch {}
    };
    poll();
    pollRef.current = setInterval(poll, 600);
    return () => { if (pollRef.current) clearInterval(pollRef.current); };
  }, [jobId]);

  // Upload + start analysis
  const startAnalysis = useCallback(async (file: File) => {
    setUploading(true); setJob(null); setJobId(null); setPdfError(null); setFrame(0);
    try {
      const form = new FormData();
      form.append("file", file);
      const r = await fetch(`${MAIN_BACKEND}/api/video/analyze?use_depth=false`, { method: "POST", body: form });
      if (!r.ok) {
        const e = await r.json().catch(() => ({}));
        setJob({ job_id: "", status: "error", current: 0, total: 0, message: "", error: (e.detail as string) || r.statusText });
        return;
      }
      const { job_id } = await r.json();
      setJobId(job_id);
    } finally { setUploading(false); }
  }, []);

  // Draw YOLO boxes on canvas over the current frame
  const drawOverlay = useCallback(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || !job?.detections_by_frame) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dets = job.detections_by_frame[Math.min(frameIdx, job.detections_by_frame.length - 1)] ?? [];
    const vw = video.videoWidth || 1, vh = video.videoHeight || 1;
    const elW = video.clientWidth, elH = video.clientHeight;
    if (elW <= 0 || elH <= 0) return;
    canvas.width = elW; canvas.height = elH;

    // letterbox math — account for object-fit:contain padding
    const videoAspect = vw / vh, elAspect = elW / elH;
    let cx: number, cy: number, cw: number, ch: number;
    if (elAspect > videoAspect) {
      ch = elH; cw = elH * videoAspect; cx = (elW - cw) / 2; cy = 0;
    } else {
      cw = elW; ch = elW / videoAspect; cx = 0; cy = (elH - ch) / 2;
    }
    const sx = cw / vw, sy = ch / vh;

    ctx.clearRect(0, 0, elW, elH);
    dets.forEach((det) => {
      const [x1, y1, x2, y2] = det.bbox.map(Number);
      const bx = cx + x1 * sx, by = cy + y1 * sy;
      const bw = (x2 - x1) * sx, bh = (y2 - y1) * sy;
      const c = color(det.class);
      ctx.strokeStyle = c; ctx.lineWidth = 2.5;
      ctx.strokeRect(bx, by, bw, bh);
      const label = `${det.class} ${Math.round(det.confidence * 100)}%`;
      ctx.font = "bold 12px Inter, sans-serif";
      const tw = ctx.measureText(label).width + 8;
      ctx.fillStyle = c;
      ctx.fillRect(bx, by - 20, Math.min(tw, bw + 24), 20);
      ctx.fillStyle = "#000";
      ctx.fillText(label, bx + 4, by - 5);
    });
  }, [job, frameIdx]);

  // Seek video to current frame and pause
  useEffect(() => {
    const video = videoRef.current;
    if (!video || job?.status !== "complete" || !job.fps) return;
    const t = frameIdx / job.fps;
    video.pause();
    if (Math.abs(video.currentTime - t) > 0.04) video.currentTime = t;
  }, [frameIdx, job?.status, job?.fps]);

  // Attach video events for overlay redraw
  useEffect(() => {
    const video = videoRef.current;
    if (!video || job?.status !== "complete") return;
    const redraw = () => drawOverlay();
    video.addEventListener("seeked", redraw);
    video.addEventListener("loadeddata", redraw);
    video.addEventListener("loadedmetadata", redraw);
    window.addEventListener("resize", redraw);
    drawOverlay();
    return () => {
      video.removeEventListener("seeked", redraw);
      video.removeEventListener("loadeddata", redraw);
      video.removeEventListener("loadedmetadata", redraw);
      window.removeEventListener("resize", redraw);
    };
  }, [job?.status, frameIdx, drawOverlay]);

  // Pause on load and reset frame
  useEffect(() => {
    if (job?.status === "complete") { setFrame(0); videoRef.current?.pause(); }
  }, [job?.job_id, job?.status]);

  // Arrow keys
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (job?.status !== "complete") return;
      if (e.key === "ArrowLeft")  { e.preventDefault(); setFrame(f => Math.max(0, f - 1)); }
      if (e.key === "ArrowRight") { e.preventDefault(); setFrame(f => Math.min(totalFrames - 1, f + 1)); }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [job?.status, totalFrames]);

  const isReady = job?.status === "complete" && !!job.video_url;
  const isRunning = uploading || job?.status === "running";
  const progress = isRunning && job?.total
    ? Math.round((job.current / job.total) * 100) : (uploading ? 100 : 0);
  const objectsList = job?.summary?.objects_found
    ? Object.entries(job.summary.objects_found).sort((a, b) => b[1] - a[1]) : [];
  const detCount = job?.detections_by_frame?.[frameIdx]?.length ?? 0;

  return (
    <section style={{ display: "flex", flexDirection: "column", height: "100%", overflow: "hidden", background: "#0a0f19" }}>

      {/* ── Top bar: upload + status ── */}
      <div style={{
        display: "flex", alignItems: "center", gap: "0.75rem", flexWrap: "wrap",
        padding: "0.6rem 1rem", borderBottom: "1px solid rgba(255,255,255,0.08)",
        background: "rgba(0,0,0,0.3)", flexShrink: 0,
      }}>
        <span style={{ fontWeight: 700, fontSize: "0.9rem", color: "#fff", letterSpacing: "0.05em" }}>
          AUTOMATIC — VIDEO ANALYSIS
        </span>

        <label style={{ cursor: isRunning ? "not-allowed" : "pointer", display: "flex", alignItems: "center" }}>
          <input type="file" accept=".mp4,.avi,.mov,.mkv,.webm" disabled={isRunning}
            style={{ position: "absolute", opacity: 0, width: 0, height: 0 }}
            onChange={e => { const f = e.target.files?.[0]; if (f) startAnalysis(f); e.target.value = ""; }}
          />
          <span style={{
            padding: "0.35rem 0.9rem", borderRadius: 6, fontWeight: 600, fontSize: "0.85rem",
            background: isRunning ? "rgba(0,255,102,0.25)" : "#00ff66",
            color: isRunning ? "rgba(255,255,255,0.5)" : "#0a0f19",
          }}>
            {uploading ? "Uploading…" : isRunning ? "Analysing…" : isReady ? "Upload new" : "Choose video"}
          </span>
        </label>

        {isRunning && (
          <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", flex: 1, minWidth: 160 }}>
            <div style={{ flex: 1, height: 8, background: "rgba(255,255,255,0.1)", borderRadius: 4, overflow: "hidden" }}>
              <div style={{
                height: "100%", borderRadius: 4, background: "linear-gradient(90deg,#00ff66,#00cc55)",
                width: `${progress}%`, transition: "width 0.3s",
              }} />
            </div>
            <span style={{ fontSize: "0.78rem", color: "rgba(255,255,255,0.65)", whiteSpace: "nowrap" }}>
              {uploading ? "Uploading…" : `${job?.current ?? 0} / ${job?.total ?? 0} frames · ${progress}%`}
            </span>
          </div>
        )}

        {isReady && !isRunning && (
          <span style={{ fontSize: "0.82rem", color: "#00ff66" }}>
            ✓ {job.total_frames} frames · {objectsList.length} classes · {job.fps?.toFixed(0)} fps
          </span>
        )}
        {job?.status === "error" && (
          <span style={{ fontSize: "0.82rem", color: "#ff6b6b" }}>{job.error ?? "Error"}</span>
        )}
      </div>

      {/* ── Main: video + canvas overlay ── */}
      <div style={{ flex: 1, minHeight: 0, position: "relative", background: "#000", display: "flex", alignItems: "center", justifyContent: "center" }}>
        {isReady ? (
          <>
            <video
              ref={videoRef}
              src={`${MAIN_BACKEND}${job!.video_url}`}
              style={{ width: "100%", height: "100%", objectFit: "contain", display: "block" }}
              playsInline preload="auto"
              onLoadedMetadata={() => { videoRef.current?.pause(); drawOverlay(); }}
              onSeeked={drawOverlay}
            />
            <canvas ref={canvasRef} style={{
              position: "absolute", top: 0, left: 0, width: "100%", height: "100%", pointerEvents: "none",
            }} />

            {/* Frame counter + detections overlay */}
            <div style={{
              position: "absolute", top: 10, left: 10,
              background: "rgba(0,0,0,0.65)", padding: "0.3rem 0.6rem", borderRadius: 6,
              fontFamily: "monospace", fontSize: "0.82rem", color: "#fff",
            }}>
              Frame {frameIdx + 1} / {totalFrames}
              {detCount > 0 && <span style={{ marginLeft: 8, color: "#00ff66" }}>{detCount} detection{detCount !== 1 ? "s" : ""}</span>}
            </div>

            {/* Objects summary overlay — top right */}
            {objectsList.length > 0 && (
              <div style={{
                position: "absolute", top: 10, right: 10,
                background: "rgba(0,0,0,0.7)", padding: "0.5rem 0.75rem", borderRadius: 8,
                fontSize: "0.78rem", color: "#fff", maxHeight: "40vh", overflowY: "auto", minWidth: 140,
              }}>
                <div style={{ fontWeight: 700, marginBottom: 4, letterSpacing: "0.05em", color: "rgba(255,255,255,0.55)" }}>OBJECTS DETECTED</div>
                {objectsList.map(([cls, cnt]) => (
                  <div key={cls} style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 3 }}>
                    <span style={{ width: 8, height: 8, borderRadius: 2, background: color(cls), flexShrink: 0 }} />
                    <span style={{ fontWeight: 600 }}>{cls}</span>
                    <span style={{ color: "rgba(255,255,255,0.5)", marginLeft: "auto", paddingLeft: 8 }}>×{cnt}</span>
                  </div>
                ))}
              </div>
            )}
          </>
        ) : (
          <div style={{ color: "rgba(255,255,255,0.35)", fontSize: "0.95rem", textAlign: "center", padding: "2rem" }}>
            {isRunning
              ? `Analysing… ${job?.current ?? 0} / ${job?.total ?? 0} frames`
              : "Upload a video to begin frame-by-frame YOLO analysis"}
          </div>
        )}
      </div>

      {/* ── Bottom controls ── */}
      {isReady && (
        <div style={{
          flexShrink: 0, padding: "0.5rem 1rem", background: "rgba(0,0,0,0.4)",
          borderTop: "1px solid rgba(255,255,255,0.1)",
          display: "flex", flexDirection: "column", gap: "0.4rem",
        }}>
          {/* Scrubber */}
          <input
            type="range" min={0} max={Math.max(0, totalFrames - 1)} value={frameIdx}
            onChange={e => setFrame(Number(e.target.value))}
            style={{ width: "100%", accentColor: "#00ff66", cursor: "pointer" }}
          />

          {/* Buttons row */}
          <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", flexWrap: "wrap" }}>
            <button onClick={() => setFrame(f => Math.max(0, f - 1))} disabled={frameIdx <= 0}
              style={btnStyle(frameIdx > 0)}>← Prev</button>
            <button onClick={() => setFrame(f => Math.min(totalFrames - 1, f + 1))} disabled={frameIdx >= totalFrames - 1}
              style={btnStyle(frameIdx < totalFrames - 1)}>Next →</button>

            <span style={{ fontSize: "0.78rem", color: "rgba(0,255,102,0.7)", marginLeft: 4 }}>← → keys</span>

            <div style={{ marginLeft: "auto", display: "flex", gap: "0.5rem", alignItems: "center" }}>
              {pdfError && <span style={{ fontSize: "0.78rem", color: "#ff6b6b" }}>{pdfError}</span>}
              <button onClick={async () => {
                setPdfError(null);
                try {
                  const r = await fetch(`${MAIN_BACKEND}/api/video/analysis/${job!.job_id}/report.pdf`);
                  if (!r.ok) { setPdfError("PDF failed"); return; }
                  const blob = await r.blob();
                  const url = URL.createObjectURL(blob);
                  const a = document.createElement("a"); a.href = url;
                  a.download = `cipher_report_${job!.job_id}.pdf`; a.click();
                  URL.revokeObjectURL(url);
                } catch { setPdfError("PDF failed"); }
              }} style={btnStyle(true)}>Download PDF</button>
            </div>
          </div>
        </div>
      )}
    </section>
  );
}

function btnStyle(active: boolean): React.CSSProperties {
  return {
    padding: "0.35rem 0.7rem", borderRadius: 6, fontWeight: 600, fontSize: "0.85rem",
    border: `1px solid ${active ? "rgba(0,255,102,0.5)" : "rgba(255,255,255,0.1)"}`,
    background: active ? "rgba(0,255,102,0.15)" : "transparent",
    color: active ? "#00ff66" : "rgba(255,255,255,0.3)",
    cursor: active ? "pointer" : "not-allowed",
  };
}
