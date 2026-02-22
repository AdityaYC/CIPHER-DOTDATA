import { useEffect, useState, useCallback, useRef } from "react";
import { API_BASE_URL } from "../config";

const MAIN_BACKEND = API_BASE_URL || (typeof window !== "undefined" ? window.location.origin : "http://localhost:8000");

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

interface Detection {
  class: string;
  confidence: number;
  bbox: [number, number, number, number];
  distance_meters?: number | null;
}

interface VideoAnalysisJob {
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
  const [videoJobId, setVideoJobId] = useState<string | null>(null);
  const [videoJob, setVideoJob] = useState<VideoAnalysisJob | null>(null);
  const [videoUploading, setVideoUploading] = useState(false);
  const [currentFrame, setCurrentFrame] = useState(0);
  const [pdfError, setPdfError] = useState<string | null>(null);
  const videoPlaybackRef = useRef<HTMLVideoElement>(null);
  const videoOverlayRef = useRef<HTMLCanvasElement>(null);
  const videoPollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!videoJobId || !MAIN_BACKEND) return;
    const poll = async () => {
      try {
        const r = await fetch(`${MAIN_BACKEND}/api/video/analysis/${videoJobId}`);
        const data: VideoAnalysisJob = await r.json();
        setVideoJob(data);
        if (data.status === "complete" || data.status === "error") {
          if (videoPollRef.current) {
            clearInterval(videoPollRef.current);
            videoPollRef.current = null;
          }
        }
      } catch {}
    };
    poll();
    videoPollRef.current = setInterval(poll, 800);
    return () => {
      if (videoPollRef.current) {
        clearInterval(videoPollRef.current);
        videoPollRef.current = null;
      }
    };
  }, [videoJobId, MAIN_BACKEND]);

  const startVideoAnalysis = useCallback(async (file: File) => {
    if (!MAIN_BACKEND) return;
    setVideoUploading(true);
    setVideoJob(null);
    setVideoJobId(null);
    setPdfError(null);
    try {
      const form = new FormData();
      form.append("file", file);
      const r = await fetch(`${MAIN_BACKEND}/api/video/analyze?use_depth=false`, {
        method: "POST",
        body: form,
      });
      if (!r.ok) {
        const err = await r.json().catch(() => ({}));
        setVideoJob({
          job_id: "",
          status: "error",
          current: 0,
          total: 0,
          message: "",
          error: (err.detail as string) || r.statusText,
        });
        return;
      }
      const { job_id } = await r.json();
      setVideoJobId(job_id);
    } finally {
      setVideoUploading(false);
    }
  }, [MAIN_BACKEND]);

  const totalFrames = videoJob?.status === "complete" && videoJob?.total_frames != null
    ? Math.max(0, videoJob.total_frames - 1)
    : 0;
  const frameIndex = Math.max(0, Math.min(currentFrame, totalFrames));

  const drawVideoOverlay = useCallback(() => {
    const video = videoPlaybackRef.current;
    const canvas = videoOverlayRef.current;
    const job = videoJob;
    if (!video || !canvas || !job || job.status !== "complete" || !job.detections_by_frame) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const idx = Math.min(frameIndex, job.detections_by_frame.length - 1);
    const dets = job.detections_by_frame[idx] ?? [];
    const vw = video.videoWidth || 1;
    const vh = video.videoHeight || 1;
    const elW = video.clientWidth;
    const elH = video.clientHeight;
    if (elW <= 0 || elH <= 0) return;
    canvas.width = elW;
    canvas.height = elH;
    canvas.style.width = `${elW}px`;
    canvas.style.height = `${elH}px`;
    ctx.clearRect(0, 0, elW, elH);
    if (idx < 0 || dets.length === 0) return;
    const videoAspect = vw / vh;
    const elAspect = elW / elH;
    let contentX: number, contentY: number, contentW: number, contentH: number;
    if (elAspect > videoAspect) {
      contentH = elH;
      contentW = elH * videoAspect;
      contentX = (elW - contentW) / 2;
      contentY = 0;
    } else {
      contentW = elW;
      contentH = elW / videoAspect;
      contentX = 0;
      contentY = (elH - contentH) / 2;
    }
    const scaleX = contentW / vw;
    const scaleY = contentH / vh;
    dets.forEach((det) => {
      const bbox = det.bbox;
      if (!bbox || bbox.length < 4) return;
      const x1 = Number(bbox[0]);
      const y1 = Number(bbox[1]);
      const x2 = Number(bbox[2]);
      const y2 = Number(bbox[3]);
      const bx = contentX + x1 * scaleX;
      const by = contentY + y1 * scaleY;
      const bw = (x2 - x1) * scaleX;
      const bh = (y2 - y1) * scaleY;
      const distStr = det.distance_meters != null ? ` · ${Math.round(det.distance_meters * 100 / 25)}` : "";
      const label = `${det.class}${distStr}`;
      ctx.strokeStyle = getSemanticClassColor(det.class);
      ctx.lineWidth = 3;
      ctx.strokeRect(bx, by, bw, bh);
      ctx.font = "bold 13px Inter, sans-serif";
      const textW = Math.min(ctx.measureText(label).width + 10, bw + 20);
      ctx.fillStyle = getSemanticClassColor(det.class);
      ctx.fillRect(bx, by - 20, textW, 20);
      ctx.fillStyle = "#fff";
      ctx.fillText(label, bx + 5, by - 5);
    });
  }, [videoJob, frameIndex]);

  // Sync video time to current frame
  useEffect(() => {
    const video = videoPlaybackRef.current;
    if (!video || videoJob?.status !== "complete" || !videoJob?.fps) return;
    const t = frameIndex / videoJob.fps;
    if (Math.abs(video.currentTime - t) > 0.05) {
      video.currentTime = t;
    }
  }, [frameIndex, videoJob?.status, videoJob?.fps]);

  // Redraw overlay when frame or video size changes; also after video loads metadata
  useEffect(() => {
    const video = videoPlaybackRef.current;
    if (!video || videoJob?.status !== "complete") return;

    const onSeeked = () => drawVideoOverlay();
    const onLoaded = () => drawVideoOverlay();
    const onResize = () => drawVideoOverlay();

    video.addEventListener("seeked", onSeeked);
    video.addEventListener("loadeddata", onLoaded);
    video.addEventListener("loadedmetadata", onLoaded);
    window.addEventListener("resize", onResize);
    drawVideoOverlay();

    return () => {
      video.removeEventListener("seeked", onSeeked);
      video.removeEventListener("loadeddata", onLoaded);
      video.removeEventListener("loadedmetadata", onLoaded);
      window.removeEventListener("resize", onResize);
    };
  }, [videoJob?.status, frameIndex, drawVideoOverlay]);

  // Reset to frame 0 when new video is ready
  useEffect(() => {
    if (videoJob?.status === "complete" && videoJob?.total_frames != null) {
      setCurrentFrame(0);
    }
  }, [videoJob?.job_id, videoJob?.status]);

  // Arrow keys: frame by frame (only when video is ready and container is focused or document)
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (videoJob?.status !== "complete" || videoJob?.total_frames == null) return;
      const total = Math.max(0, videoJob.total_frames - 1);
      if (e.key === "ArrowLeft") {
        e.preventDefault();
        setCurrentFrame((f) => Math.max(0, f - 1));
      } else if (e.key === "ArrowRight") {
        e.preventDefault();
        setCurrentFrame((f) => Math.min(total, f + 1));
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [videoJob?.status, videoJob?.total_frames]);

  const objectsList = videoJob?.status === "complete" && videoJob?.summary?.objects_found
    ? Object.entries(videoJob.summary.objects_found).sort((a, b) => b[1] - a[1])
    : [];

  return (
    <section
      className="manual-page"
      style={{
        padding: "clamp(0.75rem, 2vw, 1.25rem)",
        display: "flex",
        flexDirection: "column",
        minHeight: 0,
        flex: 1,
      }}
    >
      <h2 style={{ margin: "0 0 0.5rem", fontSize: "clamp(1.1rem, 2.5vw, 1.35rem)", color: "rgba(255,255,255,0.95)" }}>
        Automatic — Upload video
      </h2>
      <p style={{ margin: "0 0 1rem", fontSize: "0.9rem", color: "rgba(255,255,255,0.8)", maxWidth: "56ch" }}>
        Upload a video to run YOLO object detection. Step through frames with the controls below; download a PDF report when done.
      </p>

      <div style={{ display: "flex", flex: 1, minHeight: 0, gap: "clamp(0.75rem, 2vw, 1.25rem)", flexWrap: "wrap", alignContent: "flex-start" }}>
        {/* Left: upload + playback */}
        <div style={{ flex: "1 1 min(100%, 420px)", minWidth: 0, display: "flex", flexDirection: "column", gap: "0.75rem" }}>
          <div style={{
            background: "rgba(0,0,0,0.35)",
            border: "2px dashed rgba(0,255,102,0.5)",
            borderRadius: 8,
            padding: "1rem",
            display: "flex",
            flexDirection: "column",
            gap: "0.5rem",
          }}>
            <span className="ai-panel-label">UPLOAD YOUR VIDEO</span>
            <label style={{ fontSize: "0.9rem", color: "rgba(255,255,255,0.95)", cursor: videoUploading ? "not-allowed" : "pointer", display: "flex", alignItems: "center", gap: "0.5rem" }}>
              <input
                type="file"
                accept=".mp4,.avi,.mov,.mkv,.webm"
                disabled={videoUploading || videoJob?.status === "running"}
                style={{ position: "absolute", width: 0, height: 0, opacity: 0 }}
                onChange={(e) => {
                  const f = e.target.files?.[0];
                  if (f) startVideoAnalysis(f);
                  e.target.value = "";
                }}
              />
              <span style={{
                padding: "0.4rem 0.75rem",
                background: (videoUploading || videoJob?.status === "running") ? "rgba(0,255,102,0.3)" : "#00ff66",
                color: "#0a0f19", borderRadius: 6, fontWeight: 600,
                cursor: (videoUploading || videoJob?.status === "running") ? "not-allowed" : "pointer",
              }}>
                {videoUploading ? "Uploading…" : videoJob?.status === "running" ? "Analysing…" : "Choose file"}
              </span>
              <span style={{ color: "rgba(255,255,255,0.6)", fontSize: "0.82rem" }}>MP4, AVI, MOV, MKV, WEBM</span>
            </label>

            {/* Progress bar */}
            {(videoUploading || videoJob?.status === "running") && (
              <div style={{ display: "flex", flexDirection: "column", gap: "0.35rem" }}>
                <div style={{
                  width: "100%", height: 10, background: "rgba(255,255,255,0.1)",
                  borderRadius: 6, overflow: "hidden",
                }}>
                  <div style={{
                    height: "100%",
                    borderRadius: 6,
                    background: "linear-gradient(90deg, #00ff66, #00cc55)",
                    transition: "width 0.3s ease",
                    width: videoUploading
                      ? "100%"
                      : videoJob && videoJob.total > 0
                        ? `${Math.round((videoJob.current / videoJob.total) * 100)}%`
                        : "5%",
                    animation: videoUploading ? "pulse-bar 1.2s ease-in-out infinite" : "none",
                  }} />
                </div>
                <span style={{ fontSize: "0.8rem", color: "rgba(255,255,255,0.75)" }}>
                  {videoUploading
                    ? "Uploading video…"
                    : videoJob && videoJob.total > 0
                      ? `Analysing frame ${videoJob.current} of ${videoJob.total} — ${Math.round((videoJob.current / videoJob.total) * 100)}%`
                      : "Starting analysis…"}
                </span>
              </div>
            )}

            {videoJob?.status === "complete" && (
              <span style={{ fontSize: "0.82rem", color: "#00ff66" }}>
                ✓ Analysis complete — {videoJob.total_frames} frames, {Object.keys(videoJob.summary?.objects_found ?? {}).length} object classes
              </span>
            )}
            {videoJob?.status === "error" && (
              <span style={{ fontSize: "0.8rem", color: "#ff6b6b" }}>{videoJob.error ?? "Analysis failed"}</span>
            )}
            {videoJob?.status === "complete" && videoJob.video_url && (
              <div ref={containerRef} style={{ display: "flex", flexDirection: "column", gap: "0.5rem" }}>
                <div style={{ position: "relative", width: "100%", background: "#000", borderRadius: 4, overflow: "hidden" }}>
                  <video
                    ref={videoPlaybackRef}
                    src={`${MAIN_BACKEND}${videoJob.video_url}`}
                    style={{ width: "100%", display: "block", objectFit: "contain", minHeight: 200, maxHeight: "70vh" }}
                    crossOrigin="anonymous"
                    playsInline
                    controls
                    onLoadedMetadata={() => drawVideoOverlay()}
                    onSeeked={() => drawVideoOverlay()}
                  />
                  <canvas
                    ref={videoOverlayRef}
                    style={{
                      position: "absolute",
                      top: 0,
                      left: 0,
                      width: "100%",
                      height: "100%",
                      pointerEvents: "none",
                    }}
                  />
                </div>
                <div
                  style={{
                    display: "flex",
                    alignItems: "center",
                    flexWrap: "wrap",
                    gap: "0.75rem",
                    padding: "0.5rem 0.75rem",
                    background: "rgba(0,0,0,0.25)",
                    borderRadius: 8,
                    border: "1px solid rgba(255,255,255,0.1)",
                  }}
                >
                  <div style={{ display: "flex", alignItems: "center", gap: "0.25rem" }}>
                    <button
                      type="button"
                      aria-label="Previous frame"
                      disabled={frameIndex <= 0}
                      onClick={() => setCurrentFrame((f) => Math.max(0, f - 1))}
                      style={{
                        padding: "0.4rem 0.6rem",
                        background: frameIndex <= 0 ? "rgba(255,255,255,0.1)" : "rgba(0,255,102,0.2)",
                        border: "1px solid rgba(0,255,102,0.5)",
                        borderRadius: 6,
                        color: frameIndex <= 0 ? "rgba(255,255,255,0.4)" : "#00ff66",
                        cursor: frameIndex <= 0 ? "not-allowed" : "pointer",
                        fontWeight: 600,
                        fontSize: "0.9rem",
                      }}
                    >
                      ← Prev
                    </button>
                    <button
                      type="button"
                      aria-label="Next frame"
                      disabled={frameIndex >= totalFrames}
                      onClick={() => setCurrentFrame((f) => Math.min(totalFrames, f + 1))}
                      style={{
                        padding: "0.4rem 0.6rem",
                        background: frameIndex >= totalFrames ? "rgba(255,255,255,0.1)" : "rgba(0,255,102,0.2)",
                        border: "1px solid rgba(0,255,102,0.5)",
                        borderRadius: 6,
                        color: frameIndex >= totalFrames ? "rgba(255,255,255,0.4)" : "#00ff66",
                        cursor: frameIndex >= totalFrames ? "not-allowed" : "pointer",
                        fontWeight: 600,
                        fontSize: "0.9rem",
                      }}
                    >
                      Next →
                    </button>
                  </div>
                  <span style={{ fontSize: "0.9rem", color: "rgba(255,255,255,0.9)" }}>
                    Frame <strong>{frameIndex + 1}</strong> / {videoJob.total_frames ?? 0}
                    {videoJob.detections_by_frame?.[frameIndex]?.length != null && (
                      <span style={{ marginLeft: "0.35rem", color: "rgba(255,255,255,0.65)" }}>
                        · {videoJob.detections_by_frame[frameIndex].length} in frame
                      </span>
                    )}
                  </span>
                  <span style={{ fontSize: "0.75rem", color: "rgba(0,255,102,0.75)" }}>
                    ← → keys
                  </span>
                </div>
                <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", flexWrap: "wrap" }}>
                  <button
                    type="button"
                    onClick={async () => {
                      setPdfError(null);
                      try {
                        const r = await fetch(`${MAIN_BACKEND}/api/video/analysis/${videoJob.job_id}/report.pdf`);
                        if (!r.ok) {
                          const err = await r.json().catch(() => ({}));
                          setPdfError((err.detail as string) || "PDF download failed");
                          return;
                        }
                        const blob = await r.blob();
                        const url = URL.createObjectURL(blob);
                        const a = document.createElement("a");
                        a.href = url;
                        a.download = `cipher_video_report_${videoJob.job_id}.pdf`;
                        a.click();
                        URL.revokeObjectURL(url);
                      } catch {
                        setPdfError("PDF download failed");
                      }
                    }}
                    style={{
                      fontSize: "0.9rem",
                      color: "#0a0f19",
                      background: "#00ff66",
                      border: "none",
                      borderRadius: 6,
                      padding: "0.4rem 0.75rem",
                      fontWeight: 600,
                      cursor: "pointer",
                    }}
                  >
                    Download PDF report
                  </button>
                  {pdfError && (
                    <span style={{ fontSize: "0.8rem", color: "#ff6b6b" }}>{pdfError}</span>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Right: list of objects detected */}
        <div style={{ flex: "1 1 260px", minWidth: "min(100%, 260px)", maxWidth: 320, display: "flex", flexDirection: "column", gap: "0.5rem" }}>
          <span className="ai-panel-label">OBJECTS DETECTED</span>
          <div style={{
            background: "rgba(0,0,0,0.35)",
            border: "1px solid rgba(255,255,255,0.12)",
            borderRadius: 8,
            padding: "0.75rem",
            maxHeight: "min(320px, 40vh)",
            overflowY: "auto",
            flex: "1 1 auto",
            minHeight: 0,
          }}>
            {objectsList.length === 0 && !videoJob?.summary && (
              <p style={{ margin: 0, fontSize: "0.85rem", color: "rgba(255,255,255,0.6)" }}>
                Upload and analyze a video to see the list of objects found (with max count per frame).
              </p>
            )}
            {videoJob?.status === "running" && (
              <p style={{ margin: 0, fontSize: "0.85rem", color: "rgba(255,255,255,0.8)" }}>
                Building list…
              </p>
            )}
            {videoJob?.status === "error" && (
              <p style={{ margin: 0, fontSize: "0.85rem", color: "#ff6b6b" }}>
                {videoJob.error ?? "Analysis failed"}
              </p>
            )}
            {objectsList.length > 0 && (
              <ul style={{ margin: 0, paddingLeft: "1.25rem", fontSize: "0.9rem", color: "rgba(255,255,255,0.9)" }}>
                {objectsList.map(([cls, count]) => (
                  <li key={cls} style={{ marginBottom: "0.35rem", display: "flex", alignItems: "center", gap: "0.5rem" }}>
                    <span
                      style={{
                        width: 10,
                        height: 10,
                        borderRadius: 2,
                        background: getSemanticClassColor(cls),
                        flexShrink: 0,
                      }}
                    />
                    <strong>{cls}</strong>
                    <span style={{ color: "rgba(255,255,255,0.7)" }}>
                      (max {count} in a frame)
                    </span>
                  </li>
                ))}
              </ul>
            )}
          </div>
        </div>
      </div>
    </section>
  );
}
