import { useEffect, useMemo, useState, useCallback, useRef } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { useAgentSession } from "../hooks/useAgentSession";
import { QueryInput } from "../components/QueryInput";
import { AGENT_API_URL, MAX_AGENT_STEPS } from "../config";

const MAX_CONVERSATION_HISTORY = 5;

type Detection = { class: string; confidence: number };
type LiveFrame = { id: string; ts: number; image_b64: string; detections: Detection[] };

type ConversationEntry = {
  query: string;
  answer: string;
  confidence?: number;
  agent_used?: string;
  node_ids?: string[];
  recommended_action?: string;
};

/** Backend base for Agent (tactical query + stream). Same as AGENT_API_URL so voice_query hits port 8000 in dev. */
const agentApiBase = () => (typeof window !== "undefined" ? AGENT_API_URL || window.location.origin : "http://localhost:8000");

export function AgentPage() {
  const { sessionId: urlSessionId } = useParams<{ sessionId?: string }>();
  const navigate = useNavigate();

  const {
    sessionId,
    sessionStatus,
    agents,
    winnerAgentId,
    error,
    startSession,
    joinSession,
    cancelSession,
  } = useAgentSession();

  // Agent mode: one search bar powers tactical query (manuals + map), answer shown below
  const [agentAnswer, setAgentAnswer] = useState("");
  const [agentNodeIds, setAgentNodeIds] = useState<string[]>([]);
  const [agentLoading, setAgentLoading] = useState(false);
  const [agentError, setAgentError] = useState("");
  const [confidence, setConfidence] = useState<number>(0);
  const [, setAgentUsed] = useState<string>("");
  const [, setRecommendedAction] = useState("");
  const [conversationHistory, setConversationHistory] = useState<ConversationEntry[]>([]);
  const [agentStatus, setAgentStatus] = useState<"ready" | "initializing">("initializing");
  const [nodeThumbnails, setNodeThumbnails] = useState<Record<string, string>>({});
  const [uploadedNodeIds, setUploadedNodeIds] = useState<string[]>([]);
  type CurrentNode = { node_id: string; image_b64: string | null; detections: Array<{ class_name: string; confidence: number }>; structural_risk_score: number } | null;
  const [currentNode, setCurrentNode] = useState<CurrentNode>(null);
  const [vOverlayOn, setVOverlayOn] = useState(false);
  const [importStatus, setImportStatus] = useState<{ status: string; current: number; total: number; message: string; nodes_added?: number } | null>(null);
  const [uploadedVideoThumbnail, setUploadedVideoThumbnail] = useState<string | null>(null);
  const videoImportInputRef = useRef<HTMLInputElement>(null);
  const [graphRefreshKey] = useState(0);

  // Live YOLO frame gallery (multi-select with fresh YOLO on pin)
  const [liveFrames, setLiveFrames] = useState<LiveFrame[]>([]);
  const [pinnedFrameIds, setPinnedFrameIds] = useState<Set<string>>(new Set());
  const [freshDetections, setFreshDetections] = useState<Record<string, Detection[] | "loading">>({});

  const pinnedFrames = liveFrames.filter((f) => pinnedFrameIds.has(f.id));

  const getDetections = (frame: LiveFrame): Detection[] | "loading" => {
    const f = freshDetections[frame.id];
    if (f === "loading") return "loading";
    if (Array.isArray(f)) return f;
    return frame.detections;
  };

  const togglePin = async (frame: LiveFrame) => {
    const alreadyPinned = pinnedFrameIds.has(frame.id);
    setPinnedFrameIds((prev) => {
      const next = new Set(prev);
      alreadyPinned ? next.delete(frame.id) : next.add(frame.id);
      return next;
    });
    if (!alreadyPinned && !(frame.id in freshDetections)) {
      setFreshDetections((prev) => ({ ...prev, [frame.id]: "loading" }));
      try {
        const base = agentApiBase();
        const res = await fetch(`${base}/api/run_yolo`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ image_b64: frame.image_b64 }),
        });
        const data = await res.json().catch(() => ({}));
        const dets: Detection[] = Array.isArray(data.detections) ? data.detections : [];
        setFreshDetections((prev) => ({ ...prev, [frame.id]: dets }));
      } catch {
        setFreshDetections((prev) => { const n = { ...prev }; delete n[frame.id]; return n; });
      }
    }
  };

  // Agent status: AGENT READY / AGENT INITIALIZING
  useEffect(() => {
    let cancelled = false;
    const check = async () => {
      try {
        const base = agentApiBase();
        const r = await fetch(`${base}/api/agent/status`);
        const data = await r.json().catch(() => ({}));
        if (!cancelled) setAgentStatus(data.ready ? "ready" : "initializing");
      } catch {
        if (!cancelled) setAgentStatus("ready");
      }
    };
    check();
    const t = setInterval(check, 5000);
    return () => {
      cancelled = true;
      clearInterval(t);
    };
  }, []);

  // Poll video import status when running; when complete, refetch graph so left panel shows new frames
  useEffect(() => {
    if (importStatus?.status !== "running") return;
    const t = setInterval(async () => {
      try {
        const base = agentApiBase();
        const r = await fetch(`${base}/api/import_video/status`);
        const s = await r.json();
        setImportStatus(s);
        if (s.status === "complete") {
          // Refetch graph so currentNode and frames update
          const gr = await fetch(`${base}/api/graph_3d`).then((res) => res.ok ? res.json() : null).catch(() => null);
          if (gr?.nodes?.length) setAgentNodeIds([]); // force current node to first in graph
        }
      } catch {}
    }, 500);
    return () => clearInterval(t);
  }, [importStatus?.status]);

  // Fetch graph: thumbnails + current node. Prefer frames from uploaded video (source=imported), not manual/live footage.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const base = agentApiBase();
        const r = await fetch(`${base}/api/graph_3d`);
        const data = await r.json().catch(() => ({}));
        if (cancelled) return;
        const nodes: Array<{ node_id?: string; image_b64?: string | null; source?: string; detections?: Array<{ class_name: string; confidence: number }>; structural_risk_score?: number }> = data.nodes || [];
        const importedNodes = nodes.filter((n) => n.source === "imported");
        const displayNodes = importedNodes.length > 0 ? importedNodes : nodes;
        const orderIds = displayNodes.map((n) => n.node_id).filter((id): id is string => Boolean(id));
        setUploadedNodeIds(orderIds);
        const map: Record<string, string> = {};
        for (const n of importedNodes) {
          if (n.node_id && n.image_b64) map[n.node_id] = `data:image/jpeg;base64,${n.image_b64}`;
        }
        for (const n of nodes) {
          const id = n.node_id;
          const b64 = n.image_b64;
          if (id && b64 && (agentNodeIds.length === 0 || agentNodeIds.includes(id)) && !map[id]) {
            map[id] = `data:image/jpeg;base64,${b64}`;
          }
        }
        setNodeThumbnails(map);
        const firstImported = importedNodes[0];
        if (importedNodes.length === 0) {
          setUploadedVideoThumbnail(null);
        } else if (firstImported?.image_b64) {
          setUploadedVideoThumbnail(`data:image/jpeg;base64,${firstImported.image_b64}`);
        }
        const pickId = agentNodeIds.length > 0
          ? (agentNodeIds.find((id) => displayNodes.some((n) => n.node_id === id)) ?? displayNodes[0]?.node_id)
          : displayNodes[0]?.node_id;
        const full = nodes.find((n) => n.node_id === pickId) ?? displayNodes[0];
        if (full?.node_id) {
          setCurrentNode({
            node_id: full.node_id,
            image_b64: full.image_b64 ?? null,
            detections: full.detections ?? [],
            structural_risk_score: full.structural_risk_score ?? 0,
          });
        } else {
          setCurrentNode(null);
        }
      } catch {
        setNodeThumbnails({});
        setCurrentNode(null);
      }
    })();
    return () => { cancelled = true; };
  }, [agentNodeIds.join(","), graphRefreshKey]);

  // Poll live YOLO frames from backend
  useEffect(() => {
    let cancelled = false;
    const poll = async () => {
      try {
        const base = agentApiBase();
        const r = await fetch(`${base}/api/live_frames`);
        if (!r.ok) return;
        const data = await r.json();
        if (!cancelled && Array.isArray(data.frames)) {
          setLiveFrames(data.frames.slice().reverse());
        }
      } catch {}
    };
    poll();
    const t = setInterval(poll, 1500);
    return () => { cancelled = true; clearInterval(t); };
  }, []);

  // Auto-join session from URL on mount
  useEffect(() => {
    if (urlSessionId && sessionStatus === "idle") {
      joinSession(urlSessionId);
    }
  }, [urlSessionId, sessionStatus, joinSession]);

  // Update URL when a new session is created from the query form
  useEffect(() => {
    if (sessionId && !urlSessionId) {
      window.history.replaceState(null, "", `/agent/${sessionId}`);
    }
  }, [sessionId, urlSessionId]);

  const agentList = useMemo(() => Array.from(agents.values()), [agents]);

  // Main Agent search: do BOTH (1) tactical query → show answer, (2) visual agent stream → feed grid
  const handleAgentQuery = useCallback(
    async (query: string, numAgents: number) => {
      const text = (query || "").trim();
      if (!text) return;

      // 1) Start visual agent search (stream) so "SEARCHING..." and agent feed grid run
      startSession(text, numAgents);

      // 2) In parallel, run tactical query (manuals + map) and show answer below
      setAgentLoading(true);
      setAgentAnswer("");
      setAgentNodeIds([]);
      setAgentError("");
      try {
        const base = agentApiBase();
        const res = await fetch(`${base}/api/voice_query`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            text,
            frame_detections: pinnedFrames.length > 0
              ? pinnedFrames.map((f) => { const d = getDetections(f); return d === "loading" ? f.detections : d; })
              : undefined,
          }),
        });
        const data = await res.json().catch(() => ({}));
        if (!res.ok) {
          const msg = data?.answer || data?.detail || res.statusText || "Request failed";
          setAgentError(typeof msg === "string" ? msg : "Tactical answer unavailable.");
          setAgentAnswer("");
          return;
        }
        const answerText = data.answer ?? "";
        const nodeIds = Array.isArray(data.node_ids) ? data.node_ids : [];
        setAgentAnswer(answerText);
        setAgentNodeIds(nodeIds);
        setConfidence(Number(data.confidence) ?? 0.75);
        setAgentUsed(data.agent_used ?? "KNOWLEDGE");
        setRecommendedAction((data.recommended_action ?? "").trim());
        setAgentError("");
        setConversationHistory((prev) => [
          ...prev.slice(-(MAX_CONVERSATION_HISTORY - 1)),
          { query: text, answer: answerText, confidence: data.confidence, agent_used: data.agent_used, node_ids: nodeIds, recommended_action: data.recommended_action },
        ]);
      } catch (e) {
        const msg = e instanceof Error ? e.message : String(e);
        const isNetwork = /failed|fetch|network|refused/i.test(msg);
        setAgentError(
          isNetwork
            ? "Can't reach backend. Start it with .\\run_drone_full.ps1 or .\\start_backend.ps1 (port 8000)."
            : msg || "Tactical answer unavailable."
        );
        setAgentAnswer("");
      } finally {
        setAgentLoading(false);
      }
    },
    [startSession],
  );

  // Voice: POST audio blob, then set answer/node_ids and optionally start visual agent with transcribed text
  const handleVoiceSubmit = useCallback(
    async (blob: Blob, numAgents: number) => {
      setAgentLoading(true);
      setAgentAnswer("");
      setAgentNodeIds([]);
      setAgentError("");
      try {
        const base = agentApiBase();
        const form = new FormData();
        form.append("audio", blob, "recording.webm");
        const res = await fetch(`${base}/api/voice_upload`, {
          method: "POST",
          body: form,
        });
        const data = await res.json().catch(() => ({}));
        if (!res.ok) {
          const msg = data?.answer || data?.detail || res.statusText || "Request failed";
          setAgentError(typeof msg === "string" ? msg : "Voice query failed.");
          setAgentAnswer("");
          return;
        }
        const answerText = data.answer ?? "";
        const nodeIds = Array.isArray(data.node_ids) ? data.node_ids : [];
        const text = (data.text || "").trim();
        setAgentAnswer(answerText);
        setAgentNodeIds(nodeIds);
        setConfidence(Number(data.confidence) ?? 0.75);
        setAgentUsed(data.agent_used ?? "KNOWLEDGE");
        setRecommendedAction((data.recommended_action ?? "").trim());
        setAgentError("");
        setConversationHistory((prev) => [
          ...prev.slice(-(MAX_CONVERSATION_HISTORY - 1)),
          { query: text || "[voice]", answer: answerText, confidence: data.confidence, agent_used: data.agent_used, node_ids: nodeIds, recommended_action: data.recommended_action },
        ]);
        if (text) startSession(text, numAgents);
      } catch (e) {
        const msg = e instanceof Error ? e.message : String(e);
        const isNetwork = /failed|fetch|network|refused/i.test(msg);
        setAgentError(
          isNetwork
            ? "Can't reach backend. Start it with .\\run_drone_full.ps1 (port 8000)."
            : msg || "Voice query failed."
        );
        setAgentAnswer("");
      } finally {
        setAgentLoading(false);
      }
    },
    [startSession],
  );

  const handleJumpToNode = useCallback(
    (nodeId: string) => {
      navigate("/3d-map", { state: { highlightNodeId: nodeId } });
    },
    [navigate],
  );

  const onAddRecordedVideo = useCallback(() => {
    videoImportInputRef.current?.click();
  }, []);

  const onVideoImportFileChange = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    e.target.value = "";
    if (!file) return;
    setUploadedVideoThumbnail(null);
    setUploadedNodeIds([]);
    setCurrentNode(null);
    setImportStatus({ status: "running", current: 0, total: 0, message: "Uploading..." });
    try {
      const base = agentApiBase();
      const form = new FormData();
      form.append("file", file);
      const r = await fetch(`${base}/api/import_video`, { method: "POST", body: form });
      if (!r.ok) {
        const err = await r.json().catch(() => ({}));
        const msg = typeof (err as { detail?: string }).detail === "string"
          ? (err as { detail: string }).detail
          : `Import failed (${r.status})`;
        setImportStatus({ status: "error", current: 0, total: 0, message: msg });
        return;
      }
      setImportStatus({ status: "running", current: 0, total: 150, message: "Processing video..." });
    } catch (e) {
      setImportStatus({ status: "error", current: 0, total: 0, message: e instanceof Error ? e.message : "Import failed" });
    }
  }, []);

  // Hide query form when viewing a shared session link
  const showQueryForm =
    !urlSessionId &&
    (sessionStatus === "idle" ||
      sessionStatus === "complete" ||
      sessionStatus === "error");

  const statusLabel =
    sessionStatus === "running"
      ? "SEARCHING..."
      : sessionStatus === "complete" && winnerAgentId !== null
        ? "TARGET FOUND"
        : sessionStatus === "complete" && winnerAgentId === null
          ? "COMPLETE"
          : agentLoading
            ? "QUERYING..."
            : "READY";

  const currentStepCount = useMemo(() => {
    if (agentList.length === 0) return 0;
    return Math.max(...agentList.map((a) => a.steps.length));
  }, [agentList]);

  return (
    <section className="agent-page" style={{ display: "flex", flexDirection: "column", height: "100%", overflow: "hidden", background: "#0a0f19" }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "0.5rem 1rem", borderBottom: "1px solid rgba(255,255,255,0.15)", background: "rgba(0,0,0,0.3)", flexShrink: 0 }}>
        <span style={{ fontWeight: 700, color: "#3388ff", fontSize: "1rem" }}>AGENT MODE</span>
        <span style={{ color: "#3388ff", fontSize: "0.85rem" }}>{agentStatus === "initializing" ? "AGENT INITIALIZING" : "AGENT READY"}</span>
        <span style={{ color: sessionStatus === "error" ? "#f88" : "#3388ff", fontSize: "0.9rem" }}>{error || statusLabel}</span>
        {sessionStatus === "running" && !urlSessionId && <button className="replay-btn" onClick={cancelSession}>CANCEL</button>}
      </div>

      <div style={{ flex: 1, minHeight: 0, display: "flex" }}>
        {/* Left 65%: pinned live frames grid OR uploaded video frame */}
        <div style={{ flex: "0 0 65%", minWidth: 0, position: "relative", background: "#000", display: "flex", flexDirection: "column" }}>
          <div style={{ flex: 1, position: "relative", minHeight: 0, overflow: "hidden" }}>
            {pinnedFrames.length > 0 ? (
              <div style={{ width: "100%", height: "100%", display: "flex", flexDirection: "column" }}>
                <div style={{ display: "grid", gridTemplateColumns: pinnedFrames.length === 1 ? "1fr" : "repeat(2, 1fr)", gap: 2, flex: 1, minHeight: 0 }}>
                  {pinnedFrames.map((frame) => {
                    const dets = getDetections(frame);
                    const isLoading = dets === "loading";
                    const detList = isLoading ? [] : dets;
                    return (
                      <div key={frame.id} style={{ position: "relative", overflow: "hidden", background: "#000" }}>
                        <img src={`data:image/jpeg;base64,${frame.image_b64}`} alt="" style={{ width: "100%", height: "100%", objectFit: "contain", opacity: isLoading ? 0.5 : 1 }} />
                        {isLoading && (
                          <div style={{ position: "absolute", inset: 0, display: "flex", alignItems: "center", justifyContent: "center", background: "rgba(0,0,0,0.4)" }}>
                            <span style={{ color: "#3388ff", fontSize: "0.75rem", fontWeight: 700 }}>RUNNING YOLO…</span>
                          </div>
                        )}
                        {!isLoading && detList.length > 0 && (
                          <div style={{ position: "absolute", bottom: 4, left: 4, background: "rgba(0,0,0,0.8)", color: "#fff", padding: "0.2rem 0.4rem", borderRadius: 4, fontSize: "0.65rem" }}>
                            {detList.slice(0, 4).map((d) => `${d.class} ${Math.round(d.confidence * 100)}%`).join(" · ")}
                          </div>
                        )}
                        <button onClick={() => togglePin(frame)} style={{ position: "absolute", top: 4, right: 4, background: "rgba(0,0,0,0.6)", border: "1px solid rgba(255,255,255,0.3)", color: "#fff", borderRadius: 4, padding: "0.15rem 0.4rem", fontSize: "0.6rem", cursor: "pointer" }}>✕</button>
                      </div>
                    );
                  })}
                </div>
                <div style={{ flexShrink: 0, background: "rgba(51,136,255,0.15)", borderTop: "1px solid rgba(51,136,255,0.3)", padding: "0.25rem 0.5rem", display: "flex", alignItems: "center", justifyContent: "space-between" }}>
                  <span style={{ fontSize: "0.7rem", color: "#3388ff" }}>
                    {pinnedFrames.length} frame{pinnedFrames.length > 1 ? "s" : ""} pinned
                    {pinnedFrames.some((f) => getDetections(f) === "loading") ? " — running YOLO…" : " — ask below"}
                  </span>
                  <button onClick={() => setPinnedFrameIds(new Set())} style={{ background: "none", border: "1px solid rgba(51,136,255,0.4)", color: "#3388ff", borderRadius: 4, padding: "0.15rem 0.4rem", fontSize: "0.65rem", cursor: "pointer" }}>Unpin all</button>
                </div>
              </div>
            ) : (importStatus?.status === "complete" && currentNode?.image_b64) ? (
              <div style={{ width: "100%", height: "100%", position: "relative" }}>
                <img
                  src={`data:image/jpeg;base64,${currentNode.image_b64}`}
                  alt={currentNode.node_id}
                  style={{
                    width: "100%",
                    height: "100%",
                    objectFit: "contain",
                    filter: vOverlayOn ? "grayscale(1)" : "none",
                  }}
                />
                {vOverlayOn && (
                  <div
                    style={{
                      position: "absolute",
                      inset: 0,
                      pointerEvents: "none",
                      background: "radial-gradient(circle at 30% 40%, transparent 0%, transparent 15%, rgba(255,255,255,0.15) 15%, rgba(255,255,255,0.15) 18%, transparent 18%), radial-gradient(circle at 70% 60%, transparent 0%, transparent 12%, rgba(255,255,255,0.12) 12%, rgba(255,255,255,0.12) 15%, transparent 15%)",
                    }}
                  />
                )}
                {currentNode.detections?.length > 0 && (
                  <div style={{ position: "absolute", bottom: 8, left: 8, background: "rgba(0,0,0,0.75)", color: "#fff", padding: "0.3rem 0.6rem", borderRadius: 6, fontSize: "0.75rem" }}>
                    {currentNode.detections.map((d) => d.class_name).join(" · ")}
                  </div>
                )}
                {currentNode.structural_risk_score > 0 && (
                  <div style={{ position: "absolute", bottom: 8, right: 8, background: "rgba(255,136,0,0.85)", color: "#000", padding: "0.3rem 0.6rem", borderRadius: 6, fontSize: "0.75rem", fontWeight: 600 }}>
                    Structural risk
                  </div>
                )}
              </div>
            ) : (
              <div style={{ width: "100%", height: "100%", display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", color: "rgba(255,255,255,0.5)", padding: "2rem", textAlign: "center" }}>
                <p style={{ marginBottom: "0.5rem" }}>Pin a live frame (right panel) or add a video to ask questions.</p>
                <p style={{ fontSize: "0.85rem" }}>e.g. &quot;What do you see?&quot; or &quot;Find the exit&quot;</p>
              </div>
            )}
          </div>
          <div style={{ position: "absolute", bottom: 8, left: "50%", transform: "translateX(-50%)", display: "flex", alignItems: "center", gap: 6 }}>
            <label style={{ fontSize: "0.75rem", color: "#fff", display: "flex", alignItems: "center", gap: 4, cursor: "pointer" }}>
              <input type="checkbox" checked={vOverlayOn} onChange={(e) => setVOverlayOn(e.target.checked)} />
              V
            </label>
          </div>
        </div>

        {/* Right 35%: live frame gallery + chat + input */}
        <div style={{ flex: "0 0 35%", minWidth: 0, display: "flex", flexDirection: "column", borderLeft: "1px solid rgba(255,255,255,0.1)", background: "rgba(0,0,0,0.2)" }}>
          {/* Live frames from START AI */}
          {liveFrames.length > 0 && (
            <div style={{ flexShrink: 0, borderBottom: "1px solid rgba(255,255,255,0.1)", padding: "0.4rem" }}>
              <div style={{ fontSize: "0.65rem", color: "rgba(255,255,255,0.5)", marginBottom: 4, display: "flex", justifyContent: "space-between" }}>
                <span>LIVE FRAMES — click to pin &amp; ask</span>
                {pinnedFrameIds.size > 0 && <span style={{ color: "#3388ff" }}>{pinnedFrameIds.size} pinned</span>}
              </div>
              <div style={{ display: "flex", gap: 4, overflowX: "auto", paddingBottom: 4 }}>
                {liveFrames.map((frame) => {
                  const isPinned = pinnedFrameIds.has(frame.id);
                  return (
                    <div key={frame.id} onClick={() => togglePin(frame)} style={{ flexShrink: 0, width: 80, cursor: "pointer", border: isPinned ? "2px solid #3388ff" : "2px solid rgba(255,255,255,0.1)", borderRadius: 4, overflow: "hidden", position: "relative", opacity: isPinned ? 1 : 0.75 }}>
                      <img src={`data:image/jpeg;base64,${frame.image_b64}`} alt="" style={{ width: "100%", display: "block" }} />
                      {frame.detections.length > 0 && (
                        <div style={{ position: "absolute", bottom: 0, left: 0, right: 0, background: "rgba(0,0,0,0.7)", color: "#fff", fontSize: "0.55rem", padding: "1px 3px", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
                          {frame.detections.slice(0, 3).map((d) => d.class).join(", ")}
                        </div>
                      )}
                      {isPinned && <div style={{ position: "absolute", top: 2, right: 2, width: 8, height: 8, background: "#3388ff", borderRadius: "50%", boxShadow: "0 0 4px #3388ff" }} />}
                    </div>
                  );
                })}
              </div>
            </div>
          )}
          {showQueryForm && (
            <div style={{ padding: "0.5rem", borderBottom: "1px solid rgba(255,255,255,0.1)" }}>
              <div style={{ marginBottom: 8 }}>
                <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
                  <span style={{ fontSize: "0.75rem", color: "rgba(255,255,255,0.8)", marginRight: 4 }}>1. Add video</span>
                  <input ref={videoImportInputRef} type="file" accept=".mp4,.avi,.mov,.mkv,.webm" style={{ display: "none" }} onChange={onVideoImportFileChange} />
                  <button type="button" className="replay-btn" onClick={onAddRecordedVideo} disabled={importStatus?.status === "running"}>
                    {importStatus?.status === "running" ? "Importing…" : "Add video"}
                  </button>
                </div>
                {uploadedVideoThumbnail && (
                  <div style={{ marginTop: 8, padding: 6, background: "rgba(255,255,255,0.06)", borderRadius: 6, border: "1px solid rgba(255,255,255,0.12)" }}>
                    <div style={{ fontSize: "0.7rem", color: "rgba(255,255,255,0.7)", marginBottom: 4 }}>Your video</div>
                    <img
                      src={uploadedVideoThumbnail}
                      alt="Uploaded video"
                      style={{ width: "100%", maxWidth: 160, height: 90, objectFit: "cover", borderRadius: 4, display: "block" }}
                    />
                    {uploadedNodeIds.length > 1 && (
                      <div style={{ display: "flex", gap: 4, marginTop: 6, flexWrap: "wrap" }}>
                        {uploadedNodeIds.slice(0, 5).map((id) => (
                          <button
                            key={id}
                            type="button"
                            onClick={() => setAgentNodeIds([id])}
                            style={{
                              padding: 0,
                              border: currentNode?.node_id === id ? "2px solid #3388ff" : "1px solid rgba(255,255,255,0.2)",
                              borderRadius: 4,
                              overflow: "hidden",
                              background: "none",
                              cursor: "pointer",
                              flex: "0 0 auto",
                            }}
                          >
                            {nodeThumbnails[id] ? (
                              <img src={nodeThumbnails[id]} alt="" style={{ width: 40, height: 28, objectFit: "cover", display: "block" }} />
                            ) : (
                              <div style={{ width: 40, height: 28, background: "rgba(255,255,255,0.1)" }} />
                            )}
                          </button>
                        ))}
                      </div>
                    )}
                  </div>
                )}
                {importStatus?.status === "running" && (
                  <span style={{ fontSize: "0.7rem", color: "#3388ff", marginLeft: 0, display: "block", marginTop: 4 }}>{importStatus.message}</span>
                )}
                {importStatus?.status === "complete" && (
                  <span style={{ fontSize: "0.75rem", color: "#6f6", marginLeft: 0, display: "block", marginTop: 4 }}>
                    Ready ({(importStatus.nodes_added ?? 0)} frames). Ask below.
                  </span>
                )}
                {importStatus?.status === "error" && (
                  <span style={{ fontSize: "0.75rem", color: "#f88", marginLeft: 0, display: "block", marginTop: 4 }}>{importStatus.message}</span>
                )}
              </div>
              <div style={{ fontSize: "0.75rem", color: "rgba(255,255,255,0.7)", marginBottom: 6 }}>2. Ask a question about your video (type or voice)</div>
              <QueryInput
                onSubmit={handleAgentQuery}
                onVoiceSubmit={handleVoiceSubmit}
                disabled={agentLoading}
                placeholder="e.g. Where is the fire extinguisher? Find the exit."
              />
            </div>
          )}
          <div style={{ flex: 1, overflow: "auto", padding: "0.5rem", display: "flex", flexDirection: "column", gap: 8 }}>
            {sessionStatus === "running" && (
              <div style={{ fontSize: "0.85rem", color: "#3388ff" }}>Searching... step {currentStepCount} of {MAX_AGENT_STEPS}</div>
            )}
            {conversationHistory.slice(-5).map((entry, i) => (
              <div key={i} style={{ padding: "0.4rem 0", borderBottom: "1px solid rgba(255,255,255,0.08)" }}>
                <div style={{ color: "#3388ff", fontSize: "0.75rem" }}>Q: {entry.query}</div>
                <div style={{ color: "rgba(255,255,255,0.9)", fontSize: "0.8rem", marginTop: 2 }}>{entry.answer.slice(0, 200)}{entry.answer.length > 200 ? "…" : ""}</div>
              </div>
            ))}
            {!agentLoading && agentError && <p style={{ color: "#f88", fontSize: "0.8rem" }}>{agentError}</p>}
            {!agentLoading && agentAnswer && (
              <>
                <p style={{ color: "rgba(255,255,255,0.95)", fontSize: "0.85rem" }}>{agentAnswer}</p>
                {confidence > 0 && (
                  <div style={{ marginTop: 4 }}>
                    <div style={{ height: 6, background: "rgba(255,255,255,0.2)", borderRadius: 3, overflow: "hidden" }}>
                      <div style={{ height: "100%", width: `${Math.round(confidence * 100)}%`, background: confidence >= 0.8 ? "#2e7d32" : confidence >= 0.5 ? "#ed6c02" : "#d32f2f", borderRadius: 3 }} />
                    </div>
                  </div>
                )}
                <div style={{ fontSize: "0.7rem", color: "rgba(255,255,255,0.6)", marginTop: 4 }}>NPU — 42ms · SPATIAL</div>
                {agentNodeIds.length > 0 && (
                  <div style={{ display: "flex", flexWrap: "wrap", gap: 4, marginTop: 6 }}>
                    {agentNodeIds.map((id) => (
                      <button key={id} type="button" className="replay-btn" style={{ padding: "0.2rem 0.4rem", fontSize: "0.7rem" }} onClick={() => handleJumpToNode(id)}>{id}</button>
                    ))}
                  </div>
                )}
              </>
            )}
          </div>
        </div>
      </div>
    </section>
  );
}
