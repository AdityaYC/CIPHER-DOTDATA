import { useCallback, useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import * as THREE from "three";
import { API_BASE_URL } from "../config";

type Graph3DStats = {
  node_count: number;
  point_count: number;
  area_m2: number;
  survivors: number;
  hazards: number;
  exits: number;
  structural: number;
};

type Graph3DNode = {
  node_id: string;
  timestamp: number;
  gps_lat: number;
  gps_lon: number;
  altitude_m: number;
  yaw_deg: number;
  detections: Array<{ class_name: string; confidence: number; category: string }>;
  image_b64: string | null;
  pose: [number, number, number] | null;
  structural_risk_score: number;
  source?: "live" | "imported";
};

type Graph3DResponse = {
  pointcloud: Array<{ x: number; y: number; z: number; r: number; g: number; b: number }>;
  path: Array<{ x: number; y: number; z: number }>;
  nodes: Graph3DNode[];
  stats: Graph3DStats;
};

type ImportStatus = { status: string; current: number; total: number; message: string; nodes_added?: number };

const API = typeof window !== "undefined" && window.location.port !== "80" && window.location.port !== "443"
  ? (import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000")
  : (typeof window !== "undefined" ? window.location.origin : "http://localhost:8000");

const CROSS_FADE_STEPS = 8;
const CROSS_FADE_MS = 200;

export function Map3DPage() {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [data, setData] = useState<Graph3DResponse | null>(null);
  const [exporting, setExporting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [currentFrameNode, setCurrentFrameNode] = useState<Graph3DNode | null>(null);
  const [transitionFrom, setTransitionFrom] = useState<Graph3DNode | null>(null);
  const [transitionStep, setTransitionStep] = useState(0);
  const [flashBorder, setFlashBorder] = useState(false);
  const [importStatus, setImportStatus] = useState<ImportStatus | null>(null);
  const [frameFullscreen, setFrameFullscreen] = useState(false);
  const [historyStack, setHistoryStack] = useState<string[]>([]); // BACK = pop and go here
  const fileInputRef = useRef<HTMLInputElement>(null);
  const map2dCanvasRef = useRef<HTMLCanvasElement>(null);
  const sceneRef = useRef<{
    scene: THREE.Scene;
    camera: THREE.PerspectiveCamera;
    renderer: THREE.WebGLRenderer;
    points: THREE.Points | null;
    pathLine: THREE.Line | null;
    markerMeshes: THREE.Group;
    currentDroneMesh: THREE.Mesh | null;
    animationId: number;
  } | null>(null);

  const base = API_BASE_URL || API;

  const fetchGraph3d = useCallback(async () => {
    try {
      const r = await fetch(`${base}/api/graph_3d`);
      if (!r.ok) throw new Error(r.statusText);
      const json: Graph3DResponse = await r.json();
      setData(json);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load 3D graph");
    }
  }, [base]);

  useEffect(() => {
    fetchGraph3d();
    const t = setInterval(fetchGraph3d, 2000);
    return () => clearInterval(t);
  }, [fetchGraph3d]);

  useEffect(() => {
    if (data?.nodes?.length && !currentFrameNode) {
      const first = data.nodes.find((n: Graph3DNode) => n.source === "imported") ?? data.nodes[0];
      setCurrentFrameNode(first);
    }
    if (!data?.nodes?.length) {
      setTransitionFrom(null);
      setTransitionStep(0);
      setFrameFullscreen(false);
      setCurrentFrameNode(null);
      setHistoryStack([]);
    }
  }, [data?.nodes?.length, currentFrameNode]);

  // Cross-fade animation
  useEffect(() => {
    if (transitionStep <= 0 || transitionStep >= CROSS_FADE_STEPS) return;
    const interval = setInterval(() => {
      setTransitionStep((s) => (s >= CROSS_FADE_STEPS ? CROSS_FADE_STEPS : s + 1));
    }, CROSS_FADE_MS / CROSS_FADE_STEPS);
    return () => clearInterval(interval);
  }, [transitionStep]);

  useEffect(() => {
    if (transitionStep >= CROSS_FADE_STEPS && transitionFrom) {
      setTransitionFrom(null);
      setTransitionStep(0);
    }
  }, [transitionStep, transitionFrom]);

  const navigate = useCallback(async (direction: "forward" | "back" | "left" | "right" | "next" | "prev") => {
    if (!currentFrameNode || !data?.nodes) return;
    if (direction === "back") {
      if (historyStack.length === 0) {
        setFlashBorder(true);
        setTimeout(() => setFlashBorder(false), 300);
        return;
      }
      const prevId = historyStack[historyStack.length - 1];
      setHistoryStack((s) => s.slice(0, -1));
      const prevNode = data.nodes.find((n: Graph3DNode) => n.node_id === prevId);
      if (prevNode) {
        setTransitionFrom(currentFrameNode);
        setCurrentFrameNode(prevNode);
        setTransitionStep(1);
      }
      return;
    }
    try {
      const r = await fetch(`${base}/api/graph_3d/neighbor?node_id=${encodeURIComponent(currentFrameNode.node_id)}&direction=${direction}`);
      if (!r.ok) {
        setFlashBorder(true);
        setTimeout(() => setFlashBorder(false), 300);
        return;
      }
      const { node } = await r.json() as { node: Graph3DNode };
      setHistoryStack((s) => [...s, currentFrameNode.node_id]);
      setTransitionFrom(currentFrameNode);
      setCurrentFrameNode(node);
      setTransitionStep(1);
    } catch {
      setFlashBorder(true);
      setTimeout(() => setFlashBorder(false), 300);
    }
  }, [base, currentFrameNode, data?.nodes, historyStack.length]);

  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;
      if (e.key === "Escape") {
        if (frameFullscreen) {
          e.preventDefault();
          setFrameFullscreen(false);
          return;
        }
      }
      if (!currentFrameNode) return;
      if (e.key === "ArrowRight" || e.key === "ArrowUp") { e.preventDefault(); navigate("next"); }
      else if (e.key === "ArrowLeft" || e.key === "ArrowDown") { e.preventDefault(); navigate("prev"); }
    };
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, [currentFrameNode, frameFullscreen, navigate]);

  // Import status poll — when complete, refetch graph and set current frame to first frame (start of video)
  useEffect(() => {
    if (importStatus?.status !== "running") return;
    const t = setInterval(async () => {
      try {
        const r = await fetch(`${base}/api/import_video/status`);
        const s: ImportStatus = await r.json();
        setImportStatus(s);
        if (s.status === "complete" || s.status === "error") {
          const graph = await fetch(`${base}/api/graph_3d`).then((res) => res.ok ? res.json() : null);
          if (graph?.nodes?.length && s.status === "complete") {
            setData(graph);
            const firstImported = graph.nodes.find((n: Graph3DNode) => n.source === "imported");
            const first = firstImported ?? graph.nodes[0];
            setCurrentFrameNode(first);
            setFrameFullscreen(false);
          } else {
            fetchGraph3d();
          }
        }
      } catch {}
    }, 500);
    return () => clearInterval(t);
  }, [base, importStatus?.status, fetchGraph3d]);

  const onImportVideo = useCallback(() => {
    fileInputRef.current?.click();
  }, []);

  const onImportFileChange = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    e.target.value = "";
    if (!file) return;
    setImportStatus({ status: "running", current: 0, total: 0, message: "Uploading..." });
    try {
      const form = new FormData();
      form.append("file", file);
      const apiBase = base || (typeof window !== "undefined" ? window.location.origin : "");
      const url = `${apiBase.replace(/\/+$/, "")}/api/import_video`;
      const r = await fetch(url, { method: "POST", body: form });
      if (!r.ok) {
        const err = await r.json().catch(() => ({}));
        const msg = typeof (err as { detail?: string }).detail === "string"
          ? (err as { detail: string }).detail
          : (err as { detail?: string }).detail
            ? JSON.stringify((err as { detail: unknown }).detail)
            : `Import failed (${r.status})`;
        setImportStatus({ status: "error", current: 0, total: 0, message: msg });
        return;
      }
      setImportStatus({ status: "running", current: 0, total: 150, message: "Processing..." });
    } catch (e) {
      setImportStatus({ status: "error", current: 0, total: 0, message: e instanceof Error ? e.message : "Import failed" });
    }
  }, [base]);

  useEffect(() => {
    const container = containerRef.current;
    const canvas = canvasRef.current;
    if (!container || !canvas || !data) return;

    const width = container.clientWidth;
    const height = container.clientHeight;
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0a0f19);

    const camera = new THREE.PerspectiveCamera(60, width / height, 0.1, 2000);
    camera.position.set(8, 8, 8);
    camera.lookAt(0, 0, 0);

    const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

    const markerMeshes = new THREE.Group();
    if (data.pointcloud.length > 0) {
      const positions = new Float32Array(data.pointcloud.length * 3);
      const colors = new Float32Array(data.pointcloud.length * 3);
      data.pointcloud.forEach((p, i) => {
        positions[i * 3] = p.x;
        positions[i * 3 + 1] = p.y;
        positions[i * 3 + 2] = p.z;
        colors[i * 3] = p.r;
        colors[i * 3 + 1] = p.g;
        colors[i * 3 + 2] = p.b;
      });
      const geo = new THREE.BufferGeometry();
      geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
      geo.setAttribute("color", new THREE.BufferAttribute(colors, 3));
      const mat = new THREE.PointsMaterial({ size: 0.12, vertexColors: true, sizeAttenuation: true });
      const points = new THREE.Points(geo, mat);
      scene.add(points);
      sceneRef.current = { scene, camera, renderer, points, pathLine: null, markerMeshes, currentDroneMesh: null, animationId: 0 };
    } else {
      sceneRef.current = { scene, camera, renderer, points: null, pathLine: null, markerMeshes, currentDroneMesh: null, animationId: 0 };
    }

    if (data.path.length >= 2) {
      const pathPoints = data.path.map((p) => new THREE.Vector3(p.x, p.y, p.z));
      const pathGeo = new THREE.BufferGeometry().setFromPoints(pathPoints);
      const pathLine = new THREE.Line(pathGeo, new THREE.LineBasicMaterial({ color: 0xffdd00, linewidth: 2 }));
      scene.add(pathLine);
      if (sceneRef.current) sceneRef.current.pathLine = pathLine;
    }

    const typeColors: Record<string, number> = {
      survivor: 0x00ff66,
      hazard: 0xff3333,
      exit: 0x3388ff,
      structural: 0xff8800,
      clear: 0x888888,
      unknown: 0x888888,
    };
    const ordered = data.nodes;
    const lastIndex = ordered.length - 1;
    const currentId = currentFrameNode ? currentFrameNode.node_id : (ordered[lastIndex]?.node_id ?? null);
    ordered.forEach((node) => {
      const pose = node.pose;
      if (!pose) return;
      const [x, y, z] = pose;
      const isCurrent = node.node_id === currentId;
      const geom = new THREE.SphereGeometry(isCurrent ? 0.35 : 0.18, 12, 12);
      const col = node.source === "imported" ? 0xffffff : (isCurrent ? 0xffdd00 : (typeColors[node.detections?.[0]?.category ?? "clear"] ?? 0x888888));
      const mat = new THREE.MeshBasicMaterial({ color: col });
      const mesh = new THREE.Mesh(geom, mat);
      mesh.position.set(x, y, z);
      (mesh as THREE.Mesh & { nodeId?: string; node?: Graph3DNode }).nodeId = node.node_id;
      (mesh as THREE.Mesh & { nodeId?: string; node?: Graph3DNode }).node = node;
      markerMeshes.add(mesh);
      if (isCurrent && sceneRef.current) sceneRef.current.currentDroneMesh = mesh;
    });
    scene.add(markerMeshes);

    const raycaster = new THREE.Raycaster();
    const mouse = new THREE.Vector2();
    const onPointerClick = (e: MouseEvent) => {
      const rect = canvas.getBoundingClientRect();
      mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
      mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
      raycaster.setFromCamera(mouse, camera);
      const meshes: THREE.Object3D[] = [];
      markerMeshes.traverse((c: THREE.Object3D) => { if (c instanceof THREE.Mesh) meshes.push(c); });
      const hit = raycaster.intersectObjects(meshes);
      if (hit.length > 0) {
        const obj = hit[0].object as THREE.Mesh & { node?: Graph3DNode };
        if (obj.node) setCurrentFrameNode(obj.node);
      }
    };
    canvas.addEventListener("click", onPointerClick);

    let animationId = 0;
    const tick = () => {
      animationId = requestAnimationFrame(tick);
      if (sceneRef.current?.currentDroneMesh?.material instanceof THREE.MeshBasicMaterial) {
        const m = sceneRef.current.currentDroneMesh.material;
        m.color.setHSL(0.14, 1, 0.5 + 0.15 * Math.sin(performance.now() * 0.003));
      }
      renderer.render(scene, camera);
    };
    tick();
    if (sceneRef.current) sceneRef.current.animationId = animationId;

    const onResize = () => {
      if (!containerRef.current || !sceneRef.current) return;
      const w = containerRef.current.clientWidth;
      const h = containerRef.current.clientHeight;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    };
    window.addEventListener("resize", onResize);

    return () => {
      window.removeEventListener("resize", onResize);
      canvas.removeEventListener("click", onPointerClick);
      cancelAnimationFrame(animationId);
      renderer.dispose();
      if (sceneRef.current) {
        sceneRef.current.points?.geometry?.dispose();
        (sceneRef.current.points?.material as THREE.Material)?.dispose();
        sceneRef.current.pathLine?.geometry?.dispose();
        (sceneRef.current.pathLine?.material as THREE.Material)?.dispose();
      }
      sceneRef.current = null;
    };
  }, [data, currentFrameNode]);

  const onExportVr = async () => {
    setExporting(true);
    try {
      const r = await fetch(`${base}/api/export_vr`, { method: "POST" });
      if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
      const { url } = await r.json();
      window.open(`${base}${url}`, "_blank");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Export failed");
    } finally {
      setExporting(false);
    }
  };

  const stats = data?.stats;
  const hasFramePanel = data?.nodes?.length && currentFrameNode;
  const MAP_NODE_COLORS: Record<string, string> = { survivor: "#00ff66", hazard: "#ff3333", structural: "#ff8800", exit: "#3388ff", clear: "#888", unknown: "#888" };

  // 2D overhead map: nodes as dots, path, current pulsing; click dot → setCurrentFrameNode
  useEffect(() => {
    const canvas = map2dCanvasRef.current;
    if (!canvas || !data?.nodes?.length) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const W = canvas.width = 280;
    const H = canvas.height = 160;
    ctx.fillStyle = "#0a0f19";
    ctx.fillRect(0, 0, W, H);
    const nodes = data.nodes;
    const path = data.path || [];
    const xs = path.length ? path.map((p) => p.x) : nodes.map((n: Graph3DNode) => n.pose?.[0] ?? 0).filter((v: number) => typeof v === "number");
    const ys = path.length ? path.map((p) => p.y) : nodes.map((n: Graph3DNode) => n.pose?.[1] ?? 0).filter((v: number) => typeof v === "number");
    if (xs.length === 0 || ys.length === 0) return;
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);
    const pad = 16;
    const rangeX = maxX - minX || 1;
    const rangeY = maxY - minY || 1;
    const scale = Math.min((W - pad * 2) / rangeX, (H - pad * 2) / rangeY);
    const cx = (minX + maxX) / 2;
    const cy = (minY + maxY) / 2;
    const toScreen = (x: number, y: number) => ({ sx: W / 2 + (x - cx) * scale, sy: H / 2 - (y - cy) * scale });
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
    const currentId = currentFrameNode?.node_id ?? null;
    nodes.forEach((n: Graph3DNode) => {
      const pos = n.pose;
      if (!pos) return;
      const { sx, sy } = toScreen(pos[0], pos[1]);
      const cat = n.detections?.[0]?.category ?? "unknown";
      const color = MAP_NODE_COLORS[cat] ?? MAP_NODE_COLORS.unknown;
      const isCurrent = n.node_id === currentId;
      ctx.fillStyle = isCurrent ? "#fff" : color;
      ctx.beginPath();
      ctx.arc(sx, sy, isCurrent ? 5 : 3, 0, Math.PI * 2);
      ctx.fill();
      if (isCurrent) { ctx.strokeStyle = "#fff"; ctx.lineWidth = 2; ctx.stroke(); }
    });
  }, [data, currentFrameNode]);

  const onMap2dClick = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!data?.nodes?.length || !currentFrameNode) return;
    const canvas = map2dCanvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * canvas.width;
    const y = ((e.clientY - rect.top) / rect.height) * canvas.height;
    const nodes = data.nodes;
    const path = data.path || [];
    const xs = path.length ? path.map((p) => p.x) : nodes.map((n: Graph3DNode) => n.pose?.[0] ?? 0);
    const ys = path.length ? path.map((p) => p.y) : nodes.map((n: Graph3DNode) => n.pose?.[1] ?? 0);
    const minX = Math.min(...xs, 0); const maxX = Math.max(...xs, 0);
    const minY = Math.min(...ys, 0); const maxY = Math.max(...ys, 0);
    const pad = 16; const W = 280; const H = 160;
    const rangeX = maxX - minX || 1; const rangeY = maxY - minY || 1;
    const scale = Math.min((W - pad * 2) / rangeX, (H - pad * 2) / rangeY);
    const cx = (minX + maxX) / 2; const cy = (minY + maxY) / 2;
    const toWorld = (sx: number, sy: number) => ({ x: (sx - W / 2) / scale + cx, y: -(sy - H / 2) / scale + cy });
    const { x: wx, y: wy } = toWorld(x, y);
    let best: Graph3DNode | null = null;
    let bestD = 1e9;
    nodes.forEach((n: Graph3DNode) => {
      const p = n.pose;
      if (!p) return;
      const d = (p[0] - wx) ** 2 + (p[1] - wy) ** 2;
      if (d < bestD) { bestD = d; best = n; }
    });
    if (best && bestD < 2) {
      setTransitionFrom(currentFrameNode);
      setCurrentFrameNode(best);
      setTransitionStep(1);
    }
  }, [data, currentFrameNode]);

  const interactiveViewContent = (
    <>
      <div style={{ position: "absolute", top: 8, left: 8, zIndex: 2, fontFamily: "monospace", fontSize: frameFullscreen ? "1rem" : "0.85rem", color: "#fff" }}>
        X: {(currentFrameNode?.pose?.[0] ?? 0).toFixed(2)} | Y: {(currentFrameNode?.pose?.[1] ?? 0).toFixed(2)} | Z: {(currentFrameNode?.pose?.[2] ?? 0).toFixed(2)} | YAW: {(currentFrameNode?.yaw_deg ?? 0).toFixed(0)}°
      </div>
      {frameFullscreen && (
        <button type="button" className="replay-btn" onClick={() => setFrameFullscreen(false)} style={{ position: "absolute", top: 8, right: 8, zIndex: 3 }}>Exit full screen</button>
      )}
      <div style={{ flex: 1, position: "relative", background: "#000", minHeight: 0 }}>
        {transitionFrom && transitionStep > 0 && transitionStep < CROSS_FADE_STEPS && (
          <div style={{ position: "absolute", inset: 0, opacity: 1 - transitionStep / CROSS_FADE_STEPS, transition: "opacity 0.025s linear" }}>
            {transitionFrom.image_b64 ? <img src={`data:image/jpeg;base64,${transitionFrom.image_b64}`} alt="" style={{ width: "100%", height: "100%", objectFit: "contain" }} /> : <div style={{ width: "100%", height: "100%", display: "flex", alignItems: "center", justifyContent: "center", color: "#666" }}>{transitionFrom.node_id}</div>}
          </div>
        )}
        <div style={{ position: "absolute", inset: 0, opacity: transitionFrom && transitionStep > 0 ? transitionStep / CROSS_FADE_STEPS : 1 }}>
          {currentFrameNode?.image_b64 ? <img src={`data:image/jpeg;base64,${currentFrameNode.image_b64}`} alt="" style={{ width: "100%", height: "100%", objectFit: "contain" }} /> : <div style={{ width: "100%", height: "100%", display: "flex", alignItems: "center", justifyContent: "center", color: "#666" }}>{currentFrameNode ? currentFrameNode.node_id : "No node"}</div>}
        </div>
      </div>
      <div style={{ position: "absolute", bottom: 8, left: 8, zIndex: 2, fontSize: "0.8rem", color: "rgba(255,255,255,0.9)" }}>INTERACTIVE WORLDS</div>
      <div style={{ position: "absolute", bottom: 8, left: "50%", transform: "translateX(-50%)", display: "flex", gap: 6, zIndex: 2 }}>
        <button type="button" className="replay-btn" onClick={() => navigate("forward")} style={{ padding: "0.35rem 0.6rem" }}>FORWARD</button>
        <button type="button" className="replay-btn" onClick={() => navigate("left")} style={{ padding: "0.35rem 0.6rem" }}>LEFT</button>
        <button type="button" className="replay-btn" onClick={() => navigate("back")} style={{ padding: "0.35rem 0.6rem" }}>BACK</button>
        <button type="button" className="replay-btn" onClick={() => navigate("right")} style={{ padding: "0.35rem 0.6rem" }}>RIGHT</button>
      </div>
    </>
  );

  return (
    <section className="map3d-page" style={{ display: "flex", flexDirection: "column", height: "100%", overflow: "hidden", background: "#0a0f19" }}>
      {frameFullscreen && (
        <div style={{ position: "fixed", inset: 0, zIndex: 9999, background: "#000", display: "flex", flexDirection: "column" }}>
          <div style={{ flex: 1, position: "relative", display: "flex", flexDirection: "column", minHeight: 0 }}>{interactiveViewContent}</div>
        </div>
      )}

      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "0.5rem 1rem", background: "rgba(0,0,0,0.4)", borderBottom: "1px solid rgba(255,255,255,0.1)", flexWrap: "wrap", gap: "0.5rem" }}>
        <div style={{ display: "flex", gap: "1.5rem", fontSize: "0.8rem", color: "rgba(255,255,255,0.9)", alignItems: "center", flexWrap: "wrap" }}>
          <span>Nodes: <strong>{stats?.node_count ?? 0}</strong></span>
          <span>Area: <strong>{stats?.area_m2 ?? 0}</strong> m²</span>
          <span style={{ color: "#00ff66" }}>Survivors: {stats?.survivors ?? 0}</span>
          <span style={{ color: "#ff3333" }}>Hazards: {stats?.hazards ?? 0}</span>
          <span style={{ color: "#3388ff" }}>Exits: {stats?.exits ?? 0}</span>
          <span style={{ color: "#ff8800" }}>Structural: {stats?.structural ?? 0}</span>
        </div>
        <div style={{ display: "flex", gap: "0.5rem", alignItems: "center", marginLeft: "auto" }}>
          <input ref={fileInputRef} type="file" accept=".mp4,.avi,.mov,.mkv,.webm" style={{ display: "none" }} onChange={onImportFileChange} />
          <button type="button" className="replay-btn" onClick={fetchGraph3d} title="Refresh map">Refresh</button>
          <button type="button" className="replay-btn" onClick={onImportVideo} disabled={importStatus?.status === "running"}>IMPORT VIDEO</button>
          <Link to="/replay" className="replay-btn" style={{ textDecoration: "none", color: "inherit" }}>Replay</Link>
          <Link to="/agent" className="replay-btn" style={{ textDecoration: "none", color: "inherit" }}>Agent</Link>
          <button type="button" className="replay-btn" onClick={onExportVr} disabled={exporting}>{exporting ? "Exporting…" : "EXPORT VR"}</button>
        </div>
      </div>

      {importStatus?.status === "error" && <div style={{ padding: "0.4rem 1rem", background: "rgba(80,0,0,0.4)", fontSize: "0.85rem", color: "#f88" }}>{importStatus.message}</div>}
      {error && <div style={{ padding: "0.5rem 1rem", background: "#4a0000", color: "#ff8888", fontSize: "0.85rem" }}>{error}</div>}

      <div style={{ flex: 1, display: "flex", minHeight: 0 }}>
        {/* Left 65%: current node frame + X Y Z YAW, INTERACTIVE WORLDS, FORWARD/LEFT/BACK/RIGHT, refresh */}
        <div style={{ flex: "0 0 65%", minWidth: 0, display: "flex", flexDirection: "column", borderRight: "1px solid rgba(255,255,255,0.1)", position: "relative", border: flashBorder ? "3px solid #ff0000" : "1px solid rgba(255,255,255,0.1)", transition: "border 0.1s ease" }}>
          {hasFramePanel ? (
            <>
              <button type="button" className="replay-btn" onClick={() => setFrameFullscreen(true)} style={{ position: "absolute", top: 8, right: 8, zIndex: 3 }}>Full screen</button>
              {interactiveViewContent}
            </>
          ) : (
            <div style={{ flex: 1, display: "flex", alignItems: "center", justifyContent: "center", color: "rgba(255,255,255,0.5)" }}>Import video to see frames. Click spheres or dots on the right to navigate.</div>
          )}
        </div>

        {/* Right 35%: top 60% 3D point cloud, bottom 40% 2D map + node count/area/type */}
        <div style={{ flex: "0 0 35%", minWidth: 0, display: "flex", flexDirection: "column", background: "#0a0f19" }}>
          <div ref={containerRef} style={{ flex: "0 0 60%", position: "relative", minHeight: 0 }}>
            <canvas ref={canvasRef} style={{ display: "block", width: "100%", height: "100%" }} />
          </div>
          <div style={{ flex: "0 0 40%", display: "flex", flexDirection: "column", padding: "0.5rem", borderTop: "1px solid rgba(255,255,255,0.1)" }}>
            <canvas ref={map2dCanvasRef} width={280} height={160} style={{ width: "100%", maxHeight: 140, objectFit: "contain", cursor: "pointer", background: "#0a0f19" }} onClick={onMap2dClick} />
            <div style={{ fontSize: "0.75rem", color: "rgba(255,255,255,0.8)", marginTop: 4 }}>
              Nodes: {stats?.node_count ?? 0} · Area: {(stats?.area_m2 ?? 0).toFixed(1)} m²
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
