import { useCallback, useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { API_BASE_URL } from "../config";

type ReplayRow = {
  index: number;
  timestampLabel: string;
  timestampNs?: string;
  timestampSeconds: number;
  x: number;
  y: number;
  z: number;
  pitchDeg?: number;
  yawDeg?: number;
  rollDeg?: number;
};

type BoundRow = ReplayRow & {
  imageFile?: File;
  imageUrl?: string;
};

type ImagesManifest = {
  images?: string[];
};

const AUTO_TRAJECTORY_URL = "/replay/trajectory.csv";
const AUTO_MANIFEST_URL = "/replay/images_manifest.json";
const AUTO_IMAGES_BASE = "/replay/images_extracted";

const replayApiBase = () =>
  typeof window !== "undefined"
    ? API_BASE_URL || window.location.origin
    : "http://localhost:8000";

function normalizeKey(name: string): string {
  return name.trim().toLowerCase().replace(/[^a-z0-9#]+/g, "");
}

function parseNumber(value: string): number {
  const n = Number(value);
  return Number.isFinite(n) ? n : 0;
}

function quatToEulerDeg(
  qw: number,
  qx: number,
  qy: number,
  qz: number,
): { pitchDeg: number; yawDeg: number; rollDeg: number } {
  const norm = Math.hypot(qw, qx, qy, qz);
  if (norm <= 0) {
    return { pitchDeg: 0, yawDeg: 0, rollDeg: 0 };
  }
  const w = qw / norm;
  const x = qx / norm;
  const y = qy / norm;
  const z = qz / norm;

  const sinrCosp = 2 * (w * x + y * z);
  const cosrCosp = 1 - 2 * (x * x + y * y);
  const roll = Math.atan2(sinrCosp, cosrCosp);

  const sinp = 2 * (w * y - z * x);
  const pitch = Math.abs(sinp) >= 1 ? Math.sign(sinp) * (Math.PI / 2) : Math.asin(sinp);

  const sinyCosp = 2 * (w * z + x * y);
  const cosyCosp = 1 - 2 * (y * y + z * z);
  const yaw = Math.atan2(sinyCosp, cosyCosp);

  return {
    pitchDeg: (pitch * 180) / Math.PI,
    yawDeg: (yaw * 180) / Math.PI,
    rollDeg: (roll * 180) / Math.PI,
  };
}

function parseTrajectoryCsv(text: string): ReplayRow[] {
  const lines = text
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => line.length > 0);
  if (lines.length < 2) return [];

  const header = lines[0].split(",").map((v) => v.trim());
  const keys = header.map(normalizeKey);

  const idx = (candidates: string[]): number =>
    keys.findIndex((key) => candidates.some((candidate) => key.startsWith(candidate)));

  const iTimestampNs = idx(["#timestampns", "timestampns"]);
  const iTimestampS = idx(["timestamps", "timestamp"]);
  const iX = idx(["prsrxm", "xm", "x"]);
  const iY = idx(["prsrym", "ym", "y"]);
  const iZ = idx(["prsrzm", "zm", "z"]);
  const iPitch = idx(["pitchdeg"]);
  const iYaw = idx(["yawdeg"]);
  const iRoll = idx(["rolldeg"]);
  const iQw = idx(["qrsw"]);
  const iQx = idx(["qrsx"]);
  const iQy = idx(["qrsy"]);
  const iQz = idx(["qrsz"]);

  if (iX < 0 || iY < 0 || iZ < 0) {
    return [];
  }

  const out: ReplayRow[] = [];
  let firstTimestampNs: number | null = null;
  for (let lineIndex = 1; lineIndex < lines.length; lineIndex += 1) {
    const cols = lines[lineIndex].split(",").map((v) => v.trim());
    if (cols.length < 4) continue;

    const x = parseNumber(cols[iX] ?? "0");
    const y = parseNumber(cols[iY] ?? "0");
    const z = parseNumber(cols[iZ] ?? "0");

    let timestampNsRaw = iTimestampNs >= 0 ? cols[iTimestampNs] : undefined;
    let timestampSeconds = 0;
    let timestampLabel = `${lineIndex - 1}`;

    if (timestampNsRaw) {
      const tNs = parseNumber(timestampNsRaw);
      if (firstTimestampNs === null) firstTimestampNs = tNs;
      timestampSeconds = (tNs - firstTimestampNs) / 1e9;
      timestampLabel = timestampNsRaw;
    } else if (iTimestampS >= 0) {
      timestampSeconds = parseNumber(cols[iTimestampS] ?? "0");
      timestampLabel = String(timestampSeconds.toFixed(3));
    }

    let pitchDeg = iPitch >= 0 ? parseNumber(cols[iPitch] ?? "0") : undefined;
    let yawDeg = iYaw >= 0 ? parseNumber(cols[iYaw] ?? "0") : undefined;
    let rollDeg = iRoll >= 0 ? parseNumber(cols[iRoll] ?? "0") : undefined;

    // If Euler columns are missing but quaternion exists, derive Euler for display.
    if ((pitchDeg === undefined || yawDeg === undefined || rollDeg === undefined) && iQw >= 0 && iQx >= 0 && iQy >= 0 && iQz >= 0) {
      const e = quatToEulerDeg(
        parseNumber(cols[iQw] ?? "1"),
        parseNumber(cols[iQx] ?? "0"),
        parseNumber(cols[iQy] ?? "0"),
        parseNumber(cols[iQz] ?? "0"),
      );
      pitchDeg = e.pitchDeg;
      yawDeg = e.yawDeg;
      rollDeg = e.rollDeg;
    }

    out.push({
      index: out.length,
      timestampLabel,
      timestampNs: timestampNsRaw,
      timestampSeconds,
      x,
      y,
      z,
      pitchDeg,
      yawDeg,
      rollDeg,
    });
  }

  return out;
}

export function ReplayPage() {
  const [rows, setRows] = useState<ReplayRow[]>([]);
  const [boundRows, setBoundRows] = useState<BoundRow[]>([]);
  const [frameIndex, setFrameIndex] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [playbackSpeed, setPlaybackSpeed] = useState<0.5 | 1 | 2 | 4>(1); // 0.5x, 1x, 2x, 4x
  const baseFps = 15;
  const fps = baseFps * playbackSpeed;
  const [error, setError] = useState("");
  const [, setAutoStatus] = useState("Trying default replay assets...");
  const [imageUrl, setImageUrl] = useState("");

  const totalFrames = boundRows.length;
  const current = totalFrames > 0 ? boundRows[Math.min(frameIndex, totalFrames - 1)] : null;

  useEffect(() => {
    if (!playing || totalFrames <= 1) return;
    const intervalMs = Math.max(1, Math.round(1000 / Math.max(1, fps)));
    const id = window.setInterval(() => {
      setFrameIndex((prev) => (prev + 1 >= totalFrames ? 0 : prev + 1));
    }, intervalMs);
    return () => window.clearInterval(id);
  }, [playing, totalFrames, fps]);

  useEffect(() => {
    if (!current?.imageFile) {
      setImageUrl(current?.imageUrl ?? "");
      return;
    }
    const url = URL.createObjectURL(current.imageFile);
    setImageUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [current?.imageFile, current?.imageUrl]);

  const bindImagesToRows = (inputRows: ReplayRow[], imageNames: string[]): BoundRow[] => {
    const byStem = new Map<string, string>();
    imageNames.forEach((name) => {
      const stem = name.replace(/\.[^.]+$/, "");
      if (!byStem.has(stem)) byStem.set(stem, name);
    });

    return inputRows.map((row, index) => {
      let chosen = "";
      if (row.timestampNs && byStem.has(row.timestampNs)) {
        chosen = byStem.get(row.timestampNs) ?? "";
      } else if (index < imageNames.length) {
        chosen = imageNames[index];
      }
      return {
        ...row,
        imageUrl: chosen ? `${AUTO_IMAGES_BASE}/${encodeURIComponent(chosen)}` : undefined,
      };
    });
  };

  const tryLoadDefaults = async () => {
    setError("");
    setAutoStatus("Trying default replay assets...");
    setPlaying(false);
    setFrameIndex(0);
    try {
      const [trajRes, manifestRes] = await Promise.all([
        fetch(AUTO_TRAJECTORY_URL),
        fetch(AUTO_MANIFEST_URL),
      ]);
      if (!trajRes.ok || !manifestRes.ok) {
        throw new Error("Default replay assets not found in frontend/public/replay");
      }

      const trajText = await trajRes.text();
      const parsedRows = parseTrajectoryCsv(trajText);
      if (!parsedRows.length) {
        throw new Error("Default trajectory.csv is present but not parseable.");
      }

      const manifest = (await manifestRes.json()) as ImagesManifest;
      const imageNames = Array.isArray(manifest.images) ? manifest.images : [];
      const merged = bindImagesToRows(parsedRows, imageNames);

      setRows(parsedRows);
      setBoundRows(merged);
      setAutoStatus(`Loaded defaults: ${parsedRows.length} poses, ${imageNames.length} images`);
      setError("");
    } catch (err) {
      setRows([]);
      setBoundRows([]);
      setAutoStatus("No default replay assets found. Upload files manually or export replay assets.");
      if (err instanceof Error) {
        setError(err.message);
      }
    }
  };

  useEffect(() => {
    void tryLoadDefaults();
    // Run once on page mount.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const nodeLabel = useMemo(() => {
    if (!current || totalFrames === 0) return "Node 0 / 0";
    return `Node ${current.index + 1} / ${totalFrames}`;
  }, [current, totalFrames]);
  const timestampLabel = useMemo(() => {
    if (!current) return "";
    return current.timestampSeconds.toFixed(3) + "s";
  }, [current]);

  const onTrajectoryFile = async (file: File | null) => {
    setError("");
    setPlaying(false);
    setFrameIndex(0);
    if (!file) {
      setRows([]);
      setBoundRows([]);
      return;
    }
    try {
      const text = await file.text();
      const parsed = parseTrajectoryCsv(text);
      if (!parsed.length) {
        setError("Could not parse trajectory CSV (missing expected headers).");
        setRows([]);
        setBoundRows([]);
        return;
      }
      setRows(parsed);
      setBoundRows(parsed.map((row) => ({ ...row })));
      setAutoStatus(`Loaded uploaded trajectory: ${parsed.length} poses`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to read trajectory file");
      setRows([]);
      setBoundRows([]);
    }
  };

  const onImagesSelected = (files: FileList | null) => {
    if (!files || rows.length === 0) return;
    const picked = Array.from(files).filter((file) => /\.(png|jpe?g|webp)$/i.test(file.name));
    if (!picked.length) {
      setError("No image files selected.");
      return;
    }

    const byStem = new Map<string, File>();
    for (const file of picked) {
      const stem = file.name.replace(/\.[^.]+$/, "");
      if (!byStem.has(stem)) byStem.set(stem, file);
    }
    const sortedByName = [...picked].sort((a, b) => a.name.localeCompare(b.name));

    const merged = rows.map((row, index) => {
      let imageFile: File | undefined;
      if (row.timestampNs && byStem.has(row.timestampNs)) {
        imageFile = byStem.get(row.timestampNs);
      }
      if (!imageFile) {
        imageFile = sortedByName[index];
      }
      return { ...row, imageFile };
    });

    setBoundRows(merged);
    setFrameIndex(0);
    setAutoStatus(`Loaded uploaded images: ${picked.length}`);
    setError("");
  };

  /** Load frames from imported video (world graph). Add video on Agent or 3D World first. */
  const loadFromImportedVideo = useCallback(async () => {
    setError("");
    setPlaying(false);
    setFrameIndex(0);
    try {
      const base = replayApiBase();
      const r = await fetch(`${base}/api/graph_3d`);
      if (!r.ok) throw new Error("Failed to load map data");
      const data = await r.json();
      const nodes: Array<{ node_id?: string; image_b64?: string | null; pose?: [number, number, number] | null }> =
        data.nodes || [];
      const withFrames = nodes.filter((n) => n.image_b64);
      if (withFrames.length === 0) {
        setError("No video frames in map. Add a recorded video on the Agent or 3D World tab first.");
        setRows([]);
        setBoundRows([]);
        setAutoStatus("No imported video — add video on Agent or 3D World first.");
        return;
      }
      const bound: BoundRow[] = withFrames.map((n, i) => {
        const [x = 0, y = 0, z = 0] = n.pose || [];
        return {
          index: i,
          timestampLabel: `${i + 1}`,
          timestampSeconds: i * 0.1,
          x,
          y,
          z,
          imageUrl: n.image_b64 ? `data:image/jpeg;base64,${n.image_b64}` : undefined,
        };
      });
      setRows(bound.map((b) => ({ ...b, imageUrl: undefined })));
      setBoundRows(bound);
      setAutoStatus(`Loaded imported video: ${bound.length} frames. Use Play to replay.`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load imported video");
      setRows([]);
      setBoundRows([]);
      setAutoStatus("Load failed. Is the backend running?");
    }
  }, []);

  return (
    <section className="replay-page" style={{ display: "flex", flexDirection: "column", height: "100%", overflow: "hidden" }}>
      {/* Frame top 85% — hard cut, no overlays except timestamp top-left */}
      <div style={{ flex: "0 0 85%", minHeight: 0, position: "relative", background: "#000" }}>
        {imageUrl ? (
          <img src={imageUrl} alt="Replay frame" style={{ width: "100%", height: "100%", objectFit: "contain" }} />
        ) : (
          <div style={{ width: "100%", height: "100%", display: "flex", alignItems: "center", justifyContent: "center", color: "rgba(255,255,255,0.5)", fontSize: "0.9rem" }}>
            Import video on 3D World or Agent, then click &quot;Load from imported video&quot;. Or upload CSV + images.
          </div>
        )}
        {timestampLabel && (
          <div style={{ position: "absolute", top: 8, left: 8, fontFamily: "monospace", fontSize: "0.85rem", color: "#fff", background: "rgba(0,0,0,0.6)", padding: "0.25rem 0.5rem", borderRadius: 4 }}>
            {timestampLabel}
          </div>
        )}
        {!!error && <div style={{ position: "absolute", bottom: 8, left: 8, right: 8, color: "#f88", fontSize: "0.8rem" }}>{error}</div>}
      </div>

      {/* Controls bottom 15%: scrubber, PLAY/PAUSE, speed 0.5x/1x/2x/4x, Node N / M */}
      <div style={{ flex: "0 0 15%", minHeight: 0, padding: "0.5rem 1rem", display: "flex", flexDirection: "column", gap: 8, background: "rgba(0,0,0,0.3)", borderTop: "1px solid rgba(255,255,255,0.1)" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <input
            type="range"
            min={0}
            max={Math.max(0, totalFrames - 1)}
            value={Math.min(frameIndex, Math.max(0, totalFrames - 1))}
            onChange={(e) => setFrameIndex(parseInt(e.target.value, 10))}
            disabled={totalFrames === 0}
            style={{ flex: 1, minWidth: 0 }}
          />
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
          <button type="button" className="replay-btn" onClick={() => setPlaying((p) => !p)} disabled={totalFrames <= 1}>
            {playing ? "PAUSE" : "PLAY"}
          </button>
          {([0.5, 1, 2, 4] as const).map((s) => (
            <button
              key={s}
              type="button"
              className="replay-btn"
              style={{ padding: "0.25rem 0.5rem", background: playbackSpeed === s ? "var(--swiss-accent)" : "transparent" }}
              onClick={() => setPlaybackSpeed(s)}
            >
              {s}x
            </button>
          ))}
          <span style={{ fontFamily: "monospace", fontSize: "0.85rem", color: "rgba(255,255,255,0.9)" }}>{nodeLabel}</span>
          <button type="button" className="replay-btn" onClick={loadFromImportedVideo} style={{ marginLeft: "auto" }}>Load from imported video</button>
          <label className="replay-btn" style={{ cursor: "pointer", margin: 0 }}>CSV<input type="file" accept=".csv" style={{ display: "none" }} onChange={(e) => onTrajectoryFile(e.target.files?.[0] ?? null)} /></label>
          <label className="replay-btn" style={{ cursor: "pointer", margin: 0 }}>Images<input type="file" accept=".png,.jpg,.jpeg,.webp" multiple style={{ display: "none" }} onChange={(e) => onImagesSelected(e.target.files)} /></label>
          <Link to="/3d-map" className="replay-btn" style={{ textDecoration: "none", color: "inherit" }}>3D World</Link>
          <Link to="/agent" className="replay-btn" style={{ textDecoration: "none", color: "inherit" }}>Agent</Link>
        </div>
      </div>
    </section>
  );
}


