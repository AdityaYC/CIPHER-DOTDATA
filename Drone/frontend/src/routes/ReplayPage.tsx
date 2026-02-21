import { useEffect, useMemo, useState } from "react";

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
  const [fps, setFps] = useState(15);
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

  const progressLabel = useMemo(() => {
    if (!current) return "No trajectory loaded";
    return `Frame ${current.index + 1}/${totalFrames} | t=${current.timestampSeconds.toFixed(3)}s`;
  }, [current, totalFrames]);

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

  return (
    <section className="replay-page">
      {/* Viewport takes most of the space */}
      <div className="viewport-card replay-viewport">
        {imageUrl ? (
          <img
            className="viewport-image"
            src={imageUrl}
            alt="Replay frame"
          />
        ) : (
          <div className="replay-empty">Load trajectory CSV + images to begin replay.</div>
        )}
        {current && (
          <div className="replay-coords">
            <div>x: {current.x.toFixed(3)} m</div>
            <div>y: {current.y.toFixed(3)} m</div>
            <div>z: {current.z.toFixed(3)} m</div>
          </div>
        )}
        {/* Status overlay */}
        <div className="replay-status-overlay">
          <span className="pose">{progressLabel}</span>
          {!!error && <span className="status error">{error}</span>}
        </div>
      </div>

      {/* Controls bar at bottom */}
      <div className="replay-bottom-bar">
        <div className="replay-slider-wrap">
          <input
            className="replay-slider"
            type="range"
            min={0}
            max={Math.max(0, totalFrames - 1)}
            value={Math.min(frameIndex, Math.max(0, totalFrames - 1))}
            onChange={(e) => setFrameIndex(parseInt(e.target.value, 10))}
            disabled={totalFrames === 0}
          />
        </div>

        <div className="replay-controls">
          <button type="button" className="replay-btn" onClick={() => void tryLoadDefaults()}>
            Reload
          </button>
          <button
            type="button"
            className="replay-btn"
            onClick={() => setPlaying((p) => !p)}
            disabled={totalFrames <= 1}
          >
            {playing ? "Pause" : "Play"}
          </button>
          <button
            type="button"
            className="replay-btn"
            onClick={() => setFrameIndex((i) => Math.max(0, i - 1))}
            disabled={totalFrames === 0}
          >
            Prev
          </button>
          <button
            type="button"
            className="replay-btn"
            onClick={() => setFrameIndex((i) => Math.min(totalFrames - 1, i + 1))}
            disabled={totalFrames === 0}
          >
            Next
          </button>
          <label className="replay-fps">
            FPS
            <input
              type="number"
              min={1}
              max={120}
              value={fps}
              onChange={(e) => setFps(Math.max(1, parseInt(e.target.value || "1", 10)))}
            />
          </label>

          <div className="replay-tools">
            <label className="replay-input">
              Upload CSV
              <input
                type="file"
                accept=".csv,text/csv"
                onChange={(e) => onTrajectoryFile(e.target.files?.[0] ?? null)}
              />
            </label>
            <label className="replay-input">
              Upload Images
              <input
                type="file"
                accept=".png,.jpg,.jpeg,.webp"
                multiple
                onChange={(e) => onImagesSelected(e.target.files)}
              />
            </label>
          </div>
        </div>
      </div>
    </section>
  );
}


