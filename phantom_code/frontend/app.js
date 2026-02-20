/**
 * PHANTOM CODE — Tactical map, detection display, mission selector, LLM advisory panel
 */

const MAP_WIDTH = 800;
const MAP_HEIGHT = 600;

// Zone layout (must match backend config)
const CAMERA_ZONES = {
  "Drone-1": { x_min: 50, y_min: 50, x_max: 370, y_max: 550 },
  "Drone-2": { x_min: 430, y_min: 50, x_max: 750, y_max: 550 },
};
const ZONE_LABELS = { "Drone-1": "SECTOR ALPHA", "Drone-2": "SECTOR BRAVO" };

// Class to color (person=green, vehicle=blue, object=gray)
const CLASS_COLORS = {
  person: "#00ff66",
  bicycle: "#4488ff",
  car: "#4488ff",
  motorcycle: "#4488ff",
  bus: "#4488ff",
  truck: "#4488ff",
  default: "#888888",
};

const canvas = document.getElementById("tactical-map");
const ctx = canvas.getContext("2d");
const advisoryText = document.getElementById("advisory-text");
const advisoryMeta = document.getElementById("advisory-meta");
const statusNpu = document.getElementById("status-npu");
const statusYolo = document.getElementById("status-yolo");
const statusDrone1 = document.getElementById("status-drone-1");
const statusDrone2 = document.getElementById("status-drone-2");
const statusLlm = document.getElementById("status-llm");
const statusMode = document.getElementById("status-mode");

let currentMission = "search_rescue";
let lastAdvisoryText = "";
let pulsePhase = {};

function getClassColor(className) {
  return CLASS_COLORS[className] || CLASS_COLORS.default;
}

function drawGrid() {
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
}

function drawZones(feeds) {
  const droneIds = ["Drone-1", "Drone-2"];
  droneIds.forEach((droneId) => {
    const zone = CAMERA_ZONES[droneId];
    if (!zone) return;
    const active = feeds && feeds[droneId];
    ctx.strokeStyle = active ? "#00ff66" : "#ff4444";
    ctx.setLineDash([6, 4]);
    ctx.lineWidth = 2;
    ctx.strokeRect(zone.x_min, zone.y_min, zone.x_max - zone.x_min, zone.y_max - zone.y_min);
    ctx.setLineDash([]);

    const label = ZONE_LABELS[droneId] || droneId;
    ctx.font = "11px Courier New";
    ctx.fillStyle = active ? "#00ff66" : "#ff4444";
    ctx.fillText(label, zone.x_min + 4, zone.y_min + 14);
    if (!active) {
      ctx.fillStyle = "#ff4444";
      ctx.font = "12px Courier New";
      ctx.fillText("FEED LOST", zone.x_min + (zone.x_max - zone.x_min) / 2 - 30, zone.y_min + (zone.y_max - zone.y_min) / 2 - 4);
    }
  });
}

function drawDetections(detections) {
  if (!detections) return;
  const droneIds = ["Drone-1", "Drone-2"];
  droneIds.forEach((droneId) => {
    const list = detections[droneId] || [];
    const zone = CAMERA_ZONES[droneId];
    if (!zone) return;
    list.forEach((d, i) => {
      const mx = d.map_x;
      const my = d.map_y;
      const key = `${droneId}-${i}-${mx}-${my}`;
      if (!pulsePhase[key]) pulsePhase[key] = 0;
      pulsePhase[key] += 0.08;
      const pulse = 0.7 + 0.3 * Math.sin(pulsePhase[key]);
      const r = 6 * pulse;
      const color = getClassColor(d.class);
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(mx, my, r, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = "#0a0f19";
      ctx.lineWidth = 1;
      ctx.stroke();
      ctx.fillStyle = "#c0c8d4";
      ctx.font = "10px Courier New";
      ctx.fillText(`${d.class} ${(d.confidence * 100).toFixed(0)}%`, mx - 20, my + r + 12);
    });
  });
  // Trim pulsePhase to avoid unbounded growth
  const keys = Object.keys(pulsePhase);
  if (keys.length > 200) {
    keys.slice(0, 100).forEach((k) => delete pulsePhase[k]);
  }
}

function drawNoContacts(detections, feeds) {
  const droneIds = ["Drone-1", "Drone-2"];
  droneIds.forEach((droneId) => {
    const zone = CAMERA_ZONES[droneId];
    if (!zone || !feeds[droneId]) return;
    const list = detections[droneId] || [];
    if (list.length === 0) {
      ctx.fillStyle = "rgba(192, 200, 212, 0.25)";
      ctx.font = "11px Courier New";
      ctx.fillText("NO CONTACTS", zone.x_min + (zone.x_max - zone.x_min) / 2 - 35, zone.y_min + (zone.y_max - zone.y_min) / 2);
    }
  });
}

function renderMap(detections, feeds) {
  ctx.fillStyle = "#0a0f19";
  ctx.fillRect(0, 0, MAP_WIDTH, MAP_HEIGHT);
  drawGrid();
  drawZones(feeds);
  drawDetections(detections);
  drawNoContacts(detections, feeds);
}

async function fetchJson(url) {
  try {
    const r = await fetch(url);
    return r.ok ? await r.json() : null;
  } catch {
    return null;
  }
}

async function pollDetections() {
  const data = await fetchJson("/api/detections");
  if (data) {
    const status = await fetchJson("/api/status");
    const feeds = status ? status.feeds : {};
    renderMap(data, feeds);
  }
}

async function pollAdvisory() {
  const data = await fetchJson("/api/advisory");
  if (data && data.text) {
    advisoryText.textContent = data.text;
    const missionLabel = data.mission ? data.mission.replace(/_/g, " ") : "—";
    advisoryMeta.textContent = `Mission: ${missionLabel} | ${data.timestamp || ""}`;
    lastAdvisoryText = data.text;
  }
}

async function pollStatus() {
  const data = await fetchJson("/api/status");
  if (!data) return;
  const npu = data.npu_provider || "—";
  const isQnn = npu.indexOf("QNN") >= 0;
  statusNpu.textContent = `NPU: ${npu} ${isQnn ? "✓" : "✗"}`;
  statusNpu.className = isQnn ? "ok" : "bad";
  const ms = data.yolo_latency_ms != null ? Math.round(data.yolo_latency_ms) : "—";
  statusYolo.textContent = `YOLO: ${ms}ms`;
  statusYolo.className = ms < 100 ? "ok" : ms < 200 ? "warn" : "bad";
  statusDrone1.textContent = data.feeds["Drone-1"] ? "Drone-1: ACTIVE" : "Drone-1: FEED LOST";
  statusDrone1.className = data.feeds["Drone-1"] ? "ok" : "bad";
  statusDrone2.textContent = data.feeds["Drone-2"] ? "Drone-2: ACTIVE" : "Drone-2: FEED LOST";
  statusDrone2.className = data.feeds["Drone-2"] ? "ok" : "bad";
}

function setMissionButtons() {
  document.querySelectorAll(".mission-buttons button").forEach((btn) => {
    const m = btn.getAttribute("data-mission");
    btn.classList.toggle("selected", m === currentMission);
  });
}

document.querySelectorAll(".mission-buttons button").forEach((btn) => {
  btn.addEventListener("click", async () => {
    const mission = btn.getAttribute("data-mission");
    if (!mission) return;
    currentMission = mission;
    setMissionButtons();
    advisoryText.textContent = "Updating tactical analysis...";
    try {
      await fetch("/api/mission", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ mission }),
      });
    } catch (e) {
      console.warn(e);
    }
  });
});

// Feed thumbnails
function updateFeedImages() {
  ["Drone-1", "Drone-2"].forEach((id) => {
    const el = document.getElementById(`feed-${id.toLowerCase()}`);
    if (el) el.src = `/api/feed/${id}?t=${Date.now()}`;
  });
}

// Polling
setInterval(pollDetections, 200);
setInterval(pollAdvisory, 3000);
setInterval(pollStatus, 2000);
setInterval(updateFeedImages, 500);

// Initial paint and mission selection
setMissionButtons();
renderMap({}, {});
