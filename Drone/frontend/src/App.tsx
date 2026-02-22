import { useEffect, useState } from "react";
import { Route, Routes, useLocation, useNavigate } from "react-router-dom";
import { LandingPage } from "./routes/LandingPage";
import { AgentPage } from "./routes/AgentPage";
import { ManualPage } from "./routes/ManualPage";
import { ReplayPage } from "./routes/ReplayPage";
import { Map3DPage } from "./routes/Map3DPage";
import { ModeSwitch } from "./components/ModeSwitch";
import { API_BASE_URL } from "./config";

const TAB_ROUTES = ["/agent", "/manual", "/replay", "/3d-map"] as const;

type ImportStatus = { status: string; current: number; total: number; message: string; nodes_added?: number };

export function App() {
  const location = useLocation();
  const navigate = useNavigate();
  const isLanding = location.pathname === "/";
  const [importStatus, setImportStatus] = useState<ImportStatus | null>(null);

  useEffect(() => {
    if (isLanding) return;
    const base = API_BASE_URL || (typeof window !== "undefined" ? window.location.origin : "");
    const poll = async () => {
      try {
        const r = await fetch(`${base}/api/import_video/status`);
        const s: ImportStatus = await r.json();
        setImportStatus(s);
      } catch {}
    };
    poll();
    const t = setInterval(poll, 500);
    return () => clearInterval(t);
  }, [isLanding]);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key !== "Tab" || isLanding) return;
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement || (e.target as HTMLElement).isContentEditable) return;
      e.preventDefault();
      const path = location.pathname;
      const idx = TAB_ROUTES.findIndex((r) => path === r || path.startsWith(r + "/"));
      const nextIdx = idx >= 0 ? (idx + 1) % TAB_ROUTES.length : 0;
      navigate(TAB_ROUTES[nextIdx]);
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [location.pathname, navigate, isLanding]);

  return (
    <div className="app-shell">
      {!isLanding && (
        <header className="top-bar">
          <h1 className="logo">CIPHER</h1>
          <ModeSwitch />
        </header>
      )}
      <main className="page-content">
        <Routes>
          <Route path="/" element={<LandingPage />} />
          <Route path="/manual" element={<ManualPage />} />
          <Route path="/agent/:sessionId" element={<AgentPage />} />
          <Route path="/agent" element={<AgentPage />} />
          <Route path="/replay" element={<ReplayPage />} />
          <Route path="/3d-map" element={<Map3DPage />} />
        </Routes>
      </main>
      {!isLanding && importStatus?.status === "running" && (
        <div
          className="global-import-bar"
          style={{
            position: "fixed",
            bottom: 0,
            left: 0,
            right: 0,
            height: 20,
            background: "rgba(0,60,0,0.9)",
            color: "#8f8",
            fontSize: 11,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            zIndex: 10000,
          }}
        >
          Importing video... {importStatus.total ? Math.round((100 * (importStatus.current || 0)) / importStatus.total) : 0}% · {(importStatus.nodes_added ?? 0)} nodes built
        </div>
      )}
    </div>
  );
}

