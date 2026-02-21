import { Route, Routes, useLocation } from "react-router-dom";
import { LandingPage } from "./routes/LandingPage";
import { AgentPage } from "./routes/AgentPage";
import { ManualPage } from "./routes/ManualPage";
import { ReplayPage } from "./routes/ReplayPage";
import { ModeSwitch } from "./components/ModeSwitch";

export function App() {
  const location = useLocation();
  const isLanding = location.pathname === "/";

  return (
    <div className="app-shell">
      {!isLanding && (
        <header className="top-bar">
          <h1 className="logo">DRONE VISION</h1>
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
        </Routes>
      </main>
    </div>
  );
}

