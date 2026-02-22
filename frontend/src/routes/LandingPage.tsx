import { Link } from "react-router-dom";
import { ArrowRight, Map, Play, Gauge } from "lucide-react";

export function LandingPage() {
  return (
    <div className="landing-page">
      {/* Hero Section */}
      <section className="hero-section">
        <div className="hero-content">
          <span className="hero-label">01. SYSTEM</span>
          <h1 className="hero-title">
            CIPHER
          </h1>
          <p className="hero-description">
            AI-powered tactical system with real-time object detection.
            NPU-accelerated YOLO processing with Llama Vision reasoning.
            Engineered for autonomous exploration and spatial mapping.
          </p>
          <Link to="/agent" className="hero-cta">
            Launch System
            <ArrowRight size={20} strokeWidth={2.5} />
          </Link>
        </div>

        <div className="hero-composition">
          <div className="composition-grid">
            {/* Row 1 */}
            <div className="composition-element bordered swiss-grid-pattern" />
            <div className="composition-element bordered">
              <div className="composition-circle" />
            </div>
            <div className="composition-element bordered swiss-dots" />
            <div className="composition-element bordered">
              <div className="composition-line horizontal" />
            </div>

            {/* Row 2 */}
            <div className="composition-element bordered">
              <div className="composition-line vertical" />
            </div>
            <div
              className="composition-element bordered"
              style={{ background: "var(--swiss-black)" }}
            />
            <div className="composition-element bordered swiss-diagonal" />
            <div className="composition-element bordered">
              <div className="composition-circle" style={{ width: "40%" }} />
            </div>

            {/* Row 3 */}
            <div className="composition-element bordered">
              <div className="composition-square" />
            </div>
            <div className="composition-element bordered swiss-grid-pattern" />
            <div
              className="composition-element bordered"
              style={{ background: "var(--swiss-accent)" }}
            />
            <div className="composition-element bordered">
              <div className="composition-line horizontal" />
              <div className="composition-line vertical" />
            </div>

            {/* Row 4 */}
            <div className="composition-element bordered swiss-dots" />
            <div className="composition-element bordered">
              <div className="composition-line vertical" />
            </div>
            <div className="composition-element bordered swiss-diagonal" />
            <div className="composition-element bordered">
              <div className="composition-circle" style={{ width: "80%" }} />
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="features-section">
        <div className="section-header">
          <span className="section-number">02. CAPABILITIES</span>
          <h1 className="landing-title">DRONE VISION</h1>
          <h2 className="section-title">THREE MODES</h2>
        </div>

        <div className="features-grid">
          <Link to="/agent" className="feature-card" style={{ textDecoration: "none", color: "inherit" }}>
            <div className="feature-icon">
              <Gauge size={28} strokeWidth={2} />
            </div>
            <h3 className="feature-title">Agent Mode</h3>
            <p className="feature-description">
              Autonomous navigation through learned policies. Intelligent pathfinding
              and obstacle avoidance. AI-driven exploration of reconstructed
              environments.
            </p>
          </Link>

          <Link to="/manual" className="feature-card" style={{ textDecoration: "none", color: "inherit" }}>
            <div className="feature-icon">
              <Map size={28} strokeWidth={2} />
            </div>
            <h3 className="feature-title">Manual Control</h3>
            <p className="feature-description">
              Direct pose manipulation with 6-DOF freedom. Navigate three-dimensional
              space with precision keyboard controls. Real-time image synthesis from
              arbitrary viewpoints.
            </p>
          </Link>

          <Link to="/replay" className="feature-card" style={{ textDecoration: "none", color: "inherit" }}>
            <div className="feature-icon">
              <Play size={28} strokeWidth={2} />
            </div>
            <h3 className="feature-title">Trajectory Replay</h3>
            <p className="feature-description">
              Temporal visualization of recorded paths. Frame-accurate playback with
              variable speed control. Upload custom trajectory data for analysis and
              validation.
            </p>
          </Link>
        </div>
      </section>

      {/* Technical Specs */}
      <section className="features-section" style={{ borderBottom: "none" }}>
        <div className="section-header">
          <span className="section-number">03. SPECIFICATIONS</span>
          <h2 className="section-title">TECHNICAL</h2>
        </div>

        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: "var(--space-8)" }}>
          <div style={{ padding: "var(--space-6)", border: "var(--border-2) solid var(--swiss-black)" }}>
            <div style={{ fontSize: "3rem", fontWeight: 900, lineHeight: 1, marginBottom: "var(--space-2)" }}>
              6-DOF
            </div>
            <div style={{ fontSize: "0.875rem", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.05em" }}>
              Degrees of Freedom
            </div>
          </div>

          <div style={{ padding: "var(--space-6)", border: "var(--border-2) solid var(--swiss-black)", background: "var(--swiss-muted)" }} className="swiss-grid-pattern">
            <div style={{ fontSize: "3rem", fontWeight: 900, lineHeight: 1, marginBottom: "var(--space-2)" }}>
              60Hz
            </div>
            <div style={{ fontSize: "0.875rem", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.05em" }}>
              Update Rate
            </div>
          </div>

          <div style={{ padding: "var(--space-6)", border: "var(--border-2) solid var(--swiss-black)" }}>
            <div style={{ fontSize: "3rem", fontWeight: 900, lineHeight: 1, marginBottom: "var(--space-2)" }}>
              &lt;2ms
            </div>
            <div style={{ fontSize: "0.875rem", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.05em" }}>
              Latency
            </div>
          </div>

          <div style={{ padding: "var(--space-6)", border: "var(--border-2) solid var(--swiss-black)", background: "var(--swiss-muted)" }} className="swiss-dots">
            <div style={{ fontSize: "3rem", fontWeight: 900, lineHeight: 1, marginBottom: "var(--space-2)" }}>
              ∞
            </div>
            <div style={{ fontSize: "0.875rem", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.05em" }}>
              View Range
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}

