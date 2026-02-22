type TrajectoryData = {
  agentId: number;
  points: Array<{ x: number; y: number; step: number }>;
  isWinner: boolean;
};

type Props = {
  trajectories: TrajectoryData[];
  width?: number;
  height?: number;
};

export function AgentTrajectoryMap({
  trajectories,
  width = 180,
  height = 180,
}: Props) {
  // Collect all points to compute bounds
  const allPoints = trajectories.flatMap((t) => t.points);
  if (allPoints.length === 0) {
    return (
      <svg width={width} height={height} style={{ display: "block" }}>
        <rect width={width} height={height} fill="rgba(0,0,0,0.85)" />
        <text
          x={width / 2}
          y={height / 2}
          textAnchor="middle"
          fill="rgba(255,255,255,0.3)"
          fontSize={10}
          fontWeight={700}
          fontFamily="ui-monospace, 'SF Mono', Menlo, monospace"
        >
          NO DATA
        </text>
      </svg>
    );
  }

  const padding = 20;
  const xs = allPoints.map((p) => p.x);
  const ys = allPoints.map((p) => p.y);
  let minX = Math.min(...xs);
  let maxX = Math.max(...xs);
  let minY = Math.min(...ys);
  let maxY = Math.max(...ys);

  // Ensure non-zero range
  if (maxX - minX < 0.1) {
    minX -= 1;
    maxX += 1;
  }
  if (maxY - minY < 0.1) {
    minY -= 1;
    maxY += 1;
  }

  const rangeX = maxX - minX;
  const rangeY = maxY - minY;
  const drawW = width - padding * 2;
  const drawH = height - padding * 2;

  const toSvg = (x: number, y: number) => ({
    sx: padding + ((x - minX) / rangeX) * drawW,
    sy: padding + ((y - minY) / rangeY) * drawH,
  });

  const COLORS = ["#FF3000", "#FFFFFF", "#888888", "#CCCCCC"];

  return (
    <svg width={width} height={height} style={{ display: "block" }}>
      <rect width={width} height={height} fill="rgba(0,0,0,0.85)" />

      {/* Grid lines */}
      {[0.25, 0.5, 0.75].map((f) => (
        <g key={f} opacity={0.15}>
          <line
            x1={padding}
            y1={padding + f * drawH}
            x2={width - padding}
            y2={padding + f * drawH}
            stroke="white"
            strokeWidth={0.5}
          />
          <line
            x1={padding + f * drawW}
            y1={padding}
            x2={padding + f * drawW}
            y2={height - padding}
            stroke="white"
            strokeWidth={0.5}
          />
        </g>
      ))}

      {trajectories.map((traj, idx) => {
        if (traj.points.length === 0) return null;
        const color = traj.isWinner ? COLORS[0] : COLORS[idx + 1] ?? COLORS[2];
        const svgPoints = traj.points.map((p) => toSvg(p.x, p.y));
        const pathD = svgPoints
          .map((p, i) => `${i === 0 ? "M" : "L"} ${p.sx} ${p.sy}`)
          .join(" ");

        const start = svgPoints[0];
        const end = svgPoints[svgPoints.length - 1];

        return (
          <g key={traj.agentId}>
            {/* Path line */}
            <path
              d={pathD}
              fill="none"
              stroke={color}
              strokeWidth={traj.isWinner ? 2 : 1.5}
              strokeDasharray={traj.isWinner ? "none" : "4 3"}
              opacity={0.9}
            />

            {/* Start marker (square) */}
            <rect
              x={start.sx - 3}
              y={start.sy - 3}
              width={6}
              height={6}
              fill="none"
              stroke={color}
              strokeWidth={1.5}
            />

            {/* End marker (circle) */}
            <circle
              cx={end.sx}
              cy={end.sy}
              r={4}
              fill={traj.isWinner ? color : "none"}
              stroke={color}
              strokeWidth={1.5}
            />

            {/* Step dots */}
            {svgPoints.slice(1, -1).map((p, i) => (
              <circle
                key={i}
                cx={p.sx}
                cy={p.sy}
                r={2}
                fill={color}
                opacity={0.6}
              />
            ))}
          </g>
        );
      })}
    </svg>
  );
}
