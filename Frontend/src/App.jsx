import { useEffect, useState, useRef } from "react";
import { io } from "socket.io-client";

const SOCKET_URL = "http://localhost:8000";

function LiveVideoFeed({ frameData, keypoints, width, height }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    if (!frameData || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");

    // Create image from base64
    const img = new Image();
    img.onload = () => {
      canvas.width = img.width;
      canvas.height = img.height;
      ctx.drawImage(img, 0, 0);
    };
    img.src = `data:image/jpeg;base64,${frameData}`;
  }, [frameData]);

  return (
    <div className="video-container">
      <div className="video-header">
        <span className="live-badge">● LIVE</span>
        <span className="resolution">
          {width}x{height}
        </span>
      </div>
      <canvas ref={canvasRef} className="video-canvas" />
    </div>
  );
}

function HypeGauge({ value }) {
  const v = Math.max(0, Math.min(1, value ?? 0)); // clamp 0..1
  const rotation = -135 + v * 270;
  const hue = v * 120;

  // We set pathLength=100, so "100" is the full arc length in dash units
  const dashArray = 100;
  const dashOffset = 100 - v * 100;

  return (
    <div className="hype-gauge">
      <svg
        viewBox="0 0 200 200"
        className="gauge-svg"
        preserveAspectRatio="xMidYMid meet"
      >
        {/* Track */}
        <path
          d="M 30 150 A 80 80 0 1 1 170 150"
          fill="none"
          stroke="rgba(255,255,255,0.1)"
          strokeWidth="12"
          strokeLinecap="round"
          pathLength="100"
        />

        {/* Progress (single continuous stroke, no split ends) */}
        <path
          d="M 30 150 A 80 80 0 1 1 170 150"
          fill="none"
          stroke={`hsl(${hue}, 100%, 50%)`}
          strokeWidth="12"
          strokeLinecap="round"
          pathLength="100"
          strokeDasharray={dashArray}
          strokeDashoffset={dashOffset}
          style={{
            filter: `drop-shadow(0 0 10px hsl(${hue}, 100%, 50%))`,
            transition: "stroke-dashoffset 150ms linear, stroke 150ms linear",
          }}
        />

        {/* Needle */}
        <g transform={`rotate(${rotation} 100 100)`}>
          <line
            x1="100"
            y1="100"
            x2="100"
            y2="20"
            stroke="white"
            strokeWidth="3"
            strokeLinecap="round"
          />
          <circle cx="100" cy="100" r="8" fill="white" />
        </g>

        {/* Text */}
        <text x="100" y="125" textAnchor="middle" className="gauge-text">
          {(v * 100).toFixed(0)}%
        </text>
        <text x="100" y="150" textAnchor="middle" className="gauge-label">
          HYPE
        </text>
      </svg>
    </div>
  );
}

function EnergyBar({ value, max = 15 }) {
  const pct = Math.min(100, (value / max) * 100);
  const hue = Math.min(120, pct * 1.2);

  return (
    <div className="energy-bar">
      <div className="energy-label">ENERGY</div>
      <div className="energy-track">
        <div
          className="energy-fill"
          style={{
            width: `${pct}%`,
            background: `linear-gradient(90deg, hsl(${
              hue * 0.3
            }, 100%, 50%), hsl(${hue}, 100%, 50%))`,
          }}
        />
      </div>
      <div className="energy-value">{value.toFixed(1)}</div>
    </div>
  );
}

function Heatmap({ data, zones }) {
  if (!data || data.length === 0) return null;
  const maxVal = Math.max(...data.flat(), 1);

  return (
    <div className="heatmap">
      <div className="heatmap-title">ENERGY ZONES</div>
      <div
        className="heatmap-grid"
        style={{
          gridTemplateColumns: `repeat(${data[0]?.length || 8}, 1fr)`,
        }}
      >
        {data.map((row, i) =>
          row.map((val, j) => {
            const intensity = val / maxVal;
            const zone = zones?.[i]?.[j] || "low";
            const colors = { low: 200, medium: 45, high: 0 };
            const h = colors[zone];
            return (
              <div
                key={`${i}-${j}`}
                className={`heatmap-cell ${zone}`}
                style={{
                  backgroundColor: `hsla(${h}, 100%, ${30 + intensity * 40}%, ${
                    0.3 + intensity * 0.7
                  })`,
                  boxShadow:
                    intensity > 0.6
                      ? `0 0 ${intensity * 15}px hsla(${h}, 100%, 50%, 0.8)`
                      : "none",
                }}
              />
            );
          })
        )}
      </div>
    </div>
  );
}

function StatCard({ icon, value, label }) {
  return (
    <div className="stat-card">
      <span className="stat-icon">{icon}</span>
      <div className="stat-info">
        <div className="stat-value">{value}</div>
        <div className="stat-label">{label}</div>
      </div>
    </div>
  );
}

export default function App() {
  const [connected, setConnected] = useState(false);
  const [data, setData] = useState(null);

  useEffect(() => {
    const socket = io(SOCKET_URL, {
      transports: ["websocket", "polling"],
      reconnection: true,
    });

    socket.on("connect", () => {
      console.log("Connected!");
      setConnected(true);
    });

    socket.on("disconnect", () => {
      console.log("Disconnected");
      setConnected(false);
    });

    socket.on("vibe_update", (payload) => {
      setData(payload);
    });

    return () => socket.disconnect();
  }, []);

  return (
    <div className="app">
      <header className="header">
        <h1 className="logo">◉ VIBE-CHECK</h1>
        <div className={`status ${connected ? "online" : "offline"}`}>
          <span className="status-dot" />
          {connected ? "LIVE" : "OFFLINE"}
          {data && (
            <span className="latency">
              {data.latencyMs}ms | {data.fps} FPS
            </span>
          )}
        </div>
      </header>

      {!connected && !data && (
        <div className="loading">
          <div className="spinner" />
          <p>Connecting to camera...</p>
          <p className="hint">Make sure backend is running: python main.py</p>
        </div>
      )}

      {data && (
        <main className="main">
          <div className="left-panel">
            <LiveVideoFeed
              frameData={data.frame}
              keypoints={data.keypoints}
              width={data.frameWidth}
              height={data.frameHeight}
            />

            <div className="stats-row">
              <StatCard value={data.peopleCount} label="People" />
              <StatCard
                value={`${(data.handsUpRatio * 100).toFixed(0)}%`}
                label="Hands Up"
              />
              <StatCard
                value={`${(data.jumpingRatio * 100).toFixed(0)}%`}
                label="Jumping"
              />
            </div>
          </div>

          <div className="right-panel">
            <HypeGauge value={data.hypeScore} />
            <EnergyBar value={data.meanEnergy} />
            <Heatmap data={data.heatmap} zones={data.zones} />
          </div>
        </main>
      )}
    </div>
  );
}
