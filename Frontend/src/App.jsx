import { useEffect, useState, useRef } from "react";
import { io } from "socket.io-client";

const SOCKET_URL = "http://localhost:8000";

// ============================================================================
// FAKE GENRE DATA FOR DEMO
// Each genre has a COUNT (number of people who checked in with that genre)
// ============================================================================

const INITIAL_GENRES = [
  { name: "Deep House", count: 24 },
  { name: "Tech House", count: 21 },
  { name: "Progressive House", count: 18 },
  { name: "EDM Pop", count: 15 },
  { name: "Future Bass", count: 14 },
  { name: "Melodic Techno", count: 12 },
  { name: "Bass House", count: 11 },
  { name: "Tropical House", count: 9 },
  { name: "Club Pop", count: 8 },
  { name: "Drum & Bass", count: 7 },
  { name: "Electro House", count: 6 },
  { name: "Afro House", count: 5 },
];

const INITIAL_CHECKINS = [
  {
    name: "Alex M.",
    genres: ["Deep House", "Tech House", "Melodic Techno"],
    time: 2,
  },
  {
    name: "Sarah K.",
    genres: ["EDM Pop", "Future Bass", "Tropical House"],
    time: 5,
  },
  {
    name: "Mike R.",
    genres: ["Tech House", "Bass House", "Drum & Bass"],
    time: 8,
  },
  {
    name: "Jessica L.",
    genres: ["Progressive House", "Deep House", "Club Pop"],
    time: 12,
  },
  {
    name: "Chris T.",
    genres: ["Melodic Techno", "Progressive House", "Afro House"],
    time: 15,
  },
];

const FAKE_NAMES = [
  "Jordan P.",
  "Taylor S.",
  "Morgan W.",
  "Casey B.",
  "Riley D.",
  "Quinn H.",
  "Avery N.",
  "Jamie F.",
  "Drew K.",
  "Peyton M.",
  "Skyler R.",
  "Reese T.",
  "Finley C.",
  "Parker J.",
  "Blake W.",
];

/**
 * AUTO-ANIMATION EXPLAINED:
 *
 * This hook simulates new people checking into the club.
 * Every 3 seconds, there's a 40% chance a new "person" checks in.
 *
 * When someone checks in:
 * 1. A random name is picked from FAKE_NAMES
 * 2. 3 random genres are assigned to them
 * 3. Each of those genres gets +1 to their count
 * 4. The genres are re-sorted by count (highest first)
 * 5. The new person appears in "Recent Check-ins"
 *
 * This creates a realistic live-updating effect for the demo.
 */
function useSimulatedCheckins(enabled) {
  const [genres, setGenres] = useState(INITIAL_GENRES);
  const [checkins, setCheckins] = useState(INITIAL_CHECKINS);
  const [totalCheckins, setTotalCheckins] = useState(47);

  useEffect(() => {
    if (!enabled) return;

    const genreNames = INITIAL_GENRES.map((g) => g.name);

    const interval = setInterval(() => {
      // 40% chance of new check-in each interval
      if (Math.random() > 0.6) {
        // Pick random name
        const newName =
          FAKE_NAMES[Math.floor(Math.random() * FAKE_NAMES.length)];

        // Pick 3 unique random genres for this person
        const shuffled = [...genreNames].sort(() => Math.random() - 0.5);
        const newGenres = shuffled.slice(0, 3);

        // Add to check-ins list (newest first, keep max 10)
        setCheckins((prev) => [
          {
            name: newName,
            genres: newGenres,
            time: 0,
          },
          ...prev.slice(0, 9),
        ]);

        // Increment total
        setTotalCheckins((prev) => prev + 1);

        // Update genre counts (+1 for each genre this person likes)
        setGenres((prev) => {
          const updated = prev.map((g) => {
            if (newGenres.includes(g.name)) {
              return { ...g, count: g.count + 1 };
            }
            return g;
          });
          // Re-sort by count (highest first)
          return updated.sort((a, b) => b.count - a.count);
        });
      }

      // Age the check-in times (for "X min ago" display)
      setCheckins((prev) => prev.map((c) => ({ ...c, time: c.time + 0.05 })));
    }, 3000); // Run every 3 seconds

    return () => clearInterval(interval);
  }, [enabled]);

  return { genres, checkins, totalCheckins };
}

// ============================================================================
// COMPONENTS
// ============================================================================

function LiveVideoFeed({ frameData, width, height }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    if (!frameData || !canvasRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
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

/**
 * FIXED HYPE GAUGE
 *
 * The issue was the arc path and stroke-dasharray calculation.
 * This version uses a proper SVG arc that goes from left to right
 * without any breaks.
 */
function HypeGauge({ value }) {
  // Clamp value between 0 and 1
  const clampedValue = Math.max(0, Math.min(1, value));

  // Arc parameters
  const radius = 70;
  const strokeWidth = 12;
  const centerX = 100;
  const centerY = 100;

  // Arc goes from 225° to -45° (270° sweep, bottom-left to bottom-right)
  const startAngle = 225 * (Math.PI / 180);
  const endAngle = -45 * (Math.PI / 180);
  const totalAngle = 270 * (Math.PI / 180);

  // Calculate arc length
  const circumference = 2 * Math.PI * radius;
  const arcLength = (totalAngle / (2 * Math.PI)) * circumference;

  // How much of the arc to fill based on value
  const fillLength = arcLength * clampedValue;

  // Calculate start and end points of the arc
  const startX = centerX + radius * Math.cos(startAngle);
  const startY = centerY - radius * Math.sin(startAngle);
  const endX = centerX + radius * Math.cos(endAngle);
  const endY = centerY - radius * Math.sin(endAngle);

  // SVG arc path (large arc, clockwise)
  const arcPath = `M ${startX} ${startY} A ${radius} ${radius} 0 1 1 ${endX} ${endY}`;

  // Needle rotation: 0% = -135°, 100% = 135°
  const needleRotation = -135 + clampedValue * 270;

  // Color: red (0) to green (120) based on value
  const hue = clampedValue * 120;
  const color = `hsl(${hue}, 100%, 50%)`;

  return (
    <div className="hype-gauge">
      <svg viewBox="0 0 200 200" className="gauge-svg">
        {/* Background arc (gray) */}
        <path
          d={arcPath}
          fill="none"
          stroke="rgba(255,255,255,0.1)"
          strokeWidth={strokeWidth}
          strokeLinecap="round"
        />

        {/* Filled arc (colored) */}
        <path
          d={arcPath}
          fill="none"
          stroke={color}
          strokeWidth={strokeWidth}
          strokeLinecap="round"
          strokeDasharray={`${fillLength} ${arcLength}`}
          style={{
            filter: `drop-shadow(0 0 8px ${color})`,
            transition: "stroke-dasharray 0.3s ease-out",
          }}
        />

        {/* Needle */}
        <g
          transform={`rotate(${needleRotation} ${centerX} ${centerY})`}
          style={{ transition: "transform 0.3s ease-out" }}
        >
          <line
            x1={centerX}
            y1={centerY}
            x2={centerX}
            y2={centerY - radius + 15}
            stroke="white"
            strokeWidth="3"
            strokeLinecap="round"
          />
          <circle cx={centerX} cy={centerY} r="8" fill="white" />
        </g>

        {/* Center text */}
        <text
          x={centerX}
          y={centerY + 35}
          textAnchor="middle"
          className="gauge-text"
        >
          {(clampedValue * 100).toFixed(0)}%
        </text>
        <text
          x={centerX}
          y={centerY + 55}
          textAnchor="middle"
          className="gauge-label"
        >
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

function GenreRankings({ genres, checkins, totalCheckins }) {
  const maxCount = genres[0]?.count || 1;

  return (
    <div className="genre-panel">
      <div className="genre-header">
        <div className="genre-title">🎵 CROWD MUSIC TASTE</div>
        <div className="checkin-count">{totalCheckins} guests</div>
      </div>

      <div className="genre-list">
        {genres.slice(0, 8).map((genre, i) => (
          <div
            key={genre.name}
            className={`genre-item ${i < 3 ? "top-three" : ""}`}
          >
            <span className={`genre-rank ${i < 3 ? "highlight" : ""}`}>
              #{i + 1}
            </span>
            <span className="genre-name">{genre.name}</span>
            <div className="genre-bar-track">
              <div
                className="genre-bar-fill"
                style={{ width: `${(genre.count / maxCount) * 100}%` }}
              />
            </div>
            <span className="genre-count">{genre.count} people</span>
          </div>
        ))}
      </div>

      <div className="recent-checkins">
        <div className="checkins-title">📱 Recent Check-ins</div>
        {checkins.slice(0, 4).map((c, i) => (
          <div key={i} className="checkin-item">
            <span className="checkin-name">{c.name}</span>
            <span className="checkin-time">
              {c.time < 1 ? "just now" : `${Math.floor(c.time)}m ago`}
            </span>
            <div className="checkin-genres">
              {c.genres.map((g) => (
                <span key={g} className="genre-tag">
                  {g}
                </span>
              ))}
            </div>
          </div>
        ))}
      </div>

      <div className="qr-prompt">
        <div className="qr-text">
          📲 Scan QR at entrance → Connect Spotify → Get FREE drink!
        </div>
      </div>
    </div>
  );
}

// ============================================================================
// MAIN APP
// ============================================================================

export default function App() {
  const [connected, setConnected] = useState(false);
  const [data, setData] = useState(null);

  // Simulated genre data (always runs for demo)
  const { genres, checkins, totalCheckins } = useSimulatedCheckins(true);

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
          {connected ? "LIVE" : "CAMERA OFFLINE"}
          {data && (
            <span className="latency">
              {data.latencyMs}ms | {data.fps} FPS
            </span>
          )}
        </div>
      </header>

      <main className="main">
        {/* LEFT COLUMN: Video + Genre List */}
        <div className="left-column">
          {/* Video Feed (smaller) */}
          {data ? (
            <LiveVideoFeed
              frameData={data.frame}
              width={data.frameWidth}
              height={data.frameHeight}
            />
          ) : (
            <div className="video-placeholder">
              <div className="placeholder-content">
                <div className="spinner" />
                <p>Connecting to camera...</p>
                <p className="hint">Run: python main.py</p>
              </div>
            </div>
          )}

          {/* Genre Rankings (under video) */}
          <GenreRankings
            genres={genres}
            checkins={checkins}
            totalCheckins={totalCheckins}
          />
        </div>

        {/* RIGHT COLUMN: Stats + Hype + Heatmap */}
        <div className="right-column">
          {/* Stats Row */}
          <div className="stats-row">
            <StatCard
              icon="👥"
              value={data?.peopleCount || 0}
              label="In Frame"
            />
            <StatCard
              icon="🙌"
              value={`${((data?.handsUpRatio || 0) * 100).toFixed(0)}%`}
              label="Hands Up"
            />
            <StatCard
              icon="⚡"
              value={(data?.meanEnergy || 0).toFixed(1)}
              label="Energy"
            />
          </div>

          {/* Hype Gauge */}
          <HypeGauge value={data?.hypeScore || 0} />

          {/* Energy Bar */}
          <EnergyBar value={data?.meanEnergy || 0} />

          {/* Heatmap */}
          {data && <Heatmap data={data.heatmap} zones={data.zones} />}
        </div>
      </main>
    </div>
  );
}
