import { useState, useCallback, useRef, useEffect } from "react";

const COLORS = {
  bg: "#0a0e17",
  surface: "#111827",
  surfaceLight: "#1a2332",
  border: "#1e2d3d",
  borderActive: "#2563eb",
  text: "#e2e8f0",
  textMuted: "#64748b",
  green: "#22c55e",
  greenDark: "#15803d",
  greenGlow: "rgba(34,197,94,0.15)",
  blue: "#3b82f6",
  blueDark: "#1d4ed8",
  blueGlow: "rgba(59,130,246,0.15)",
  amber: "#f59e0b",
  red: "#ef4444",
  purple: "#a855f7",
  teal: "#14b8a6",
};

const SIGN_COLORS = [
  { id: "GREEN", color: COLORS.green, glow: COLORS.greenGlow, label: "Green" },
  { id: "BLUE", color: COLORS.blue, glow: COLORS.blueGlow, label: "Blue" },
];

const DEFAULT_NODES = [
  { id: "START", x: 300, y: 520, type: "start" },
  { id: "FORK", x: 300, y: 340, type: "fork" },
  { id: "WP_L", x: 130, y: 180, type: "waypoint" },
  { id: "WP_R", x: 480, y: 220, type: "waypoint" },
  { id: "DEST", x: 130, y: 60, type: "destination" },
];

const DEFAULT_EDGES = [
  { from: "START", to: "FORK", dist: 30, sign: null },
  { from: "FORK", to: "WP_L", dist: 25, sign: "GREEN" },
  { from: "FORK", to: "WP_R", dist: 35, sign: "BLUE" },
  { from: "WP_L", to: "DEST", dist: 20, sign: null },
  { from: "WP_R", to: "DEST", dist: 35, sign: null },
];

function dijkstra(nodes, edges, startId, destId) {
  const adj = {};
  nodes.forEach((n) => (adj[n.id] = []));
  edges.forEach((e) => {
    adj[e.from]?.push({ to: e.to, dist: e.dist, sign: e.sign });
    adj[e.to]?.push({ to: e.from, dist: e.dist, sign: e.sign });
  });

  const dist = {};
  const prev = {};
  const visited = new Set();
  nodes.forEach((n) => (dist[n.id] = Infinity));
  dist[startId] = 0;

  while (true) {
    let u = null;
    let best = Infinity;
    for (const id in dist) {
      if (!visited.has(id) && dist[id] < best) {
        best = dist[id];
        u = id;
      }
    }
    if (!u || u === destId) break;
    visited.add(u);

    for (const { to, dist: d } of adj[u]) {
      if (!visited.has(to) && dist[u] + d < dist[to]) {
        dist[to] = dist[u] + d;
        prev[to] = u;
      }
    }
  }

  if (dist[destId] === Infinity) return null;

  const path = [];
  let cur = destId;
  while (cur) {
    path.unshift(cur);
    cur = prev[cur];
  }

  const pathEdges = [];
  for (let i = 0; i < path.length - 1; i++) {
    const e = edges.find(
      (e) =>
        (e.from === path[i] && e.to === path[i + 1]) ||
        (e.to === path[i] && e.from === path[i + 1])
    );
    if (e) pathEdges.push(e);
  }

  return { path, edges: pathEdges, distance: dist[destId] };
}

function allRoutes(nodes, edges, startId, destId) {
  const adj = {};
  nodes.forEach((n) => (adj[n.id] = []));
  edges.forEach((e) => {
    adj[e.from]?.push({ to: e.to, dist: e.dist, sign: e.sign, edge: e });
    adj[e.to]?.push({ to: e.from, dist: e.dist, sign: e.sign, edge: e });
  });

  const results = [];
  const dfs = (cur, visited, path, edgePath, totalDist) => {
    if (cur === destId) {
      results.push({ path: [...path], edges: [...edgePath], distance: totalDist });
      return;
    }
    for (const { to, dist: d, edge } of adj[cur]) {
      if (!visited.has(to)) {
        visited.add(to);
        path.push(to);
        edgePath.push(edge);
        dfs(to, visited, path, edgePath, totalDist + d);
        path.pop();
        edgePath.pop();
        visited.delete(to);
      }
    }
  };

  const v = new Set([startId]);
  dfs(startId, v, [startId], [], 0);
  results.sort((a, b) => a.distance - b.distance);
  return results;
}

function NodeIcon({ type, x, y, selected, onClick }) {
  const r = type === "fork" ? 18 : 16;
  const fill =
    type === "start" ? COLORS.green
    : type === "destination" ? COLORS.purple
    : type === "fork" ? COLORS.amber
    : COLORS.textMuted;

  return (
    <g onClick={onClick} style={{ cursor: "pointer" }}>
      <circle cx={x} cy={y} r={r + 6} fill={fill} opacity={selected ? 0.25 : 0} />
      <circle cx={x} cy={y} r={r} fill={COLORS.surface} stroke={fill} strokeWidth={selected ? 2.5 : 1.5} />
      <text x={x} y={y + 1} textAnchor="middle" dominantBaseline="central" fill={fill} fontSize="11" fontWeight="600" fontFamily="monospace">
        {type === "start" ? "A" : type === "destination" ? "B" : type === "fork" ? "F" : "W"}
      </text>
    </g>
  );
}

function EdgeLine({ edge, nodes, isOnPath, signColor }) {
  const fromN = nodes.find((n) => n.id === edge.from);
  const toN = nodes.find((n) => n.id === edge.to);
  if (!fromN || !toN) return null;

  const stroke = isOnPath ? (signColor || COLORS.green) : COLORS.border;
  const width = isOnPath ? 3 : 1.5;
  const opacity = isOnPath ? 0.9 : 0.4;
  const mx = (fromN.x + toN.x) / 2;
  const my = (fromN.y + toN.y) / 2;

  return (
    <g>
      <line x1={fromN.x} y1={fromN.y} x2={toN.x} y2={toN.y} stroke={stroke} strokeWidth={width} opacity={opacity} strokeLinecap="round" />
      <rect x={mx - 18} y={my - 10} width={36} height={20} rx={4} fill={COLORS.surface} stroke={stroke} strokeWidth={0.5} opacity={0.9} />
      <text x={mx} y={my + 1} textAnchor="middle" dominantBaseline="central" fill={isOnPath ? stroke : COLORS.textMuted} fontSize="10" fontFamily="monospace" fontWeight="500">
        {edge.dist}cm
      </text>
      {edge.sign && (
        <rect
          x={mx - 6}
          y={my - 22}
          width={12}
          height={8}
          rx={2}
          fill={edge.sign === "GREEN" ? COLORS.green : COLORS.blue}
          opacity={0.9}
        />
      )}
    </g>
  );
}

export default function VectorVanceDashboard() {
  const [nodes, setNodes] = useState(DEFAULT_NODES);
  const [edges, setEdges] = useState(DEFAULT_EDGES);
  const [piIp, setPiIp] = useState("192.168.1.100");
  const [piPort, setPiPort] = useState("5000");
  const [sendStatus, setSendStatus] = useState(null);
  const [selectedNode, setSelectedNode] = useState(null);
  const [editingEdge, setEditingEdge] = useState(null);
  const svgRef = useRef(null);

  const startNode = nodes.find((n) => n.type === "start");
  const destNode = nodes.find((n) => n.type === "destination");

  const shortest = startNode && destNode ? dijkstra(nodes, edges, startNode.id, destNode.id) : null;
  const routes = startNode && destNode ? allRoutes(nodes, edges, startNode.id, destNode.id) : [];

  const forkEdge = shortest?.edges.find((e) => e.sign);
  const chosenSign = forkEdge?.sign || null;
  const chosenColor = chosenSign === "GREEN" ? COLORS.green : chosenSign === "BLUE" ? COLORS.blue : null;

  const sendToPi = useCallback(async () => {
    if (!chosenSign) return;
    setSendStatus("sending");
    try {
      const res = await fetch(`http://${piIp}:${piPort}/command`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action: "FOLLOW_COLOR", color: chosenSign, route: shortest?.path }),
      });
      if (res.ok) setSendStatus("sent");
      else setSendStatus("error");
    } catch {
      setSendStatus("error");
    }
    setTimeout(() => setSendStatus(null), 3000);
  }, [chosenSign, piIp, piPort, shortest]);

  const updateEdgeDist = (idx, val) => {
    const next = [...edges];
    next[idx] = { ...next[idx], dist: Math.max(1, parseInt(val) || 1) };
    setEdges(next);
  };

  const updateEdgeSign = (idx, sign) => {
    const next = [...edges];
    next[idx] = { ...next[idx], sign: next[idx].sign === sign ? null : sign };
    setEdges(next);
  };

  const handleSvgMouseDown = (e) => {
    if (!selectedNode) return;
    const svg = svgRef.current;
    const pt = svg.createSVGPoint();
    pt.x = e.clientX;
    pt.y = e.clientY;
    const svgP = pt.matrixTransform(svg.getScreenCTM().inverse());

    const onMove = (ev) => {
      pt.x = ev.clientX;
      pt.y = ev.clientY;
      const p = pt.matrixTransform(svg.getScreenCTM().inverse());
      setNodes((prev) =>
        prev.map((n) => (n.id === selectedNode ? { ...n, x: Math.round(p.x), y: Math.round(p.y) } : n))
      );
    };
    const onUp = () => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
    };
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
  };

  return (
    <div style={{ background: COLORS.bg, color: COLORS.text, fontFamily: "'JetBrains Mono', 'SF Mono', 'Fira Code', monospace", minHeight: "100vh", padding: "0" }}>
      <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet" />

      <div style={{ borderBottom: `1px solid ${COLORS.border}`, padding: "16px 24px", display: "flex", alignItems: "center", gap: 16 }}>
        <div style={{ width: 8, height: 8, borderRadius: "50%", background: chosenColor || COLORS.textMuted, boxShadow: chosenColor ? `0 0 8px ${chosenColor}` : "none" }} />
        <span style={{ fontSize: 15, fontWeight: 700, letterSpacing: "0.05em" }}>VECTORVANCE</span>
        <span style={{ fontSize: 11, color: COLORS.textMuted, letterSpacing: "0.1em" }}>NAV CONTROL</span>
        <div style={{ flex: 1 }} />
        {chosenSign && (
          <div style={{ display: "flex", alignItems: "center", gap: 8, padding: "6px 14px", borderRadius: 6, background: chosenSign === "GREEN" ? COLORS.greenGlow : COLORS.blueGlow, border: `1px solid ${chosenColor}40` }}>
            <div style={{ width: 10, height: 10, borderRadius: 2, background: chosenColor }} />
            <span style={{ fontSize: 12, fontWeight: 600, color: chosenColor }}>
              FOLLOW {chosenSign}
            </span>
          </div>
        )}
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 320px", minHeight: "calc(100vh - 53px)" }}>
        <div style={{ padding: 24, borderRight: `1px solid ${COLORS.border}` }}>
          <div style={{ fontSize: 11, color: COLORS.textMuted, marginBottom: 12, letterSpacing: "0.08em" }}>
            TRACK MAP — drag nodes to reposition
          </div>

          <div style={{ background: COLORS.surface, borderRadius: 10, border: `1px solid ${COLORS.border}`, overflow: "hidden" }}>
            <svg ref={svgRef} viewBox="0 0 620 600" width="100%" style={{ display: "block" }} onMouseDown={handleSvgMouseDown}>
              <rect width="620" height="600" fill={COLORS.surface} />

              <defs>
                <pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse">
                  <path d="M 40 0 L 0 0 0 40" fill="none" stroke={COLORS.border} strokeWidth="0.3" opacity="0.5" />
                </pattern>
              </defs>
              <rect width="620" height="600" fill="url(#grid)" />

              {edges.map((e, i) => {
                const onPath = shortest?.edges.includes(e);
                const sc = e.sign === "GREEN" ? COLORS.green : e.sign === "BLUE" ? COLORS.blue : null;
                return <EdgeLine key={i} edge={e} nodes={nodes} isOnPath={onPath} signColor={onPath ? sc || COLORS.green : null} />;
              })}

              {nodes.map((n) => (
                <NodeIcon
                  key={n.id}
                  type={n.type}
                  x={n.x}
                  y={n.y}
                  selected={selectedNode === n.id}
                  onClick={() => setSelectedNode(selectedNode === n.id ? null : n.id)}
                />
              ))}

              {nodes.map((n) => (
                <text key={n.id + "_label"} x={n.x} y={n.y - 26} textAnchor="middle" fill={COLORS.textMuted} fontSize="9" fontFamily="monospace" letterSpacing="0.05em">
                  {n.id}
                </text>
              ))}

              {shortest && (
                <g>
                  <rect x={10} y={555} width={200} height={36} rx={6} fill={COLORS.bg} stroke={chosenColor || COLORS.border} strokeWidth={0.5} opacity={0.95} />
                  <text x={20} y={577} fill={chosenColor || COLORS.green} fontSize="12" fontFamily="monospace" fontWeight="600">
                    Shortest: {shortest.distance}cm
                  </text>
                  <text x={145} y={577} fill={COLORS.textMuted} fontSize="10" fontFamily="monospace">
                    {shortest.path.join("→")}
                  </text>
                </g>
              )}
            </svg>
          </div>
        </div>

        <div style={{ padding: 20, display: "flex", flexDirection: "column", gap: 16, overflowY: "auto" }}>
          <div style={{ background: COLORS.surface, borderRadius: 8, border: `1px solid ${COLORS.border}`, padding: 16 }}>
            <div style={{ fontSize: 11, color: COLORS.textMuted, marginBottom: 12, letterSpacing: "0.08em" }}>EDGES</div>
            {edges.map((e, i) => (
              <div key={i} style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8, padding: "6px 0", borderBottom: i < edges.length - 1 ? `1px solid ${COLORS.border}` : "none" }}>
                <span style={{ fontSize: 10, color: COLORS.textMuted, width: 90, flexShrink: 0 }}>
                  {e.from}→{e.to}
                </span>
                <input
                  type="number"
                  value={e.dist}
                  onChange={(ev) => updateEdgeDist(i, ev.target.value)}
                  style={{ width: 50, background: COLORS.bg, border: `1px solid ${COLORS.border}`, borderRadius: 4, color: COLORS.text, padding: "3px 6px", fontSize: 12, fontFamily: "monospace", textAlign: "center" }}
                />
                <span style={{ fontSize: 9, color: COLORS.textMuted }}>cm</span>
                <div style={{ flex: 1 }} />
                {SIGN_COLORS.map((s) => (
                  <button
                    key={s.id}
                    onClick={() => updateEdgeSign(i, s.id)}
                    style={{
                      width: 20,
                      height: 14,
                      borderRadius: 3,
                      border: `1.5px solid ${e.sign === s.id ? s.color : COLORS.border}`,
                      background: e.sign === s.id ? s.color : "transparent",
                      cursor: "pointer",
                      opacity: e.sign === s.id ? 1 : 0.4,
                    }}
                    title={`Assign ${s.label} sign`}
                  />
                ))}
              </div>
            ))}
          </div>

          <div style={{ background: COLORS.surface, borderRadius: 8, border: `1px solid ${COLORS.border}`, padding: 16 }}>
            <div style={{ fontSize: 11, color: COLORS.textMuted, marginBottom: 12, letterSpacing: "0.08em" }}>ALL ROUTES</div>
            {routes.map((r, i) => {
              const isShortest = i === 0;
              const routeSign = r.edges.find((e) => e.sign)?.sign;
              const rc = routeSign === "GREEN" ? COLORS.green : routeSign === "BLUE" ? COLORS.blue : COLORS.textMuted;
              return (
                <div
                  key={i}
                  style={{
                    padding: "8px 10px",
                    marginBottom: 6,
                    borderRadius: 6,
                    background: isShortest ? `${rc}15` : "transparent",
                    border: `1px solid ${isShortest ? rc + "40" : COLORS.border}`,
                  }}
                >
                  <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                    <span style={{ fontSize: 13, fontWeight: 600, color: isShortest ? rc : COLORS.text }}>
                      {r.distance}cm
                    </span>
                    {isShortest && (
                      <span style={{ fontSize: 9, background: rc, color: COLORS.bg, padding: "1px 6px", borderRadius: 3, fontWeight: 700 }}>
                        SHORTEST
                      </span>
                    )}
                    {routeSign && (
                      <div style={{ width: 10, height: 10, borderRadius: 2, background: rc, marginLeft: "auto" }} />
                    )}
                  </div>
                  <div style={{ fontSize: 10, color: COLORS.textMuted, marginTop: 4 }}>
                    {r.path.join(" → ")}
                  </div>
                </div>
              );
            })}
            {routes.length === 0 && (
              <div style={{ fontSize: 11, color: COLORS.textMuted }}>No routes found</div>
            )}
          </div>

          <div style={{ background: COLORS.surface, borderRadius: 8, border: `1px solid ${COLORS.border}`, padding: 16 }}>
            <div style={{ fontSize: 11, color: COLORS.textMuted, marginBottom: 12, letterSpacing: "0.08em" }}>
              SEND TO PI
            </div>
            <div style={{ display: "flex", gap: 6, marginBottom: 10 }}>
              <input
                value={piIp}
                onChange={(e) => setPiIp(e.target.value)}
                placeholder="Pi IP"
                style={{ flex: 1, background: COLORS.bg, border: `1px solid ${COLORS.border}`, borderRadius: 4, color: COLORS.text, padding: "6px 8px", fontSize: 12, fontFamily: "monospace" }}
              />
              <input
                value={piPort}
                onChange={(e) => setPiPort(e.target.value)}
                placeholder="Port"
                style={{ width: 55, background: COLORS.bg, border: `1px solid ${COLORS.border}`, borderRadius: 4, color: COLORS.text, padding: "6px 8px", fontSize: 12, fontFamily: "monospace", textAlign: "center" }}
              />
            </div>
            <button
              onClick={sendToPi}
              disabled={!chosenSign}
              style={{
                width: "100%",
                padding: "10px",
                borderRadius: 6,
                border: `1.5px solid ${chosenColor || COLORS.border}`,
                background: chosenColor ? `${chosenColor}20` : "transparent",
                color: chosenColor || COLORS.textMuted,
                fontFamily: "monospace",
                fontSize: 13,
                fontWeight: 700,
                cursor: chosenSign ? "pointer" : "default",
                letterSpacing: "0.05em",
                opacity: chosenSign ? 1 : 0.4,
              }}
            >
              {sendStatus === "sending" ? "SENDING..." : sendStatus === "sent" ? "SENT" : sendStatus === "error" ? "FAILED — RETRY" : chosenSign ? `SEND: FOLLOW ${chosenSign}` : "NO ROUTE"}
            </button>
            {chosenSign && (
              <div style={{ fontSize: 10, color: COLORS.textMuted, marginTop: 8, textAlign: "center" }}>
                POST http://{piIp}:{piPort}/command
              </div>
            )}
          </div>

          <div style={{ background: COLORS.surface, borderRadius: 8, border: `1px solid ${COLORS.border}`, padding: 16 }}>
            <div style={{ fontSize: 11, color: COLORS.textMuted, marginBottom: 8, letterSpacing: "0.08em" }}>LEGEND</div>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 6 }}>
              {[
                { color: COLORS.green, label: "Start" },
                { color: COLORS.purple, label: "Destination" },
                { color: COLORS.amber, label: "Fork" },
                { color: COLORS.textMuted, label: "Waypoint" },
              ].map(({ color, label }) => (
                <div key={label} style={{ display: "flex", alignItems: "center", gap: 6 }}>
                  <div style={{ width: 8, height: 8, borderRadius: "50%", background: color }} />
                  <span style={{ fontSize: 10, color: COLORS.textMuted }}>{label}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
