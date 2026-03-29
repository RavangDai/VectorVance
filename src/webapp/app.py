"""
app.py - VectorVance Web Dashboard Server
──────────────────────────────────────────
A real Flask website that serves the navigation dashboard.
Run it on your laptop and open it in any browser.

INSTALL:
  pip install flask

USAGE:
  cd webapp
  python app.py                    → http://localhost:3000
  python app.py --port 8080        → http://localhost:8080

Then open the URL in Chrome/Edge/Firefox.
"""

from flask import Flask, render_template, request, jsonify
import argparse
import json
import time
import heapq

app = Flask(__name__)

# ─────────────────────────────────────────────────────────────────────
#  IN-MEMORY STATE
# ─────────────────────────────────────────────────────────────────────
track_state = {
    "nodes": [
        {"id": "START", "x": 300, "y": 520, "type": "start"},
        {"id": "FORK",  "x": 300, "y": 340, "type": "fork"},
        {"id": "WP_L",  "x": 130, "y": 180, "type": "waypoint"},
        {"id": "WP_R",  "x": 480, "y": 220, "type": "waypoint"},
        {"id": "DEST",  "x": 130, "y": 60,  "type": "destination"},
    ],
    "edges": [
        {"from": "START", "to": "FORK", "dist": 30, "sign": None},
        {"from": "FORK",  "to": "WP_L", "dist": 25, "sign": "GREEN"},
        {"from": "FORK",  "to": "WP_R", "dist": 35, "sign": "BLUE"},
        {"from": "WP_L",  "to": "DEST", "dist": 20, "sign": None},
        {"from": "WP_R",  "to": "DEST", "dist": 35, "sign": None},
    ],
    "pi_ip": "192.168.1.100",
    "pi_port": "5000",
}

last_command_sent = {"command": None, "timestamp": 0}


# ─────────────────────────────────────────────────────────────────────
#  DIJKSTRA (server-side)
# ─────────────────────────────────────────────────────────────────────
def dijkstra(nodes, edges, start_id, dest_id):
    adj = {n["id"]: [] for n in nodes}
    for e in edges:
        adj[e["from"]].append({"to": e["to"], "dist": e["dist"], "sign": e.get("sign")})
        adj[e["to"]].append({"to": e["from"], "dist": e["dist"], "sign": e.get("sign")})

    dist = {n["id"]: float("inf") for n in nodes}
    dist[start_id] = 0
    prev = {n["id"]: None for n in nodes}
    visited = set()
    pq = [(0, start_id)]

    while pq:
        d, u = heapq.heappop(pq)
        if u in visited:
            continue
        visited.add(u)
        if u == dest_id:
            break
        for nb in adj.get(u, []):
            nd = d + nb["dist"]
            if nd < dist[nb["to"]]:
                dist[nb["to"]] = nd
                prev[nb["to"]] = u
                heapq.heappush(pq, (nd, nb["to"]))

    if dist[dest_id] == float("inf"):
        return None

    path = []
    cur = dest_id
    while cur:
        path.append(cur)
        cur = prev[cur]
    path.reverse()

    path_edges = []
    for i in range(len(path) - 1):
        for e in edges:
            if (e["from"] == path[i] and e["to"] == path[i+1]) or \
               (e["to"] == path[i] and e["from"] == path[i+1]):
                path_edges.append(e)
                break

    return {"path": path, "edges": path_edges, "distance": dist[dest_id]}


def all_routes(nodes, edges, start_id, dest_id):
    adj = {n["id"]: [] for n in nodes}
    for e in edges:
        adj[e["from"]].append({"to": e["to"], "dist": e["dist"], "sign": e.get("sign"), "edge": e})
        adj[e["to"]].append({"to": e["from"], "dist": e["dist"], "sign": e.get("sign"), "edge": e})

    results = []

    def dfs(cur, visited, path, edge_path, total):
        if cur == dest_id:
            results.append({"path": list(path), "edges": list(edge_path), "distance": total})
            return
        for nb in adj.get(cur, []):
            if nb["to"] not in visited:
                visited.add(nb["to"])
                path.append(nb["to"])
                edge_path.append(nb["edge"])
                dfs(nb["to"], visited, path, edge_path, total + nb["dist"])
                path.pop()
                edge_path.pop()
                visited.discard(nb["to"])

    dfs(start_id, {start_id}, [start_id], [], 0)
    results.sort(key=lambda r: r["distance"])
    return results


# ─────────────────────────────────────────────────────────────────────
#  ROUTES
# ─────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("dashboard.html")


@app.route("/api/track", methods=["GET"])
def get_track():
    """Return current track state + computed routes."""
    nodes = track_state["nodes"]
    edges = track_state["edges"]

    start = next((n for n in nodes if n["type"] == "start"), None)
    dest = next((n for n in nodes if n["type"] == "destination"), None)

    shortest = None
    routes = []
    chosen_sign = None

    if start and dest:
        shortest = dijkstra(nodes, edges, start["id"], dest["id"])
        routes = all_routes(nodes, edges, start["id"], dest["id"])
        if shortest:
            fork_edge = next((e for e in shortest["edges"] if e.get("sign")), None)
            chosen_sign = fork_edge["sign"] if fork_edge else None

    return jsonify({
        "nodes": nodes,
        "edges": edges,
        "shortest": shortest,
        "routes": routes,
        "chosen_sign": chosen_sign,
        "pi_ip": track_state["pi_ip"],
        "pi_port": track_state["pi_port"],
    })


@app.route("/api/track", methods=["POST"])
def update_track():
    """Update track layout (nodes, edges, pi settings)."""
    data = request.get_json()
    if "nodes" in data:
        track_state["nodes"] = data["nodes"]
    if "edges" in data:
        track_state["edges"] = data["edges"]
    if "pi_ip" in data:
        track_state["pi_ip"] = data["pi_ip"]
    if "pi_port" in data:
        track_state["pi_port"] = data["pi_port"]
    return jsonify({"status": "ok"})


@app.route("/api/send", methods=["POST"])
def send_to_pi():
    """Forward a command to the Pi's Flask server."""
    import urllib.request
    import urllib.error

    data = request.get_json()
    color = data.get("color")
    route = data.get("route", [])

    if not color:
        return jsonify({"error": "No color specified"}), 400

    pi_url = f"http://{track_state['pi_ip']}:{track_state['pi_port']}/command"
    payload = json.dumps({
        "action": "FOLLOW_COLOR",
        "color": color,
        "route": route
    }).encode("utf-8")

    try:
        req = urllib.request.Request(
            pi_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            result = json.loads(resp.read().decode())

        last_command_sent["command"] = {"color": color, "route": route}
        last_command_sent["timestamp"] = time.time()

        return jsonify({"status": "sent", "pi_response": result})

    except urllib.error.URLError as e:
        return jsonify({"status": "error", "message": f"Cannot reach Pi: {e}"}), 502
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VectorVance Dashboard")
    parser.add_argument("--port", type=int, default=3000)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    print(f"\n  VectorVance Web Dashboard")
    print(f"  Open: http://localhost:{args.port}")
    print(f"  Pi target: {track_state['pi_ip']}:{track_state['pi_port']}\n")

    app.run(host="0.0.0.0", port=args.port, debug=args.debug)