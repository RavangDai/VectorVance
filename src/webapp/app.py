"""
app.py — VectorVance Smart Intersection Dashboard
══════════════════════════════════════════════════
USAGE:  python app.py          →  http://<pi-ip>:3000

Car endpoints:
  POST /api/fork_detected   {"colors": ["red","green","blue"]}
  GET  /api/poll_command     → {"command": "green"} or {"command": null}
  POST /api/complete         → marks route done

Dashboard endpoints:
  GET  /api/state            → full current state
  POST /api/execute_route    {"color": "green"}
  POST /api/reset
"""

from flask import Flask, render_template, jsonify, request
import time

app = Flask(__name__)

# ── Track database: the AI "brain" ────────────────────────────────────────────
TRACK_DB = {
    "red":   {"label": "Red Path",   "type": "dead_end", "distance": 15,
              "description": "Dead End",    "emoji": "🔴"},
    "green": {"label": "Green Path", "type": "optimal",  "distance": 40,
              "description": "Short Route", "emoji": "🟢"},
    "blue":  {"label": "Blue Path",  "type": "long_way", "distance": 95,
              "description": "Long Route",  "emoji": "🔵"},
}

# ── Shared state ───────────────────────────────────────────────────────────────
_state = {
    "status":            "driving",   # driving | fork_detected | executing | done
    "detected_colors":   [],
    "recommended_color": None,
    "chosen_color":      None,
    "command":           None,
    "command_ts":        0,
    "log":               [],
}


def _recommend(colors):
    """Pick the shortest non-dead-end path from the detected colors."""
    candidates = [
        (c, TRACK_DB[c]) for c in colors
        if c in TRACK_DB and TRACK_DB[c]["type"] != "dead_end"
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda x: x[1]["distance"])[0]


def _log(msg):
    _state["log"].insert(0, {"time": time.strftime("%H:%M:%S"), "msg": msg})
    _state["log"] = _state["log"][:30]


# ── Dashboard routes ───────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("dashboard.html")


@app.route("/api/state")
def get_state():
    paths = {
        c: TRACK_DB[c]
        for c in _state["detected_colors"]
        if c in TRACK_DB
    }
    return jsonify({
        "status":            _state["status"],
        "detected_colors":   _state["detected_colors"],
        "paths":             paths,
        "recommended_color": _state["recommended_color"],
        "chosen_color":      _state["chosen_color"],
        "log":               _state["log"],
    })


@app.route("/api/execute_route", methods=["POST"])
def execute_route():
    color = request.get_json(force=True).get("color", "").lower()
    if color not in TRACK_DB:
        return jsonify({"error": "invalid color"}), 400
    _state["chosen_color"] = color
    _state["command"]      = color
    _state["command_ts"]   = time.time()
    _state["status"]       = "executing"
    info = TRACK_DB[color]
    _log(f"EXECUTE → {color.upper()} ({info['description']}, {info['distance']} cm)")
    return jsonify({"ok": True})


@app.route("/api/reset", methods=["POST"])
def reset():
    _state.update({
        "status":            "driving",
        "detected_colors":   [],
        "recommended_color": None,
        "chosen_color":      None,
        "command":           None,
        "command_ts":        0,
    })
    _log("System reset — VectorVance driving")
    return jsonify({"ok": True})


# ── Car (Pi) endpoints ─────────────────────────────────────────────────────────

@app.route("/api/fork_detected", methods=["POST"])
def fork_detected():
    data   = request.get_json(force=True)
    colors = [c.lower() for c in data.get("colors", []) if c.lower() in TRACK_DB]
    if not colors:
        return jsonify({"error": "no valid colors"}), 400

    _state["status"]            = "fork_detected"
    _state["detected_colors"]   = colors
    _state["recommended_color"] = _recommend(colors)
    _state["chosen_color"]      = None
    _state["command"]           = None

    _log(f"Fork detected: {', '.join(c.upper() for c in colors)}")
    rec = _state["recommended_color"]
    if rec:
        _log(f"AI recommends: {rec.upper()} ({TRACK_DB[rec]['description']})")
    return jsonify({"ok": True, "recommended": rec})


@app.route("/api/poll_command")
def poll_command():
    """Car polls this every loop to check for a pending route command."""
    cmd = _state["command"]
    if cmd and (time.time() - _state["command_ts"]) < 30:
        return jsonify({"command": cmd})
    return jsonify({"command": None})


@app.route("/api/complete", methods=["POST"])
def complete():
    """Car calls this when it finishes executing the route."""
    chosen = _state.get("chosen_color", "?")
    _state["status"]  = "done"
    _state["command"] = None
    _log(f"Route complete: {chosen.upper()}")
    return jsonify({"ok": True})


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=3000)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    print(f"VectorVance dashboard: http://0.0.0.0:{args.port}")
    app.run(host="0.0.0.0", port=args.port, debug=args.debug)
