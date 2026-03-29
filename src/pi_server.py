"""
pi_server.py - VectorVance Pi Command Server
─────────────────────────────────────────────
Runs on the Raspberry Pi. Receives route commands from the web dashboard
over WiFi and stores the current command for the driving loop to read.

INSTALL (on Pi):
  pip install flask --break-system-packages

USAGE:
  python pi_server.py              → starts on port 5000
  python pi_server.py --port 8080  → custom port

The web dashboard POSTs to:
  POST http://<pi-ip>:5000/command
  Body: {"action": "FOLLOW_COLOR", "color": "GREEN", "route": ["START","FORK","WP_L","DEST"]}

The driving loop reads:
  from pi_server import get_current_command, is_command_ready
  cmd = get_current_command()  → {"color": "GREEN", "route": [...]} or None
"""

import argparse
import threading
import json
import time

# Global command store (thread-safe)
_lock = threading.Lock()
_current_command = None
_command_timestamp = 0


def get_current_command():
    """Read the latest command from the web dashboard. Returns dict or None."""
    with _lock:
        return _current_command


def is_command_ready():
    """Check if a command has been received."""
    with _lock:
        return _current_command is not None


def get_command_age():
    """Seconds since last command was received."""
    with _lock:
        if _command_timestamp == 0:
            return float('inf')
        return time.time() - _command_timestamp


def clear_command():
    """Clear the current command (after it's been executed)."""
    global _current_command
    with _lock:
        _current_command = None


def _set_command(cmd):
    """Internal: store a new command."""
    global _current_command, _command_timestamp
    with _lock:
        _current_command = cmd
        _command_timestamp = time.time()


def start_server(port=5000):
    """Start the Flask server in a background thread."""
    try:
        from flask import Flask, request, jsonify
    except ImportError:
        print("Flask not installed. Run: pip install flask --break-system-packages")
        return False

    app = Flask(__name__)

    @app.route('/command', methods=['POST'])
    def receive_command():
        """Receive a route command from the web dashboard."""
        try:
            data = request.get_json()
            if not data:
                return jsonify({"error": "No JSON body"}), 400

            action = data.get("action")
            color = data.get("color")
            route = data.get("route", [])

            if action != "FOLLOW_COLOR" or color not in ("GREEN", "BLUE"):
                return jsonify({"error": "Invalid command"}), 400

            _set_command({
                "color": color,
                "route": route,
                "received_at": time.time()
            })

            print(f">>> RECEIVED: Follow {color} — Route: {' -> '.join(route)}")

            return jsonify({
                "status": "ok",
                "message": f"Following {color} path",
                "route": route
            })

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route('/status', methods=['GET'])
    def get_status():
        """Check current command status."""
        cmd = get_current_command()
        return jsonify({
            "has_command": cmd is not None,
            "command": cmd,
            "age_seconds": round(get_command_age(), 1)
        })

    @app.route('/clear', methods=['POST'])
    def clear():
        """Clear the current command."""
        clear_command()
        return jsonify({"status": "cleared"})

    @app.route('/', methods=['GET'])
    def index():
        """Health check."""
        return jsonify({
            "service": "VectorVance Pi Server",
            "status": "running",
            "has_command": is_command_ready()
        })

    # Disable Flask's default logging for cleaner output
    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.WARNING)

    def run():
        print(f"VectorVance Pi Server starting on port {port}...")
        print(f"Web dashboard should POST to: http://<this-pi-ip>:{port}/command")
        print(f"Check status: http://<this-pi-ip>:{port}/status")
        print()
        app.run(host='0.0.0.0', port=port, debug=False)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return True


# ─────────────────────────────────────────────────────────────────────
#  STANDALONE MODE (run directly to test)
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='VectorVance Pi Server')
    parser.add_argument('--port', type=int, default=5000, help='Port (default: 5000)')
    args = parser.parse_args()

    try:
        from flask import Flask
    except ImportError:
        print("Flask not installed!")
        print("Run: pip install flask --break-system-packages")
        exit(1)

    # In standalone mode, run in foreground
    from flask import Flask, request, jsonify

    app = Flask(__name__)

    @app.route('/command', methods=['POST'])
    def receive_command():
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON body"}), 400

        color = data.get("color")
        route = data.get("route", [])

        _set_command({"color": color, "route": route, "received_at": time.time()})
        print(f"\n>>> COMMAND: Follow {color}")
        print(f"    Route: {' -> '.join(route)}")

        return jsonify({"status": "ok", "message": f"Following {color}"})

    @app.route('/status', methods=['GET'])
    def status():
        cmd = get_current_command()
        return jsonify({"has_command": cmd is not None, "command": cmd})

    @app.route('/clear', methods=['POST'])
    def clear():
        clear_command()
        return jsonify({"status": "cleared"})

    @app.route('/', methods=['GET'])
    def index():
        return jsonify({"service": "VectorVance Pi Server", "status": "running"})

    print(f"\nVectorVance Pi Server")
    print(f"Port: {args.port}")
    print(f"Waiting for commands from web dashboard...")
    print(f"Test: curl http://localhost:{args.port}/status\n")

    app.run(host='0.0.0.0', port=args.port, debug=False)
