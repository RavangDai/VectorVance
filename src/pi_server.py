"""
pi_server.py - VectorVance Web Dashboard Server
─────────────────────────────────────────────────
Runs as a background thread inside main.py.
Serves the live dashboard, MJPEG video stream, telemetry API,
and receives commands from the browser.

Routes:
  GET  /              → dashboard HTML
  GET  /video_feed    → MJPEG stream (multipart/x-mixed-replace)
  GET  /api/status    → telemetry JSON (polled every 500 ms by dashboard)
  POST /api/command   → send a command  {"action": "toggle_auto" | "emergency_stop" | "reset" | "set_speed", "value": ...}

API used by main.py:
  start_server(port)        → start Flask in background thread
  push_frame(frame)         → push latest debug frame (numpy BGR)
  push_telemetry(data)      → push latest telemetry dict
  get_pending_command()     → returns command dict or None
  clear_command()           → call after executing the command

Install Flask on Pi:
  pip install flask --break-system-packages
"""

import os
import cv2
import time
import threading
import json

# ── Thread-safe shared state ──────────────────────────────────────────────────

_frame_lock    = threading.Lock()
_latest_frame  = None          # latest numpy BGR frame from main loop

_telem_lock    = threading.Lock()
_telemetry     = {}            # latest telemetry dict from main loop

_cmd_lock      = threading.Lock()
_pending_cmd   = None          # command dict waiting to be executed

_manual_keys_lock = threading.Lock()
_manual_keys      = {"w": False, "a": False, "s": False, "d": False}


# ── Public API (called from main.py) ─────────────────────────────────────────

def push_frame(frame):
    """Push the latest processed frame. Called every loop iteration."""
    global _latest_frame
    with _frame_lock:
        _latest_frame = frame.copy() if frame is not None else None


def push_telemetry(data: dict):
    """Push the latest telemetry dict. Called every loop iteration."""
    global _telemetry
    with _telem_lock:
        _telemetry = data


def get_pending_command() -> dict | None:
    """Return the pending command dict (or None). Called by main loop."""
    with _cmd_lock:
        return _pending_cmd


def clear_command():
    """Clear the pending command after it has been executed."""
    global _pending_cmd
    with _cmd_lock:
        _pending_cmd = None


def get_manual_keys() -> dict:
    """Return current WASD key state for manual drive. Called every loop iteration."""
    with _manual_keys_lock:
        return dict(_manual_keys)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _encode_frame_jpeg(quality: int = 75) -> bytes | None:
    with _frame_lock:
        if _latest_frame is None:
            return None
        ret, buf = cv2.imencode(
            '.jpg', _latest_frame,
            [cv2.IMWRITE_JPEG_QUALITY, quality]
        )
        return buf.tobytes() if ret else None


def _mjpeg_generator():
    """Yields MJPEG boundary frames for the /video_feed route."""
    while True:
        jpeg = _encode_frame_jpeg()
        if jpeg:
            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n'
                + jpeg +
                b'\r\n'
            )
        else:
            time.sleep(0.05)


# ── Flask server ──────────────────────────────────────────────────────────────

def start_server(port: int = 5000) -> bool:
    """
    Start the Flask dashboard server in a daemon background thread.
    Returns True on success, False if Flask is not installed.
    """
    try:
        from flask import Flask, Response, request, jsonify, send_from_directory
    except ImportError:
        print("[WebServer] Flask not installed.")
        print("[WebServer]   pip install flask --break-system-packages")
        return False

    app = Flask(__name__)

    # Silence Flask request logs (main.py console stays clean)
    import logging
    logging.getLogger('werkzeug').setLevel(logging.ERROR)

    # ── Resolve template path ─────────────────────────────────────────
    _here = os.path.dirname(os.path.abspath(__file__))
    _template_dir = os.path.join(_here, 'webapp', 'templates')

    # ── Routes ───────────────────────────────────────────────────────

    @app.route('/')
    def index():
        return send_from_directory(_template_dir, 'dashboard.html')

    @app.route('/video_feed')
    def video_feed():
        return Response(
            _mjpeg_generator(),
            mimetype='multipart/x-mixed-replace; boundary=frame'
        )

    @app.route('/api/status')
    def api_status():
        with _telem_lock:
            data = dict(_telemetry)
        data['server_time'] = round(time.time(), 2)
        return jsonify(data)

    @app.route('/api/command', methods=['POST'])
    def api_command():
        global _pending_cmd
        body = request.get_json(silent=True)
        if not body:
            return jsonify({"error": "No JSON body"}), 400

        action = body.get("action")
        valid = {"toggle_auto", "emergency_stop", "reset", "set_speed",
                 "set_target_color", "manual_drive", "set_mode"}
        if action not in valid:
            return jsonify({"error": f"Unknown action '{action}'"}), 400

        # manual_drive is a continuous key-state update — bypass the one-shot cmd queue
        if action == "manual_drive":
            global _manual_keys
            with _manual_keys_lock:
                _manual_keys = body.get("keys", {"w": False, "a": False,
                                                  "s": False, "d": False})
            return jsonify({"status": "ok", "action": action})

        with _cmd_lock:
            _pending_cmd = body

        print(f"[WebServer] Command received: {body}")
        return jsonify({"status": "ok", "action": action})

    @app.route('/api/ping')
    def api_ping():
        return jsonify({"status": "alive", "time": time.time()})

    # ── Start in daemon thread ────────────────────────────────────────

    def _run():
        print(f"[WebServer] Dashboard running → http://<pi-ip>:{port}/")
        app.run(
            host='0.0.0.0',
            port=port,
            debug=False,
            use_reloader=False,
            threaded=True
        )

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return True
