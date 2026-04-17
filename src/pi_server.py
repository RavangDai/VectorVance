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

_rear_frame_lock   = threading.Lock()
_latest_rear_frame = None      # latest numpy BGR frame from rear ELP camera

_telem_lock    = threading.Lock()
_telemetry     = {}            # latest telemetry dict from main loop

_cmd_lock      = threading.Lock()
_pending_cmd   = None          # command dict waiting to be executed

_manual_keys_lock = threading.Lock()
_manual_keys      = {"w": False, "a": False, "s": False, "d": False}

_sentry_events_lock = threading.Lock()
_sentry_events      = []       # list of event dicts from SentryMonitor


# ── Public API (called from main.py) ─────────────────────────────────────────

def push_frame(frame):
    """Push the latest processed frame. Called every loop iteration."""
    global _latest_frame
    with _frame_lock:
        _latest_frame = frame.copy() if frame is not None else None


def push_rear_frame(frame):
    """Push the latest rear camera frame. Called from the rear camera thread."""
    global _latest_rear_frame
    with _rear_frame_lock:
        _latest_rear_frame = frame.copy() if frame is not None else None


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


def push_sentry_events(events: list):
    """Push latest sentry event list. Called from main loop when in SENTRY mode."""
    global _sentry_events
    with _sentry_events_lock:
        _sentry_events = list(events)


def clear_sentry_events():
    """Clear the sentry event log. Called when user requests clear."""
    global _sentry_events
    with _sentry_events_lock:
        _sentry_events = []


# ── Internal helpers ──────────────────────────────────────────────────────────

def _encode_frame_jpeg(quality: int = 55) -> bytes | None:
    with _frame_lock:
        if _latest_frame is None:
            return None
        ret, buf = cv2.imencode(
            '.jpg', _latest_frame,
            [cv2.IMWRITE_JPEG_QUALITY, quality]
        )
        return buf.tobytes() if ret else None


def _encode_rear_frame_jpeg(quality: int = 70) -> bytes | None:
    with _rear_frame_lock:
        if _latest_rear_frame is None:
            return None
        ret, buf = cv2.imencode(
            '.jpg', _latest_rear_frame,
            [cv2.IMWRITE_JPEG_QUALITY, quality]
        )
        return buf.tobytes() if ret else None


_STREAM_FPS = 15          # hard cap — protects Pi 3 CPU during heavy perception
_STREAM_PERIOD = 1.0 / _STREAM_FPS

def _mjpeg_generator():
    """Yields MJPEG boundary frames for the /video_feed route, FPS-capped."""
    next_ts = 0.0
    while True:
        wait = next_ts - time.time()
        if wait > 0:
            time.sleep(wait)
        next_ts = time.time() + _STREAM_PERIOD
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


def _rear_mjpeg_generator():
    """Yields MJPEG boundary frames for the /rear_video_feed route."""
    while True:
        jpeg = _encode_rear_frame_jpeg()
        if jpeg:
            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n'
                + jpeg +
                b'\r\n'
            )
        else:
            time.sleep(0.05)


# ── AI track target resolver ─────────────────────────────────────────────────

# All classes the tracker supports (mirrors TRACK_TARGETS keys in item_tracker.py)
_TRACK_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "bus", "truck",
    "bird", "cat", "dog", "horse", "sheep", "cow",
    "backpack", "umbrella", "sports ball", "bottle", "cup",
    "chair", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "book", "teddy bear",
]

# Common synonyms → canonical COCO class name
_SYNONYMS: dict[str, str] = {
    "ball": "sports ball", "soccer ball": "sports ball",
    "football": "sports ball", "basketball": "sports ball",
    "tennis ball": "sports ball", "volleyball": "sports ball",
    "phone": "cell phone", "mobile": "cell phone",
    "smartphone": "cell phone", "iphone": "cell phone",
    "computer": "laptop", "laptop computer": "laptop", "notebook": "laptop",
    "mug": "cup", "glass": "cup", "coffee": "cup",
    "stuffed animal": "teddy bear", "stuffed toy": "teddy bear",
    "toy bear": "teddy bear", "plush": "teddy bear",
    "bag": "backpack", "backpack": "backpack",
    "kitten": "cat", "kitty": "cat",
    "puppy": "dog", "pup": "dog",
    "bike": "bicycle", "motorbike": "motorcycle", "scooter": "motorcycle",
    "tv remote": "remote", "controller": "remote",
    "man": "person", "woman": "person", "child": "person",
    "kid": "person", "human": "person", "people": "person", "guy": "person",
    "keys": "keyboard", "keyboard": "keyboard",
}


def _resolve_track_target(text: str) -> str | None:
    """
    Map a free-text command to a COCO class name.
    Tries Claude API first (if anthropic + ANTHROPIC_API_KEY available),
    then falls back to keyword/synonym matching.
    Returns None if nothing matches.
    """
    matched = _resolve_via_gemini(text)
    if matched:
        return matched
    return _resolve_via_keywords(text)


def _resolve_via_gemini(text: str) -> str | None:
    """Call Gemini Flash to extract the target object. Returns None on any failure."""
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        return None
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        classes_str = ", ".join(_TRACK_CLASSES)
        prompt = (
            f"Map this command to a COCO object class name for an autonomous car tracker.\n"
            f"Valid classes: {classes_str}\n"
            f"Reply with ONLY the exact class name from the list, or 'none' if nothing matches.\n"
            f"Examples:\n"
            f"  'go get that ball' → sports ball\n"
            f"  'follow the dog' → dog\n"
            f"  'chase the flying bird' → bird\n"
            f"  'track the red car' → car\n"
            f"  'go to the fridge' → none\n\n"
            f"Command: {text}"
        )
        model  = genai.GenerativeModel("gemini-2.5-flash")
        result = model.generate_content(prompt).text.strip().lower()
        if result in _TRACK_CLASSES:
            return result
        return None
    except Exception as e:
        print(f"[AI Track] Gemini API error: {e}")
        return None


def _resolve_via_keywords(text: str) -> str | None:
    """Keyword + synonym matching fallback — works fully offline."""
    lower = text.lower()
    # Check synonyms first (more specific)
    for syn, cls in _SYNONYMS.items():
        if syn in lower:
            return cls
    # Then check direct class names
    for cls in _TRACK_CLASSES:
        if cls in lower:
            return cls
    return None


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

    @app.route('/rear_video_feed')
    def rear_video_feed():
        return Response(
            _rear_mjpeg_generator(),
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
                 "set_target_color", "manual_drive", "set_mode",
                 "set_track_target", "track_click",
                 "arm_sentry", "disarm_sentry", "clear_sentry_events"}
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

    @app.route('/api/ai_track', methods=['POST'])
    def api_ai_track():
        global _pending_cmd
        body = request.get_json(silent=True) or {}
        text = str(body.get('text', '')).strip()
        if not text:
            return jsonify({"error": "No text provided"}), 400

        matched = _resolve_track_target(text)
        if matched:
            with _cmd_lock:
                _pending_cmd = {"action": "set_track_target", "value": matched}
            print(f"[AI Track] '{text}' → '{matched}'")
            return jsonify({"matched": matched, "status": "ok"})
        else:
            return jsonify({"matched": None, "status": "no_match",
                            "message": f"No trackable object found in: '{text}'"}), 200

    @app.route('/api/sentry_events')
    def api_sentry_events():
        with _sentry_events_lock:
            events = list(_sentry_events)
        return jsonify(events)

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
