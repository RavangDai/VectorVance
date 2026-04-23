"""
sentry.py — Stationary Surveillance (SENTRY) Mode
──────────────────────────────────────────────────
Parks the car, locks the wheels, and uses the front camera for:
  • Motion detection  — frame differencing + contour area threshold
  • Person/object DNN — SSD MobileNet v2 (shared net from dnn_detector)
  • Fire detection    — HSV colour thresholding for flame signatures
  • Event log         — timestamped deque, max MAX_EVENTS entries
  • Snapshot saving   — writes JPEGs to /home/pi/sentry_snaps/
  • ntfy.sh alerts    — set NTFY_TOPIC env var (optional: NTFY_URL for self-hosted)

Usage from main.py:
    sentry = SentryMonitor(net=shared_net)
    sentry.arm()
    debug = sentry.process(frame)   # call every frame; motors must be stopped by caller
    events = sentry.events          # list of event dicts (newest first)
    sentry.disarm()
"""

import os
import cv2
import time
import threading
import numpy as np
from collections import deque
from datetime import datetime

# ── Classes we alert on (SSD MobileNet v2 COCO) ─────────────────────────────
_SENTRY_CLASSES = {
    0:  "person",
    1:  "bicycle",
    2:  "car",
    3:  "motorcycle",
    5:  "bus",
    7:  "truck",
    14: "bird",
    15: "cat",
    16: "dog",
}
_INPUT_SIZE = (300, 300)

SNAP_DIR          = "/home/pi/sentry_snaps"
MAX_EVENTS        = 100
ALERT_COOLDOWN_S  = 30.0   # seconds between alerts of the same type
DNN_SKIP          = 8      # run DNN every Nth frame (CPU budget)
MOTION_AREA_MIN   = 1200   # px² — contours smaller than this are noise
PERSON_CONF       = 0.50   # minimum DNN confidence
FIRE_PIXEL_RATIO  = 0.015  # fraction of frame that must be fire-coloured

# ntfy.sh priority + tag per event type
_NTFY_META = {
    "fire":   ("5",      "rotating_light,fire",            "FIRE DETECTED"),
    "person": ("4",      "bust_in_silhouette,warning",     "PERSON DETECTED"),
    "motion": ("3",      "wave",                           "MOTION DETECTED"),
}


class SentryMonitor:

    def __init__(self, net=None, ntfy_topic: str = "", ntfy_url: str = ""):
        self._net             = net
        self._armed           = False
        self._prev_gray       = None
        self._frame_count     = 0
        self._last_dnn_dets   = []   # list of (label, conf, (x1,y1,x2,y2))
        self._events          = deque(maxlen=MAX_EVENTS)
        self._events_lock     = threading.Lock()
        self._alert_times     = {}   # event_type → last alert timestamp
        self._snap_count      = 0
        self.motion_active    = False
        self.person_active    = False
        self.fire_active      = False

        self._ntfy_topic = ntfy_topic or os.environ.get("NTFY_TOPIC", "")
        self._ntfy_url   = (ntfy_url  or os.environ.get("NTFY_URL",   "https://ntfy.sh")).rstrip("/")

        os.makedirs(SNAP_DIR, exist_ok=True)

    # ── Public API ────────────────────────────────────────────────────────────

    def arm(self):
        self._armed       = True
        self._prev_gray   = None
        self._frame_count = 0
        print("[Sentry] ARMED — surveillance active")

    def disarm(self):
        self._armed        = False
        self.motion_active = False
        self.person_active = False
        self.fire_active   = False
        print("[Sentry] Disarmed")

    @property
    def armed(self) -> bool:
        return self._armed

    @property
    def events(self) -> list:
        with self._events_lock:
            return list(self._events)

    def clear_events(self):
        with self._events_lock:
            self._events.clear()

    # ── Main per-frame call ───────────────────────────────────────────────────

    def process(self, frame) -> np.ndarray:
        """Run motion + DNN + fire detection. Returns annotated debug frame."""
        self._frame_count += 1
        debug = frame.copy()

        if not self._armed:
            cv2.putText(debug, "[SENTRY] DISARMED",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (80, 80, 80), 2)
            return debug

        motion_contours    = self._detect_motion(frame)
        self.motion_active = motion_contours is not None
        self.fire_active   = self._detect_fire(frame)

        if self._frame_count % DNN_SKIP == 0:
            self._last_dnn_dets = self._run_dnn(frame)

        self.person_active = any(
            label == "person" for label, _, _ in self._last_dnn_dets
        )

        # Fire takes priority; always alert regardless of other states
        if self.fire_active:
            self._maybe_log_event("fire", frame)
        if self.person_active:
            self._maybe_log_event("person", frame)
        if self.motion_active:
            self._maybe_log_event("motion", frame)

        return self._draw_overlay(debug, motion_contours)

    # ── Detection helpers ─────────────────────────────────────────────────────

    def _detect_motion(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if self._prev_gray is None:
            self._prev_gray = gray
            return None

        diff  = cv2.absdiff(self._prev_gray, gray)
        self._prev_gray = gray

        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        thresh    = cv2.dilate(thresh, None, iterations=2)

        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        big = [c for c in cnts if cv2.contourArea(c) > MOTION_AREA_MIN]
        return big if big else None

    def _detect_fire(self, frame) -> bool:
        """HSV colour threshold for fire signatures (red-orange-yellow, high brightness)."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Orange/yellow flame core: H=0-40, high saturation, high value
        mask_low  = cv2.inRange(hsv, np.array([0,  120, 150]), np.array([40, 255, 255]))
        # Upper-red hue wrap (H=170-180)
        mask_high = cv2.inRange(hsv, np.array([170, 120, 150]), np.array([180, 255, 255]))
        combined  = cv2.bitwise_or(mask_low, mask_high)
        fire_px   = cv2.countNonZero(combined)
        threshold = int(frame.shape[0] * frame.shape[1] * FIRE_PIXEL_RATIO)
        return fire_px > threshold

    def _run_dnn(self, frame) -> list:
        if self._net is None:
            return []
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            frame, 1.0 / 127.5, _INPUT_SIZE, (127.5, 127.5, 127.5), swapRB=True)
        self._net.setInput(blob)
        out  = self._net.forward()
        dets = []
        for det in out[0, 0]:
            conf     = float(det[2])
            if conf < PERSON_CONF:
                continue
            class_id = int(det[1]) - 1   # TF uses 1-indexed COCO IDs
            label    = _SENTRY_CLASSES.get(class_id)
            if label is None:
                continue
            x1 = int(det[3] * w);  y1 = int(det[4] * h)
            x2 = int(det[5] * w);  y2 = int(det[6] * h)
            dets.append((label, conf, (x1, y1, x2, y2)))
        return dets

    # ── Event logging + alerting ──────────────────────────────────────────────

    def _maybe_log_event(self, event_type: str, frame):
        now  = time.time()
        last = self._alert_times.get(event_type, 0)
        if now - last < ALERT_COOLDOWN_S:
            return
        self._alert_times[event_type] = now

        snap_path = self._save_snapshot(frame)
        ts        = datetime.now().strftime("%H:%M:%S")
        event     = {"time": ts, "type": event_type, "snap": snap_path}
        with self._events_lock:
            self._events.appendleft(event)

        print(f"[Sentry] EVENT: {event_type} @ {ts}  snap={snap_path}")
        self._send_ntfy(event_type, snap_path)

    def _save_snapshot(self, frame) -> str:
        self._snap_count += 1
        ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = os.path.join(SNAP_DIR, f"sentry_{ts}_{self._snap_count:04d}.jpg")
        cv2.imwrite(fname, frame)
        return fname

    def _send_ntfy(self, event_type: str, snap_path: str):
        if not self._ntfy_topic:
            return
        priority, tags, title = _NTFY_META.get(event_type, ("3", "bell", event_type.upper()))
        url = f"{self._ntfy_url}/{self._ntfy_topic}"
        try:
            import urllib.request
            with open(snap_path, "rb") as f:
                img_data = f.read()
            req = urllib.request.Request(
                url,
                data=img_data,
                method="PUT",
                headers={
                    "Title":        title,
                    "Tags":         tags,
                    "Priority":     priority,
                    "Filename":     "snap.jpg",
                    "Content-Type": "image/jpeg",
                },
            )
            urllib.request.urlopen(req, timeout=10)
        except Exception as e:
            print(f"[Sentry] ntfy send failed: {e}")

    # ── HUD ───────────────────────────────────────────────────────────────────

    def _draw_overlay(self, frame, motion_contours) -> np.ndarray:
        h, w = frame.shape[:2]

        # DNN bounding boxes
        for label, conf, (x1, y1, x2, y2) in self._last_dnn_dets:
            color = (0, 0, 255) if label == "person" else (0, 165, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} {conf:.0%}",
                        (x1, max(y1 - 6, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Motion contour outlines
        if motion_contours:
            cv2.drawContours(frame, motion_contours, -1, (0, 255, 180), 1)

        # Top status banner — fire > person > motion > idle
        if self.fire_active:
            banner_bg, banner_txt = (0, 50, 220), "! FIRE DETECTED !"
        elif self.person_active:
            banner_bg, banner_txt = (0, 0, 180), "PERSON DETECTED"
        elif self.motion_active:
            banner_bg, banner_txt = (0, 100, 200), "MOTION DETECTED"
        else:
            banner_bg, banner_txt = (0, 50, 0), "SENTRY: watching"

        cv2.rectangle(frame, (0, 0), (w, 34), banner_bg, -1)
        cv2.putText(frame, banner_txt,
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        # Event count (top-right corner)
        with self._events_lock:
            n_events = len(self._events)
        cv2.putText(frame, f"Events: {n_events}",
                    (w - 110, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Timestamp (bottom-left)
        cv2.putText(frame, datetime.now().strftime("%H:%M:%S"),
                    (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1)

        return frame
