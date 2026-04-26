"""
sentry.py — Stationary Surveillance (SENTRY) Mode
──────────────────────────────────────────────────
Parks the car, locks the wheels, and uses the front camera for:
  • Motion detection       — frame differencing + contour area threshold
  • Person/object DNN      — SSD MobileNet v2 (shared net from dnn_detector)
  • Fire detection         — HSV colour thresholding for flame signatures
  • Smoke detection        — gray/white haze mask + per-blob variance check
  • Water leak detection   — low-sat reflective floor blob with area growth tracking
  • Weapon shape detection — elongated rigid contours near a detected person
  • Comparison snapshots   — side-by-side BEFORE/AFTER JPEG with diff highlight
  • Event log              — timestamped deque, max MAX_EVENTS entries
  • ntfy.sh alerts         — default topic: VectorVance (NTFY_TOPIC / NTFY_URL env vars)

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
SMOKE_PIXEL_RATIO = 0.010  # min fraction of frame a single smoke blob must cover
SMOKE_STDDEV_MIN  = 20.0   # min gray stddev inside a blob (flat walls score ~5-10)

# Background comparison snapshot
SNAP_BG_UPDATE_FRAMES = 30     # update clean background every N quiet frames

# Water leak detection
LEAK_ROI_FRAC        = 0.50    # bottom fraction of frame to analyse for puddles
LEAK_SAT_MAX         = 45      # puddles: very low saturation (neutral/gray)
LEAK_VAL_MIN         = 60      # minimum brightness — reflective surface
LEAK_AREA_MIN_FRAC   = 0.015   # min blob area as fraction of the floor ROI
LEAK_HISTORY_FRAMES  = 45      # rolling area window (~1.8 s at 25 fps)
LEAK_GROWTH_RATIO    = 1.40    # blob must grow 40 % within the window to alert

# Weapon shape detection
WEAPON_ASPECT_MIN    = 4.0     # elongation gate: longer / shorter side of min-area rect
WEAPON_AREA_MIN      = 600     # minimum contour area (px²)
WEAPON_SOLIDITY_MIN  = 0.60    # rigid objects are convex (solidity = area / hull area)
WEAPON_PERSON_MARGIN = 0.60    # arm-length expansion beyond each edge of person bbox

# ntfy.sh priority + tag per event type
_NTFY_META = {
    "fire":   ("5", "rotating_light,fire",            "FIRE DETECTED"),
    "weapon": ("5", "rotating_light,no_entry",        "WEAPON DETECTED"),
    "smoke":  ("4", "cloud,rotating_light",           "SMOKE DETECTED"),
    "leak":   ("4", "droplet,warning",                "WATER LEAK DETECTED"),
    "person": ("4", "bust_in_silhouette,warning",     "PERSON DETECTED"),
    "motion": ("3", "wave",                           "MOTION DETECTED"),
}

# Label colours used in the BEFORE/AFTER comparison snapshot (BGR)
_SNAP_LABEL_COLORS = {
    "fire":   (0,   80,  255),
    "weapon": (0,  200,  255),
    "smoke":  (200, 200, 255),
    "leak":   (255, 160,   0),
    "person": (80,  80,  255),
    "motion": (255, 200,   0),
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
        self._snap_count         = 0
        self.motion_active       = False
        self.person_active       = False
        self.fire_active         = False
        self.smoke_active        = False
        self.leak_active         = False
        self.weapon_active       = False
        self._bg_frame           = None   # last quiet frame — used for BEFORE half of comparison snap
        self._bg_quiet_frames    = 0
        self._leak_area_history  = deque(maxlen=LEAK_HISTORY_FRAMES)

        self._ntfy_topic = ntfy_topic or os.environ.get("NTFY_TOPIC", "VectorVance")
        self._ntfy_url   = (ntfy_url  or os.environ.get("NTFY_URL",   "https://ntfy.sh")).rstrip("/")

        os.makedirs(SNAP_DIR, exist_ok=True)

    # ── Public API ────────────────────────────────────────────────────────────

    def arm(self):
        self._armed              = True
        self._prev_gray          = None
        self._frame_count        = 0
        self._bg_frame           = None
        self._bg_quiet_frames    = 0
        self._leak_area_history.clear()
        print("[Sentry] ARMED — surveillance active")

    def disarm(self):
        self._armed              = False
        self.motion_active       = False
        self.person_active       = False
        self.fire_active         = False
        self.smoke_active        = False
        self.leak_active         = False
        self.weapon_active       = False
        self._bg_frame           = None
        self._bg_quiet_frames    = 0
        self._leak_area_history.clear()
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
        """Run all detectors and return annotated debug frame."""
        self._frame_count += 1
        debug = frame.copy()

        if not self._armed:
            cv2.putText(debug, "[SENTRY] DISARMED",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (80, 80, 80), 2)
            return debug

        motion_contours    = self._detect_motion(frame)
        smoke_blobs        = self._detect_smoke(frame)
        leak_blobs         = self._detect_leak(frame)
        self.motion_active = motion_contours is not None
        self.smoke_active  = smoke_blobs  is not None
        self.leak_active   = leak_blobs   is not None
        self.fire_active   = self._detect_fire(frame)

        if self._frame_count % DNN_SKIP == 0:
            self._last_dnn_dets = self._run_dnn(frame)

        self.person_active = any(
            label == "person" for label, _, _ in self._last_dnn_dets
        )
        weapon_contours    = self._detect_weapon(frame)
        self.weapon_active = weapon_contours is not None

        # ── Background update (quiet frames only) ────────────────────
        any_active = (self.fire_active or self.weapon_active or self.smoke_active
                      or self.person_active or self.motion_active or self.leak_active)
        if any_active:
            self._bg_quiet_frames = 0
        else:
            self._bg_quiet_frames += 1
            if self._bg_quiet_frames >= SNAP_BG_UPDATE_FRAMES:
                self._bg_frame        = frame.copy()
                self._bg_quiet_frames = 0

        # ── Alerts  (fire > weapon > smoke > person > motion > leak) ─
        if self.fire_active:
            self._maybe_log_event("fire",   frame)
        if self.weapon_active:
            self._maybe_log_event("weapon", frame)
        if self.smoke_active:
            self._maybe_log_event("smoke",  frame)
        if self.person_active:
            self._maybe_log_event("person", frame)
        if self.motion_active:
            self._maybe_log_event("motion", frame)
        if self.leak_active:
            self._maybe_log_event("leak",   frame)

        return self._draw_overlay(debug, motion_contours, smoke_blobs,
                                  leak_blobs, weapon_contours)

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

    def _detect_smoke(self, frame):
        """
        Gray/white haze mask + per-blob variance gate.

        Strategy:
          1. HSV mask: low saturation (0-50) + high brightness (120-255) isolates
             gray/white haze. Fire pixels are subtracted so flames don't trigger this.
          2. Morphological open removes salt-and-pepper noise; dilate merges nearby wisps.
          3. Each surviving contour must cover >= SMOKE_PIXEL_RATIO of the frame AND
             have an internal gray stddev >= SMOKE_STDDEV_MIN.
             - Flat surfaces (white walls, sky): stddev ~5-10 → filtered out.
             - Diffuse smoke haze: non-uniform density → stddev 20+.

        Returns a list of passing contours, or None if no smoke is detected.
        """
        hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Low-saturation, high-brightness haze
        mask = cv2.inRange(hsv, np.array([0,   0, 120]), np.array([180, 50, 255]))

        # Subtract fire-coloured pixels so a flame doesn't double-trigger as smoke
        fire_low  = cv2.inRange(hsv, np.array([0,   120, 150]), np.array([40,  255, 255]))
        fire_high = cv2.inRange(hsv, np.array([170, 120, 150]), np.array([180, 255, 255]))
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(cv2.bitwise_or(fire_low, fire_high)))

        # Morphological cleanup: open removes specks, dilate merges neighbouring wisps
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
        mask   = cv2.dilate(mask, kernel, iterations=1)

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_blob_px = frame.shape[0] * frame.shape[1] * SMOKE_PIXEL_RATIO
        smoke_blobs = []
        for c in cnts:
            if cv2.contourArea(c) < min_blob_px:
                continue
            # Variance check — draw the contour fill onto a blank mask and sample
            blob_mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.drawContours(blob_mask, [c], -1, 255, cv2.FILLED)
            pixels = gray[blob_mask > 0]
            if len(pixels) >= 200 and float(np.std(pixels)) >= SMOKE_STDDEV_MIN:
                smoke_blobs.append(c)

        return smoke_blobs if smoke_blobs else None

    def _detect_leak(self, frame):
        """
        Puddle / water leak detector.

        Works in the bottom LEAK_ROI_FRAC of the frame (floor zone).
        A puddle is low-saturation and reflective (bright), so the HSV mask
        selects S < LEAK_SAT_MAX and V > LEAK_VAL_MIN.

        False-positive guard — area growth tracking:
          The blob area is stored in a rolling deque. A leak is confirmed only
          when the 5-frame average at the end of the window is >= LEAK_GROWTH_RATIO
          times the 5-frame average at the start. Static bright floor patches
          keep a flat area and never trigger; a spreading puddle grows.

        Returns list of contours in full-frame coordinates, or None.
        """
        h, w   = frame.shape[:2]
        roi_y  = int(h * (1.0 - LEAK_ROI_FRAC))
        roi    = frame[roi_y:, :]

        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask    = cv2.inRange(hsv_roi,
                              np.array([0,   0,           LEAK_VAL_MIN]),
                              np.array([180, LEAK_SAT_MAX, 255        ]))

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
        mask   = cv2.dilate(mask, kernel, iterations=1)

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        roi_area = roi.shape[0] * roi.shape[1]
        big      = [c for c in cnts if cv2.contourArea(c) > roi_area * LEAK_AREA_MIN_FRAC]

        total_area = sum(cv2.contourArea(c) for c in big)
        self._leak_area_history.append(total_area)

        if len(self._leak_area_history) < LEAK_HISTORY_FRAMES:
            return None

        hist    = list(self._leak_area_history)
        old_avg = sum(hist[:5]) / 5
        new_avg = sum(hist[-5:]) / 5

        if old_avg < 500 or (new_avg / max(old_avg, 1)) < LEAK_GROWTH_RATIO:
            return None

        # Offset contour Y-coordinates to full-frame space
        full_cnts = [c + np.array([[[0, roi_y]]]) for c in big]
        return full_cnts if full_cnts else None

    def _detect_weapon(self, frame):
        """
        Elongated-rigid-object detector for potential weapons.

        Requires at least one person to be detected (cached DNN result) —
        the object must be near them at arm's length, reducing false positives
        from furniture, door frames, etc.

        Shape filter per contour (on Canny edges):
          • area         >= WEAPON_AREA_MIN
          • aspect ratio >= WEAPON_ASPECT_MIN  (via minAreaRect)
          • solidity     >= WEAPON_SOLIDITY_MIN (area / convex-hull area)

        Person-proximity gate:
          The contour centroid must fall inside the person bounding box
          expanded by WEAPON_PERSON_MARGIN on every side.

        Works well for a phone-screen weapon image: the DNN spots the person
        holding the phone; Canny picks up the weapon outline on the screen.

        Returns list of suspect contours, or None.
        """
        person_boxes = [
            (x1, y1, x2, y2)
            for label, _, (x1, y1, x2, y2) in self._last_dnn_dets
            if label == "person"
        ]
        if not person_boxes:
            return None

        gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges   = cv2.Canny(blurred, 30, 90)
        edges   = cv2.dilate(edges, None, iterations=1)

        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        suspects = []
        for c in cnts:
            if cv2.contourArea(c) < WEAPON_AREA_MIN:
                continue

            # Elongation via minimum-area bounding rectangle
            (_, _), (bw, bh), _ = cv2.minAreaRect(c)
            if bw < 1 or bh < 1:
                continue
            if max(bw, bh) / min(bw, bh) < WEAPON_ASPECT_MIN:
                continue

            # Solidity — rigid straight objects are close to convex
            hull      = cv2.convexHull(c)
            hull_area = cv2.contourArea(hull)
            if hull_area < 1 or cv2.contourArea(c) / hull_area < WEAPON_SOLIDITY_MIN:
                continue

            # Proximity gate — centroid inside expanded person bbox
            cx = float(np.mean(c[:, 0, 0]))
            cy = float(np.mean(c[:, 0, 1]))
            for (px1, py1, px2, py2) in person_boxes:
                pw, ph   = px2 - px1, py2 - py1
                mx, my   = pw * WEAPON_PERSON_MARGIN, ph * WEAPON_PERSON_MARGIN
                if (px1 - mx) <= cx <= (px2 + mx) and (py1 - my) <= cy <= (py2 + my):
                    suspects.append(c)
                    break

        return suspects if suspects else None

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

        snap_path = self._save_comparison_snapshot(frame, event_type)
        ts        = datetime.now().strftime("%H:%M:%S")
        event     = {"time": ts, "type": event_type, "snap": snap_path}
        with self._events_lock:
            self._events.appendleft(event)

        print(f"[Sentry] EVENT: {event_type} @ {ts}  snap={snap_path}")
        self._send_ntfy(event_type, snap_path)

    def _save_comparison_snapshot(self, trigger_frame, event_type: str) -> str:
        """
        Save a side-by-side BEFORE/AFTER JPEG.

        Left panel  — last quiet background frame (what the scene looked like).
        Right panel — current trigger frame with changed pixels tinted red.
        A thin diff pass (absdiff > 30) produces the change mask so the viewer
        immediately sees exactly what moved / appeared / disappeared.
        """
        h, w = trigger_frame.shape[:2]
        bg   = (self._bg_frame if self._bg_frame is not None
                else np.zeros((h, w, 3), dtype=np.uint8))
        bg   = cv2.resize(bg, (w, h))

        # Changed-pixel mask
        diff_gray = cv2.cvtColor(cv2.absdiff(bg, trigger_frame), cv2.COLOR_BGR2GRAY)
        _, change_mask = cv2.threshold(diff_gray, 30, 255, cv2.THRESH_BINARY)
        change_mask    = cv2.dilate(change_mask, None, iterations=2)

        # Red tint on changed pixels of the trigger frame
        trigger_hl = trigger_frame.copy()
        red_layer  = np.zeros_like(trigger_hl)
        red_layer[change_mask > 0] = (0, 0, 220)
        trigger_hl = cv2.addWeighted(trigger_hl, 0.72, red_layer, 0.28, 0)

        # Header bars with event-type label colour
        lc = _SNAP_LABEL_COLORS.get(event_type, (200, 200, 200))
        for panel, txt in ((bg, "BEFORE"), (trigger_hl, f"AFTER — {event_type.upper()}")):
            cv2.rectangle(panel, (0, 0), (w, 26), (25, 25, 25), -1)
            cv2.putText(panel, txt, (8, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                        (160, 160, 160) if txt == "BEFORE" else lc, 1)

        composite = np.hstack([bg, np.full((h, 3, 3), 55, dtype=np.uint8), trigger_hl])

        self._snap_count += 1
        ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = os.path.join(SNAP_DIR, f"sentry_{ts}_{self._snap_count:04d}.jpg")
        cv2.imwrite(fname, composite)
        return fname

    def _send_ntfy(self, event_type: str, snap_path: str):
        if not self._ntfy_topic:
            return
        priority, tags, title = _NTFY_META.get(event_type, ("3", "bell", event_type.upper()))
        url = f"{self._ntfy_url}/{self._ntfy_topic}"
        try:
            import urllib.request
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(snap_path, "rb") as f:
                img_data = f.read()
            req = urllib.request.Request(
                url,
                data=img_data,
                method="PUT",
                headers={
                    "Title":        title,
                    "Message":      f"VectorVance sentry alert at {ts}",
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

    def _draw_overlay(self, frame, motion_contours, smoke_blobs,
                       leak_blobs, weapon_contours) -> np.ndarray:
        h, w = frame.shape[:2]

        # DNN bounding boxes
        for label, conf, (x1, y1, x2, y2) in self._last_dnn_dets:
            color = (0, 0, 255) if label == "person" else (0, 165, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} {conf:.0%}",
                        (x1, max(y1 - 6, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Weapon contours — draw rotated bounding rect in amber
        if weapon_contours:
            for c in weapon_contours:
                box = np.intp(cv2.boxPoints(cv2.minAreaRect(c)))
                cv2.drawContours(frame, [box], 0, (0, 200, 255), 2)

        # Smoke blob outlines — thin light-gray
        if smoke_blobs:
            cv2.drawContours(frame, smoke_blobs, -1, (200, 200, 200), 1)

        # Leak puddle outlines — blue
        if leak_blobs:
            cv2.drawContours(frame, leak_blobs, -1, (200, 110, 0), 2)

        # Motion contour outlines — cyan-green
        if motion_contours:
            cv2.drawContours(frame, motion_contours, -1, (0, 255, 180), 1)

        # Top status banner  fire > weapon > smoke > person > motion > leak > idle
        if self.fire_active:
            banner_bg, banner_txt = (0,  50, 220),  "! FIRE DETECTED !"
        elif self.weapon_active:
            banner_bg, banner_txt = (0, 130, 200),  "! WEAPON DETECTED !"
        elif self.smoke_active:
            banner_bg, banner_txt = (60, 60, 160),  "! SMOKE DETECTED !"
        elif self.person_active:
            banner_bg, banner_txt = (0,   0, 180),  "PERSON DETECTED"
        elif self.motion_active:
            banner_bg, banner_txt = (0, 100, 200),  "MOTION DETECTED"
        elif self.leak_active:
            banner_bg, banner_txt = (130, 60,  0),  "! WATER LEAK !"
        else:
            banner_bg, banner_txt = (0,  50,   0),  "SENTRY: watching"

        cv2.rectangle(frame, (0, 0), (w, 34), banner_bg, -1)
        cv2.putText(frame, banner_txt,
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        # Event count (top-right)
        with self._events_lock:
            n_events = len(self._events)
        cv2.putText(frame, f"Events: {n_events}",
                    (w - 110, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Timestamp (bottom-left)
        cv2.putText(frame, datetime.now().strftime("%H:%M:%S"),
                    (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1)

        return frame
