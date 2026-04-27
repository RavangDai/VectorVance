"""
item_tracker.py — COCO object tracker for TRACK mode
──────────────────────────────────────────────────────
Reuses the SSD MobileNet v2 weights from `dnn_detector.py` to detect and
lock onto a user-chosen COCO class (person, bottle, dog, chair, ...).
The car then chases the largest matching bbox:

  set_target(name)        → pick a class to chase  (None clears)
  detect(frame)           → run DNN, update lock state
  get_steering_offset()   → -1.0 (hard left) … +1.0 (hard right)
  get_distance_proxy()    → 0.0 (far) … 1.0 (very close), from bbox height
  target_locked()         → True while the target is visible
  draw_overlay(frame)     → annotate bbox + status on the HUD

The tracker picks the largest on-screen instance each frame. No IOU
association — this is "chase whatever looks most like the target right now",
which is what you want for a remote-controlled pet-mode car.
"""

import os
import cv2
import numpy as np
from collections import deque

_INPUT_SIZE = (300, 300)

# User-friendly label → TensorFlow COCO class ID (1-indexed, as emitted by
# SSD MobileNet v2 from the TF object-detection zoo).
TRACK_TARGETS: dict[str, int] = {
    "person":       1,
    "bicycle":      2,
    "car":          3,
    "motorcycle":   4,
    "bus":          6,
    "truck":        8,
    "bird":         16,
    "cat":          17,
    "dog":          18,
    "horse":        19,
    "sheep":        20,
    "cow":          21,
    "backpack":     27,
    "umbrella":     28,
    "sports ball":  37,
    "bottle":       44,
    "cup":          47,
    "chair":        62,
    "laptop":       73,
    "mouse":        74,
    "remote":       75,
    "keyboard":     76,
    "cell phone":   77,
    "book":         84,
    "teddy bear":   88,
}

BALL_TRAIL_LEN = 48

# HSV colour presets for ball tracking.  Red wraps around 180°, so two ranges.
BALL_COLORS: dict[str, list] = {
    "green":  [((29,  86,   6), (64,  255, 255))],
    "yellow": [((20, 100, 100), (35,  255, 255))],
    "orange": [((5,  150, 100), (20,  255, 255))],
    "blue":   [((94,  80,   2), (126, 255, 255))],
    "red":    [((0,  120,  70), (10,  255, 255)), ((170, 120, 70), (180, 255, 255))],
    "pink":   [((140,  60, 100), (175, 255, 255))],
    "white":  [((0,    0, 180), (180,  30, 255))],
}


def _create_tracker(prefer: str = "KCF"):
    """
    Return a fresh single-object tracker.  On Pi 3 KCF is ~5-10x faster than
    CSRT for similar accuracy on medium-sized, high-contrast targets.
    Falls back: KCF → CSRT → legacy.MOSSE → None.
    """
    order = ["TrackerCSRT_create", "TrackerKCF_create"] if prefer == "CSRT" else \
            ["TrackerKCF_create", "TrackerCSRT_create"]
    for factory in order:
        fn = getattr(cv2, factory, None)
        if fn is not None:
            return fn(), factory.replace("Tracker", "").replace("_create", "")
    legacy = getattr(cv2, "legacy", None)
    if legacy is not None:
        for factory in ("TrackerKCF_create", "TrackerCSRT_create",
                        "TrackerMOSSE_create"):
            fn = getattr(legacy, factory, None)
            if fn is not None:
                return fn(), factory.replace("Tracker", "").replace("_create", "")
    return None, None


class ItemTracker:
    """
    Three tracking modes, sharing the same steering/distance API:

      mode = "CLASS"  → SSD MobileNet v2 picks the largest bbox of a COCO class
      mode = "CLICK"  → cv2.TrackerCSRT follows an arbitrary ROI the user picked
                        by clicking on the live video. Works on anything that
                        MobileNet doesn't know (custom toys, coloured balls, etc).
      mode = "BALL"   → HSV colour-range mask + minEnclosingCircle (PyImageSearch
                        style). Draws a fading 48-point contrail. No DNN required.
    """

    LOST_TIMEOUT      = 45   # frames of no-detection → start 360° search
    CLICK_ROI         = 90   # initial bbox side length (px) around a click
    REINIT_MAX_DIST   = 300  # px — max distance from last position to accept a re-detect
    VERIFY_INTERVAL   = 12   # frames — how often to cross-check tracker with DNN

    def __init__(self,
                 model_name: str       = "ssd_mobilenet_v2_coco.pb",
                 conf_threshold: float = 0.35,
                 frame_width: int      = 640,
                 frame_height: int     = 480,
                 skip_frames: int      = 3,
                 net = None,
                 prefer_tracker: str = "CSRT"):
        self.conf_threshold = conf_threshold
        self.frame_width    = frame_width
        self.frame_height   = frame_height
        self.skip_frames    = max(1, skip_frames)
        self._frame_counter = 0
        self.available      = False

        # Tracking mode + state
        self.mode: str = "CLASS"                # "CLASS" or "CLICK"
        self.target_class: str | None = None    # COCO name in CLASS mode, "click" in CLICK mode
        self.last_bbox:   tuple | None = None   # (x, y, w, h)
        self.last_conf:   float        = 0.0
        self.lost_frames:  int         = 999

        # Click/ROI tracker state
        self._click_tracker      = None
        self._click_init_bbox: tuple | None = None   # deferred init — needs a frame
        self._click_last_center: tuple | None = None  # last confirmed (cx, cy)
        self._click_verify_ctr:  int = 0              # frame counter for periodic DNN verify
        self._prefer_tracker     = prefer_tracker
        probe, probe_name        = _create_tracker(prefer_tracker)
        self.click_available     = probe is not None
        self.click_tracker_name  = probe_name or "none"
        if not self.click_available:
            print("[Tracker] No OpenCV single-object tracker available — "
                  "click-to-track disabled")
        else:
            print(f"[Tracker] Click-mode backend: {self.click_tracker_name}")

        # Ball tracker state
        self._ball_color:  str | None   = None
        self._ball_radius: int          = 0
        self._ball_center: tuple | None = None
        self._ball_trail:  deque        = deque(maxlen=BALL_TRAIL_LEN)

        # DNN net — share across the project when possible
        if net is not None:
            self._net      = net
            self.available = True
            print(f"[Tracker] CLASS mode using shared net "
                  f"({len(TRACK_TARGETS)} classes, skip={skip_frames})")
            return

        from dnn_detector import load_ssd_net
        self._net = load_ssd_net(model_name)
        if self._net is None:
            print("[Tracker] CLASS mode disabled (weights missing)")
            return
        self.available = True
        print(f"[Tracker] CLASS mode ready — {len(TRACK_TARGETS)} classes, "
              f"skip={skip_frames}")

    # ── Target selection ─────────────────────────────────────────────────────

    def _reset_state(self):
        self.target_class         = None
        self.last_bbox            = None
        self.last_conf            = 0.0
        self.lost_frames          = 999
        self._click_tracker       = None
        self._click_init_bbox     = None
        self._click_last_center   = None
        self._click_verify_ctr    = 0
        self._ball_color          = None
        self._ball_radius         = 0
        self._ball_center         = None
        self._ball_trail.clear()

    def set_target(self, name: str | None):
        """Switch to CLASS mode (COCO class lookup) or clear."""
        if name is None or name == "":
            if self.target_class is not None:
                print("[Tracker] Target cleared")
            self._reset_state()
            self.mode = "CLASS"
            return
        key = name.lower().replace("_", " ").strip()
        if key not in TRACK_TARGETS:
            print(f"[Tracker] Unknown class '{name}' — ignored")
            return
        self._reset_state()
        self.mode         = "CLASS"
        self.target_class = key
        print(f"[Tracker] Target → {key} (CLASS)")

    def set_click_target(self, cx: int, cy: int, size: int | None = None):
        """Switch to CLICK mode. Initialises a single-object tracker around (cx, cy)."""
        if not self.click_available:
            print("[Tracker] Click-tracker backend unavailable")
            return
        size = size or self.CLICK_ROI
        half = size // 2
        x1 = max(0, min(self.frame_width  - 1, cx - half))
        y1 = max(0, min(self.frame_height - 1, cy - half))
        x2 = max(0, min(self.frame_width  - 1, cx + half))
        y2 = max(0, min(self.frame_height - 1, cy + half))
        if x2 - x1 < 20 or y2 - y1 < 20:
            print(f"[Tracker] Click ROI too small at ({cx},{cy}) — ignored")
            return

        self._reset_state()
        self.mode                = "CLICK"
        self.target_class        = "click"
        self._click_init_bbox    = (x1, y1, x2 - x1, y2 - y1)
        self.last_bbox           = self._click_init_bbox   # show box immediately
        self.last_conf           = 1.0
        self.lost_frames         = 0
        print(f"[Tracker] Click target → bbox {self._click_init_bbox} "
              f"({self.click_tracker_name})")

    def set_ball_target(self, color: str | None):
        """Switch to BALL mode — HSV colour range detection + fading contrail."""
        if color is None or color == "":
            if self.target_class is not None:
                print("[Tracker] Ball target cleared")
            self._reset_state()
            self.mode = "CLASS"
            return
        key = color.lower().strip()
        if key not in BALL_COLORS:
            print(f"[Tracker] Unknown ball colour '{color}' — ignored")
            return
        self._reset_state()
        self.mode         = "BALL"
        self.target_class = f"ball:{key}"
        self._ball_color  = key
        print(f"[Tracker] Ball target → {key} (BALL)")

    # ── Detection ────────────────────────────────────────────────────────────

    def _nearest_dnn_detection(self, frame: np.ndarray,
                                center: tuple,
                                conf_override: float | None = None,
                                max_dist: float | None = None) -> "tuple | None":
        """
        Run a quick DNN pass and return the bbox (x,y,w,h) of the detection
        whose centre is closest to `center`, within max_dist pixels.
        Returns None when nothing is close enough.
        """
        threshold = conf_override if conf_override is not None else self.conf_threshold
        search_r  = max_dist if max_dist is not None else self.REINIT_MAX_DIST
        blob = cv2.dnn.blobFromImage(
            frame, scalefactor=1 / 127.5, size=_INPUT_SIZE,
            mean=(127.5, 127.5, 127.5), swapRB=True, crop=False,
        )
        self._net.setInput(blob)
        raw = self._net.forward()
        cx0, cy0 = center
        best_bbox = None
        best_dist = float("inf")
        for i in range(raw.shape[2]):
            conf = float(raw[0, 0, i, 2])
            if conf < threshold:
                continue
            x1 = int(raw[0, 0, i, 3] * self.frame_width)
            y1 = int(raw[0, 0, i, 4] * self.frame_height)
            x2 = int(raw[0, 0, i, 5] * self.frame_width)
            y2 = int(raw[0, 0, i, 6] * self.frame_height)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(self.frame_width - 1, x2), min(self.frame_height - 1, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            dcx, dcy = (x1 + x2) // 2, (y1 + y2) // 2
            dist = ((dcx - cx0) ** 2 + (dcy - cy0) ** 2) ** 0.5
            if dist < best_dist and dist < search_r:
                best_dist = dist
                best_bbox = (x1, y1, x2 - x1, y2 - y1)
        return best_bbox

    def _detect_ball(self, frame: np.ndarray):
        """HSV colour mask → minEnclosingCircle → centroid. Updates trail deque."""
        ranges = BALL_COLORS.get(self._ball_color)
        if ranges is None:
            self.lost_frames += 1
            return

        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv     = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, np.array(ranges[0][0]), np.array(ranges[0][1]))
        for lo, hi in ranges[1:]:
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, np.array(lo), np.array(hi)))

        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            self._ball_center = None
            self.last_bbox    = None
            self.lost_frames += 1
            return

        c = max(cnts, key=cv2.contourArea)
        (bx, by), radius = cv2.minEnclosingCircle(c)
        if radius < 8:
            self._ball_center = None
            self.last_bbox    = None
            self.lost_frames += 1
            return

        M = cv2.moments(c)
        if M["m00"] == 0:
            self.lost_frames += 1
            return

        cx                = int(M["m10"] / M["m00"])
        cy                = int(M["m01"] / M["m00"])
        r                 = int(radius)
        self._ball_center = (cx, cy)
        self._ball_radius = r
        self._ball_trail.appendleft((cx, cy))
        self.last_bbox    = (max(0, cx - r), max(0, cy - r), r * 2, r * 2)
        self.last_conf    = 1.0
        self.lost_frames  = 0

    def detect(self, frame: np.ndarray):
        self._frame_counter += 1
        if self.target_class is None:
            return

        # ── BALL mode — HSV colour mask, no DNN, runs every frame ──
        if self.mode == "BALL":
            self._detect_ball(frame)
            return

        # ── CLICK mode — CV tracker every frame (needs continuous updates) ──
        if self.mode == "CLICK":
            if self._click_init_bbox is not None:
                inst, _ = _create_tracker(self._prefer_tracker)
                if inst is None:
                    self.target_class = None
                    return

                # Snap to nearest DNN detection so CSRT gets exact object bounds,
                # not an arbitrary fixed-size square around the click point.
                init_bbox = self._click_init_bbox
                if self.available:
                    cx = init_bbox[0] + init_bbox[2] // 2
                    cy = init_bbox[1] + init_bbox[3] // 2
                    snapped = self._nearest_dnn_detection(
                        frame, (cx, cy),
                        conf_override=0.30,   # lenient — item may be far/small
                        max_dist=self.CLICK_ROI,
                    )
                    if snapped is not None:
                        init_bbox = snapped

                self._click_tracker = inst
                self._click_tracker.init(frame, init_bbox)
                x, y, w, h = init_bbox
                self.last_bbox          = init_bbox
                self._click_last_center = (x + w // 2, y + h // 2)
                self._click_init_bbox   = None
                return  # keep the initial bbox shown this frame
            if self._click_tracker is None:
                return

            ok, bbox = self._click_tracker.update(frame)
            self._click_verify_ctr += 1

            if ok:
                x, y, w, h = (int(v) for v in bbox)
                self.last_bbox          = (x, y, w, h)
                self.last_conf          = 0.90
                self.lost_frames        = 0
                self._click_last_center = (x + w // 2, y + h // 2)

                # Periodic DNN cross-check: if the tracker has drifted far from any
                # real detection, re-initialise on the nearest detection instead.
                if self.available and self._click_verify_ctr % self.VERIFY_INTERVAL == 0:
                    candidate = self._nearest_dnn_detection(frame, self._click_last_center)
                    if candidate is not None:
                        cx, cy, cw, ch = candidate
                        ccx, ccy = cx + cw // 2, cy + ch // 2
                        lcx, lcy = self._click_last_center
                        drift = ((ccx - lcx) ** 2 + (ccy - lcy) ** 2) ** 0.5
                        # Only re-init when DNN target has moved significantly from tracker
                        if drift > 40:
                            inst, _ = _create_tracker(self._prefer_tracker)
                            if inst is not None:
                                try:
                                    inst.init(frame, candidate)
                                    self._click_tracker = inst
                                    self.last_bbox = candidate
                                    self._click_last_center = (ccx, ccy)
                                except Exception:
                                    pass
            else:
                # Tracker lost — do NOT keep stale bbox (that causes "chasing the dot")
                self.last_bbox   = None
                self.lost_frames += 1

                # Attempt DNN re-detection near last known position
                if self.available and self._click_last_center is not None:
                    candidate = self._nearest_dnn_detection(
                        frame, self._click_last_center,
                        conf_override=0.30,  # lenient threshold — item may be far/small
                    )
                    if candidate is not None:
                        inst, _ = _create_tracker(self._prefer_tracker)
                        if inst is not None:
                            try:
                                inst.init(frame, candidate)
                                self._click_tracker  = inst
                                cx, cy, cw, ch       = candidate
                                self.last_bbox        = candidate
                                self.last_conf        = 0.75
                                self.lost_frames      = 0
                                self._click_last_center = (cx + cw // 2, cy + ch // 2)
                                print(f"[Tracker] Re-locked via DNN at {self._click_last_center}")
                            except Exception:
                                pass
            return

        # ── CLASS mode — DNN inference (gated by skip_frames) ────────────
        if not self.available:
            return
        if self._frame_counter % self.skip_frames != 0:
            return  # skip frame — don't penalize lost_frames

        blob = cv2.dnn.blobFromImage(
            frame,
            scalefactor=1 / 127.5,
            size=_INPUT_SIZE,
            mean=(127.5, 127.5, 127.5),
            swapRB=True,
            crop=False,
        )
        self._net.setInput(blob)
        raw = self._net.forward()

        tf_id = TRACK_TARGETS[self.target_class]
        best_bbox: tuple | None = None
        best_area = 0
        best_conf = 0.0

        for i in range(raw.shape[2]):
            conf = float(raw[0, 0, i, 2])
            cls  = int(raw[0, 0, i, 1])
            if conf < self.conf_threshold or cls != tf_id:
                continue
            x1 = int(raw[0, 0, i, 3] * self.frame_width)
            y1 = int(raw[0, 0, i, 4] * self.frame_height)
            x2 = int(raw[0, 0, i, 5] * self.frame_width)
            y2 = int(raw[0, 0, i, 6] * self.frame_height)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(self.frame_width - 1, x2), min(self.frame_height - 1, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            area = (x2 - x1) * (y2 - y1)
            if area > best_area:
                best_area = area
                best_bbox = (x1, y1, x2 - x1, y2 - y1)
                best_conf = conf

        if best_bbox is not None:
            self.last_bbox   = best_bbox
            self.last_conf   = best_conf
            self.lost_frames = 0
        else:
            self.lost_frames += 1

    # ── Queries ──────────────────────────────────────────────────────────────

    def target_locked(self) -> bool:
        return (self.last_bbox is not None
                and self.lost_frames < self.LOST_TIMEOUT)

    def get_steering_offset(self) -> float | None:
        if not self.target_locked():
            return None
        x, _, w, _ = self.last_bbox
        cx = x + w / 2
        half = self.frame_width / 2
        return max(-1.0, min(1.0, (cx - half) / half))

    def get_distance_proxy(self) -> float | None:
        """Bbox height as a rough distance proxy — 0=far, 1=close."""
        if not self.target_locked():
            return None
        _, _, _, bh = self.last_bbox
        return max(0.0, min(1.0, bh / self.frame_height))

    # ── Shape contour ────────────────────────────────────────────────────────

    def _extract_contour(self, frame: np.ndarray, bbox: tuple):
        """Return the largest edge contour inside bbox, offset to full-frame coords."""
        x, y, w, h = bbox
        if w < 15 or h < 15:
            return None
        roi    = frame[y:y + h, x:x + w]
        gray   = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur   = cv2.GaussianBlur(gray, (5, 5), 0)
        edges  = cv2.Canny(blur, 25, 90)
        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None
        largest = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(largest) < 40:
            return None
        return largest + np.array([[[x, y]]])   # shift to full-frame coords

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def reset(self):
        self._reset_state()
        self.mode = "CLASS"

    def draw_overlay(self, frame: np.ndarray) -> np.ndarray:
        if self.target_class is None:
            return frame
        h, w = frame.shape[:2]

        if self.target_locked():
            status, color = "LOCKED", (0, 220, 255)
        elif self.lost_frames < 60:
            status, color = "SEARCHING", (80, 200, 240)
        else:
            status, color = "LOST", (120, 120, 120)

        label = ("BALL:" + (self._ball_color or "").upper() if self.mode == "BALL"
                 else "CLICK" if self.mode == "CLICK"
                 else self.target_class.upper())
        cv2.putText(frame,
                    f"TRACK: {label}  [{status}]",
                    (10, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        if self.target_locked() and self.last_bbox is not None:
            x, y, bw, bh = self.last_bbox
            cx, cy = x + bw // 2, y + bh // 2

            if self.mode == "BALL":
                # Fading contrail — thickest near ball, thinnest at tail
                for i in range(1, len(self._ball_trail)):
                    if self._ball_trail[i - 1] is None or self._ball_trail[i] is None:
                        continue
                    thickness = max(1, int(np.sqrt(BALL_TRAIL_LEN / float(i + 1)) * 2.5))
                    cv2.line(frame, self._ball_trail[i - 1], self._ball_trail[i],
                             (0, 0, 255), thickness)
                if self._ball_center and self._ball_radius > 0:
                    cv2.circle(frame, self._ball_center, self._ball_radius, color, 2)
            else:
                # Draw actual object shape contour
                contour = self._extract_contour(frame, self.last_bbox)
                if contour is not None:
                    overlay = frame.copy()
                    cv2.drawContours(overlay, [contour], -1, color, -1)
                    cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)
                    cv2.drawContours(frame, [contour], -1, color, 2)
                else:
                    blen = min(bw, bh) // 4
                    for px, py in [(x, y), (x + bw, y), (x, y + bh), (x + bw, y + bh)]:
                        dx = blen if px == x else -blen
                        dy = blen if py == y else -blen
                        cv2.line(frame, (px, py), (px + dx, py), color, 2)
                        cv2.line(frame, (px, py), (px, py + dy), color, 2)

            # Line from bottom center to object + center dot
            cv2.line(frame, (w // 2, h - 40), (cx, cy), color, 1)
            cv2.circle(frame, (cx, cy), 5, color, -1)
            cv2.putText(frame, f"{self.last_conf:.0%}",
                        (x, max(y - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return frame
