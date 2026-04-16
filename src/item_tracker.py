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


def _create_tracker(prefer: str = "KCF"):
    """
    Return a fresh single-object tracker.  On Pi 3 KCF is ~5-10x faster than
    CSRT for similar accuracy on medium-sized, high-contrast targets.
    Falls back: KCF → CSRT → legacy.MOSSE → None.
    """
    order = ["TrackerKCF_create", "TrackerCSRT_create"] if prefer == "KCF" else \
            ["TrackerCSRT_create", "TrackerKCF_create"]
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
    Two tracking modes, sharing the same steering/distance API:

      mode = "CLASS"  → SSD MobileNet v2 picks the largest bbox of a COCO class
      mode = "CLICK"  → cv2.TrackerCSRT follows an arbitrary ROI the user picked
                        by clicking on the live video. Works on anything that
                        MobileNet doesn't know (custom toys, coloured balls, etc).
    """

    LOST_TIMEOUT = 20   # frames of no-detection → declare lost
    CLICK_ROI    = 90   # initial bbox side length (px) around a click

    def __init__(self,
                 model_name: str       = "ssd_mobilenet_v2_coco.pb",
                 conf_threshold: float = 0.45,
                 frame_width: int      = 640,
                 frame_height: int     = 480,
                 skip_frames: int      = 3,
                 net = None,
                 prefer_tracker: str = "KCF"):
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
        self._prefer_tracker     = prefer_tracker
        probe, probe_name        = _create_tracker(prefer_tracker)
        self.click_available     = probe is not None
        self.click_tracker_name  = probe_name or "none"
        if not self.click_available:
            print("[Tracker] No OpenCV single-object tracker available — "
                  "click-to-track disabled")
        else:
            print(f"[Tracker] Click-mode backend: {self.click_tracker_name}")

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
        self.target_class       = None
        self.last_bbox          = None
        self.last_conf          = 0.0
        self.lost_frames        = 999
        self._click_tracker     = None
        self._click_init_bbox   = None

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

    # ── Detection ────────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray):
        self._frame_counter += 1
        if self.target_class is None:
            return

        # ── CLICK mode — CV tracker every frame (needs continuous updates) ──
        if self.mode == "CLICK":
            if self._click_init_bbox is not None:
                inst, _ = _create_tracker(self._prefer_tracker)
                if inst is None:
                    self.target_class = None
                    return
                self._click_tracker = inst
                self._click_tracker.init(frame, self._click_init_bbox)
                self._click_init_bbox = None
                return  # keep the initial bbox shown this frame
            if self._click_tracker is None:
                return
            ok, bbox = self._click_tracker.update(frame)
            if ok:
                x, y, w, h = (int(v) for v in bbox)
                self.last_bbox   = (x, y, w, h)
                self.last_conf   = 0.90   # CV trackers don't expose a score
                self.lost_frames = 0
            else:
                self.lost_frames += 1
            return

        # ── CLASS mode — DNN inference (gated by skip_frames) ────────────
        if not self.available:
            return
        if self._frame_counter % self.skip_frames != 0:
            self.lost_frames += 1
            return

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

        label = "CLICK" if self.mode == "CLICK" else self.target_class.upper()
        cv2.putText(frame,
                    f"TRACK: {label}  [{status}]",
                    (10, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        if self.target_locked() and self.last_bbox is not None:
            x, y, bw, bh = self.last_bbox
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 2)
            cx, cy = x + bw // 2, y + bh // 2
            cv2.line(frame, (w // 2, h - 40), (cx, cy), color, 1)
            cv2.circle(frame, (cx, cy), 5, color, -1)
            cv2.putText(frame, f"{self.last_conf:.0%}",
                        (x, max(y - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return frame
