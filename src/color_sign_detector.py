"""
color_sign_detector.py - VectorVance Color Tape Navigation
──────────────────────────────────────────────────────────
At a 3-way fork, three colored tape strips mark each path.
This module detects the tapes and steers the car toward
the target color (e.g. GREEN = shortest path).

Physical setup:
  Place a colored tape strip (~8×5 cm) at the entrance of
  each fork arm — one color per arm (GREEN / BLUE / RED).
  Tell the car which color to follow via the web dashboard
  or the --target-color argument.

Detection works on the bottom half of the frame (where the
ground tape appears). Works independently of YOLO — it is
a simple HSV threshold pass, so it runs on every frame
with no performance cost.
"""

import cv2
import numpy as np
from collections import deque


# ── HSV colour ranges ─────────────────────────────────────────────────────────
# Tune lo/hi if detection is unreliable under your lighting conditions.
_COLOR_RANGES: dict[str, list] = {
    "GREEN": [
        (np.array([38,  60,  60]),  np.array([88,  255, 255])),
    ],
    "BLUE": [
        (np.array([100, 80,  50]),  np.array([135, 255, 255])),
    ],
    "RED": [
        (np.array([0,   100, 60]),  np.array([10,  255, 255])),
        (np.array([170, 100, 60]),  np.array([180, 255, 255])),
    ],
}

# BGR display colours for the HUD overlay
_COLOR_BGR: dict[str, tuple] = {
    "GREEN": (0,   220, 0),
    "BLUE":  (255, 80,  0),
    "RED":   (0,   0,   255),
}

MIN_AREA = 400   # px² — blobs smaller than this are ignored as noise


class ColorSignDetector:
    """
    Detects coloured tape strips at a 3-way fork and returns
    a steering offset so the car turns toward the target colour.
    """

    def __init__(self, target_color: str = "GREEN",
                 frame_width: int  = 640,
                 frame_height: int = 480):
        self.frame_width  = frame_width
        self.frame_height = frame_height
        self.target_color = target_color.upper()

        # Scan only the bottom half — tape is on the ground / low in frame
        self._roi_top = int(frame_height * 0.45)

        # Smooth the offset over the last 5 frames to avoid jitter
        self._offset_history: deque = deque(maxlen=5)

        # Results updated by detect()
        self.detections: dict[str, dict] = {}   # color → {x,y,w,h,cx,area,side}
        self.target_det: dict | None     = None  # detection for the target colour

        print(f"[ColorSign] Ready — target={self.target_color} | "
              f"colours={', '.join(_COLOR_RANGES)}")

    # ── public API ────────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> dict:
        """
        Run colour detection on frame. Call once per frame.
        Returns: {color_name: {x, y, w, h, cx, area, side}, ...}
        """
        roi = frame[self._roi_top:, :]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        self.detections = {}
        self.target_det = None

        for color, ranges in _COLOR_RANGES.items():
            # Build combined mask for this colour
            mask = np.zeros(roi.shape[:2], dtype=np.uint8)
            for lo, hi in ranges:
                mask |= cv2.inRange(hsv, lo, hi)

            # Morphological cleanup
            k    = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

            cnts, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if not cnts:
                continue

            best = max(cnts, key=cv2.contourArea)
            area = cv2.contourArea(best)
            if area < MIN_AREA:
                continue

            x, y, w, h = cv2.boundingRect(best)
            cx  = x + w // 2
            det = {
                "x":    x,
                "y":    y + self._roi_top,   # convert to full-frame coords
                "w":    w,
                "h":    h,
                "cx":   cx,
                "area": area,
                "side": self._zone(cx),
            }
            self.detections[color] = det
            if color == self.target_color:
                self.target_det = det

        return self.detections

    def get_steering_offset(self) -> float | None:
        """
        Smoothed offset (-1.0 … +1.0) pointing toward the target tape.
          Negative  → steer left
          Positive  → steer right
          None      → target colour not visible
        """
        if self.target_det is None:
            return None
        cx  = self.target_det["cx"]
        raw = (cx - self.frame_width / 2) / (self.frame_width / 2)
        self._offset_history.append(raw)
        return float(np.mean(self._offset_history))

    def target_visible(self) -> bool:
        """True if the target colour tape was seen in the last detect() call."""
        return self.target_det is not None

    def set_target(self, color: str):
        """Change the target colour at runtime (e.g. from web command)."""
        color = color.upper()
        if color in _COLOR_RANGES:
            self.target_color = color
            self._offset_history.clear()
            print(f"[ColorSign] Target → {color}")
        else:
            print(f"[ColorSign] Unknown colour '{color}'. "
                  f"Options: {list(_COLOR_RANGES)}")

    def draw_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Draw tape bounding boxes and status text on the frame."""
        # Subtle scan-zone line
        cv2.line(frame,
                 (0, self._roi_top), (self.frame_width, self._roi_top),
                 (60, 60, 60), 1)

        for color, det in self.detections.items():
            bgr       = _COLOR_BGR.get(color, (180, 180, 180))
            x, y, w, h = det["x"], det["y"], det["w"], det["h"]
            is_target = (color == self.target_color)
            thickness = 3 if is_target else 1

            cv2.rectangle(frame, (x, y), (x + w, y + h), bgr, thickness)
            label = f"{'>>> ' if is_target else ''}{color} ({det['side']})"
            cv2.putText(frame, label, (x, y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr,
                        2 if is_target else 1)

        # Status line on HUD
        if self.target_det:
            offset = self.get_steering_offset()
            txt = (f"TAPE: {self.target_color} "
                   f"[{self.target_det['side']}] off={offset:+.2f}")
            cv2.putText(frame, txt, (10, 215),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        _COLOR_BGR.get(self.target_color, (180, 180, 180)), 1)
        else:
            cv2.putText(frame,
                        f"TAPE: {self.target_color} — scanning...",
                        (10, 215),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), 1)

        return frame

    def reset(self):
        self._offset_history.clear()
        self.detections = {}
        self.target_det = None

    # ── internal ──────────────────────────────────────────────────────────────

    def _zone(self, cx: int) -> str:
        """Return LEFT / CENTER / RIGHT based on x position."""
        third = self.frame_width / 3
        if cx < third:
            return "LEFT"
        if cx < 2 * third:
            return "CENTER"
        return "RIGHT"
