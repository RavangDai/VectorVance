"""
color_sign_detector.py - VectorVance Color Tape Navigation
──────────────────────────────────────────────────────────
Detects coloured tape strips on the track.

Physical layout:
  FORK  : GREEN tape on one arm, BLUE tape on the other arm
  END   : RED tape at the destination (car stops here)

At a fork the car stops and waits for the user to pick
GREEN or BLUE via the web dashboard. The car then steers
toward that tape. When it sees RED it stops — arrived.
"""

import cv2
import numpy as np
from collections import deque


# ── HSV colour ranges ─────────────────────────────────────────────────────────
_COLOR_RANGES: dict[str, list] = {
    "GREEN": [
        # Raised min saturation 60→100, min value 60→80 to reject pale greens/foliage.
        (np.array([38,  100, 80]),  np.array([88,  255, 255])),
    ],
    "BLUE": [
        # Raised min saturation 80→110, min value 50→80 to reject dark/dull blues.
        (np.array([100, 110, 80]),  np.array([135, 255, 255])),
    ],
    "RED": [
        # Raised min saturation 100→140 and min value 60→100 to reject skin tones
        # and dimly-lit red objects (power strips, chairs, clothing).
        (np.array([0,   140, 100]), np.array([10,  255, 255])),
        (np.array([170, 140, 100]), np.array([180, 255, 255])),
    ],
}

_COLOR_BGR: dict[str, tuple] = {
    "GREEN": (0,   220, 0),
    "BLUE":  (255, 80,  0),
    "RED":   (0,   0,   255),
}

# RED tape at end — must be this big before we call it the destination
DESTINATION_COLOR    = "RED"
DESTINATION_MIN_AREA = 3000   # px²  — raised: tape must be large/close to trigger
DESTINATION_CONFIRM  = 6      # raised: 6 consecutive frames to reject brief flashes

MIN_AREA = 900   # px²  raised: ignore small coloured blobs in background


class ColorSignDetector:
    """
    Detects GREEN / BLUE path tapes at forks and RED destination tape at end.
    """

    def __init__(self, frame_width: int = 640, frame_height: int = 480):
        self.frame_width  = frame_width
        self.frame_height = frame_height

        # Which colour to steer toward (set by user via dashboard)
        self.target_color: str | None = None

        # Bottom 45% of frame only — raised from 0.45→0.55 to exclude background
        self._roi_top = int(frame_height * 0.55)

        # Smooth steering offset
        self._offset_history: deque = deque(maxlen=5)

        # Consecutive-frame counter for destination confirmation
        self._dest_counter = 0

        # Latest results
        self.detections: dict[str, dict] = {}
        self.target_det: dict | None     = None

        print(f"[ColorSign] Ready — fork colours: GREEN / BLUE | "
              f"destination: RED")

    # ── public API ────────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> dict:
        """
        Run colour detection. Call once per frame.
        Returns {color: {x, y, w, h, cx, area, side}} for all colours found.
        """
        roi = frame[self._roi_top:, :]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        self.detections = {}
        self.target_det = None

        for color, ranges in _COLOR_RANGES.items():
            mask = np.zeros(roi.shape[:2], dtype=np.uint8)
            for lo, hi in ranges:
                mask |= cv2.inRange(hsv, lo, hi)

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

            min_needed = DESTINATION_MIN_AREA if color == DESTINATION_COLOR else MIN_AREA
            if area < min_needed:
                continue

            x, y, w, h = cv2.boundingRect(best)
            cx  = x + w // 2
            det = {
                "x":    x,
                "y":    y + self._roi_top,
                "w":    w,
                "h":    h,
                "cx":   cx,
                "area": area,
                "side": self._zone(cx),
            }
            self.detections[color] = det
            if color == self.target_color:
                self.target_det = det

        # Update destination consecutive counter
        if DESTINATION_COLOR in self.detections:
            self._dest_counter += 1
        else:
            self._dest_counter = 0

        return self.detections

    def destination_reached(self) -> bool:
        """
        True when RED tape has been visible for DESTINATION_CONFIRM
        consecutive frames — car should stop.
        """
        return self._dest_counter >= DESTINATION_CONFIRM

    def get_steering_offset(self) -> float | None:
        """
        Smoothed offset (-1.0 … +1.0) toward the target tape.
        Negative = left, Positive = right, None = not visible.
        """
        if self.target_det is None:
            return None
        cx  = self.target_det["cx"]
        raw = (cx - self.frame_width / 2) / (self.frame_width / 2)
        self._offset_history.append(raw)
        return float(np.mean(self._offset_history))

    def target_visible(self) -> bool:
        return self.target_det is not None

    def set_target(self, color: str):
        """Set the path colour to follow (called when user picks on dashboard)."""
        color = color.upper()
        if color in ("GREEN", "BLUE"):
            self.target_color = color
            self._offset_history.clear()
            print(f"[ColorSign] Path target → {color}")
        else:
            print(f"[ColorSign] '{color}' is not a valid path colour. Use GREEN or BLUE.")

    def get_fork_options(self) -> list[str]:
        """
        Returns which path colours (GREEN / BLUE) are currently visible.
        Used to tell the dashboard what options to show.
        """
        return [c for c in ("GREEN", "BLUE") if c in self.detections]

    def draw_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Draw tape boxes and status line on frame."""
        cv2.line(frame,
                 (0, self._roi_top), (self.frame_width, self._roi_top),
                 (60, 60, 60), 1)

        for color, det in self.detections.items():
            bgr       = _COLOR_BGR.get(color, (180, 180, 180))
            x, y, w, h = det["x"], det["y"], det["w"], det["h"]
            is_target = (color == self.target_color)
            is_dest   = (color == DESTINATION_COLOR)

            thickness = 3 if (is_target or is_dest) else 1
            cv2.rectangle(frame, (x, y), (x + w, y + h), bgr, thickness)

            if is_dest:
                label = f"DESTINATION ({det['side']})"
            elif is_target:
                label = f">>> {color} ({det['side']}) <<<"
            else:
                label = f"{color} ({det['side']})"

            cv2.putText(frame, label, (x, y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr,
                        2 if (is_target or is_dest) else 1)

        # Status line
        if self.target_color and self.target_det:
            offset = self.get_steering_offset()
            txt = (f"FOLLOWING: {self.target_color} "
                   f"[{self.target_det['side']}] off={offset:+.2f}")
            cv2.putText(frame, txt, (10, 215),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        _COLOR_BGR.get(self.target_color, (180, 180, 180)), 1)
        elif self.target_color:
            cv2.putText(frame, f"SEEKING: {self.target_color}...",
                        (10, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        _COLOR_BGR.get(self.target_color, (100, 100, 100)), 1)

        return frame

    def reset(self):
        self._offset_history.clear()
        self._dest_counter = 0
        self.detections    = {}
        self.target_det    = None
        self.target_color  = None

    # ── internal ──────────────────────────────────────────────────────────────

    def _zone(self, cx: int) -> str:
        third = self.frame_width / 3
        if cx < third:        return "LEFT"
        if cx < 2 * third:    return "CENTER"
        return "RIGHT"
