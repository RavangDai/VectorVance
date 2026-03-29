"""
intersection_detector.py - VectorVance Fork/Intersection Detection
──────────────────────────────────────────────────────────────────
Detects when the road splits into a fork by analyzing:
  1. Sudden increase in detected lane lines (2 → 3+)
  2. Lane width expanding rapidly (lanes diverging)
  3. Confidence pattern: one side drops while the other holds

Works alongside perception.py — reads the same edge/line data.

USAGE:
    detector = IntersectionDetector()
    is_fork = detector.update(lines, left_conf, right_conf, lane_width)
"""

import numpy as np
from collections import deque


class IntersectionDetector:
    """Detects lane forks and intersections from vision data."""

    def __init__(self):
        # ── DETECTION THRESHOLDS ─────────────────────────────────────
        self.min_lines_for_fork = 4          # need 4+ hough lines to suspect fork
        self.lane_width_expand_ratio = 1.5   # if lane width grows 50%+ = diverging
        self.confidence_drop_threshold = 0.3 # one side confidence drops below this

        # ── HISTORY (temporal filtering) ─────────────────────────────
        self.history_size = 10
        self.line_count_history = deque(maxlen=self.history_size)
        self.lane_width_history = deque(maxlen=self.history_size)
        self.fork_score_history = deque(maxlen=8)

        # ── STATE ────────────────────────────────────────────────────
        self.is_fork_detected = False
        self.fork_confidence = 0.0
        self.frames_since_fork = 0
        self.cooldown = 0                    # frames to wait after a detection
        self.cooldown_frames = 30            # don't re-trigger for 30 frames

        # ── BASELINE (learned from first ~20 frames) ────────────────
        self.baseline_line_count = None
        self.baseline_lane_width = None
        self._calibration_frames = 0
        self._calibration_line_counts = []
        self._calibration_widths = []
        self.calibrated = False

    def update(self, num_lines, left_confidence, right_confidence,
               lane_width=None, left_fit=None, right_fit=None):
        """
        Call every frame with current vision data.

        Args:
            num_lines:        number of raw hough lines detected this frame
            left_confidence:  perception.py left lane confidence (0-1)
            right_confidence: perception.py right lane confidence (0-1)
            lane_width:       pixel distance between left and right lane at bottom
                              (None if only one lane visible)
            left_fit:         left lane polynomial fit (or None)
            right_fit:        right lane polynomial fit (or None)

        Returns:
            True if fork/intersection is detected this frame
        """
        # ── COOLDOWN ─────────────────────────────────────────────────
        if self.cooldown > 0:
            self.cooldown -= 1
            self.is_fork_detected = False
            self.fork_confidence = 0.0
            return False

        # ── CALIBRATION (first 20 frames = learn baseline) ───────────
        if not self.calibrated:
            self._calibration_frames += 1
            self._calibration_line_counts.append(num_lines)
            if lane_width is not None:
                self._calibration_widths.append(lane_width)

            if self._calibration_frames >= 20:
                self.baseline_line_count = np.mean(self._calibration_line_counts) if self._calibration_line_counts else 3
                self.baseline_lane_width = np.mean(self._calibration_widths) if self._calibration_widths else 200
                self.calibrated = True

            self.is_fork_detected = False
            return False

        # ── UPDATE HISTORY ───────────────────────────────────────────
        self.line_count_history.append(num_lines)
        if lane_width is not None:
            self.lane_width_history.append(lane_width)

        # ── COMPUTE FORK SCORE (0-1) ────────────────────────────────
        score = 0.0
        reasons = []

        # Signal 1: Line count spike
        avg_lines = np.mean(list(self.line_count_history)[-5:]) if len(self.line_count_history) >= 3 else num_lines
        if avg_lines > self.baseline_line_count * 1.8 and avg_lines >= self.min_lines_for_fork:
            line_score = min(1.0, (avg_lines - self.baseline_line_count) / self.baseline_line_count)
            score += line_score * 0.35
            reasons.append(f"lines:{avg_lines:.0f}vs{self.baseline_line_count:.0f}")

        # Signal 2: Lane width expanding
        if len(self.lane_width_history) >= 5 and lane_width is not None:
            recent_width = np.mean(list(self.lane_width_history)[-3:])
            if recent_width > self.baseline_lane_width * self.lane_width_expand_ratio:
                width_score = min(1.0, (recent_width / self.baseline_lane_width - 1.0))
                score += width_score * 0.35
                reasons.append(f"width:{recent_width:.0f}vs{self.baseline_lane_width:.0f}")

        # Signal 3: Asymmetric confidence (one lane fading = fork diverging)
        conf_diff = abs(left_confidence - right_confidence)
        min_conf = min(left_confidence, right_confidence)
        if conf_diff > 0.4 and min_conf < self.confidence_drop_threshold:
            score += 0.30
            reasons.append(f"conf_asym:{conf_diff:.2f}")

        # Signal 4: Both lane fits diverging at the top
        if left_fit is not None and right_fit is not None:
            # Check if lanes are getting wider toward the top (diverging)
            y_bottom = 480
            y_top = int(480 * 0.55)
            left_bottom = left_fit[0] * y_bottom + left_fit[1]
            right_bottom = right_fit[0] * y_bottom + right_fit[1]
            left_top = left_fit[0] * y_top + left_fit[1]
            right_top = right_fit[0] * y_top + right_fit[1]

            width_bottom = right_bottom - left_bottom
            width_top = right_top - left_top

            if width_bottom > 0 and width_top > width_bottom * 1.5:
                score += 0.20
                reasons.append("lanes_diverging_top")

        # ── TEMPORAL SMOOTHING ───────────────────────────────────────
        self.fork_score_history.append(score)
        smoothed_score = np.mean(list(self.fork_score_history))

        # ── DECISION ─────────────────────────────────────────────────
        self.fork_confidence = smoothed_score

        # Need sustained high score (not just one noisy frame)
        recent_high = sum(1 for s in list(self.fork_score_history)[-4:]
                         if s > 0.4) >= 2

        if smoothed_score > 0.45 and recent_high:
            self.is_fork_detected = True
            self.cooldown = self.cooldown_frames
            self.frames_since_fork = 0
            if reasons:
                print(f"🔀 FORK DETECTED (score={smoothed_score:.2f}) — {', '.join(reasons)}")
        else:
            self.is_fork_detected = False
            self.frames_since_fork += 1

        return self.is_fork_detected

    def get_fork_direction_hint(self, left_confidence, right_confidence):
        """
        When a fork is detected, hint which side the new path is on.
        The side with DROPPING confidence is the side that's forking away.

        Returns: "LEFT" | "RIGHT" | "UNKNOWN"
        """
        if left_confidence < right_confidence - 0.3:
            return "LEFT"    # left lane is fading = fork is to the left
        elif right_confidence < left_confidence - 0.3:
            return "RIGHT"   # right lane is fading = fork is to the right
        return "UNKNOWN"

    def reset(self):
        """Reset detector state."""
        self.line_count_history.clear()
        self.lane_width_history.clear()
        self.fork_score_history.clear()
        self.is_fork_detected = False
        self.fork_confidence = 0.0
        self.frames_since_fork = 0
        self.cooldown = 0
        self.calibrated = False
        self._calibration_frames = 0
        self._calibration_line_counts = []
        self._calibration_widths = []
