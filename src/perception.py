"""
perception.py - Lane detection optimized for Raspberry Pi 3.

Runs at 480x360 to keep CPU usage reasonable on the Pi.
Uses separate left/right line fitting instead of a single median,
with a memory fallback when one side drops out mid-curve.
"""

import cv2
import numpy as np
from collections import deque


class LaneDetector:
    def __init__(self, width=480, height=360, smoothing_window=5):
        self.width = width
        self.height = height
        self.smoothing_window = smoothing_window
        self.error_history = deque(maxlen=smoothing_window)

        # trapezoid ROI - wider top than a simple triangle to catch curves
        roi_bottom = height
        roi_top = int(height * 0.55)
        roi_top_width_fraction = 0.55

        self.roi_vertices = np.array([[
            (0, roi_bottom),
            (int(width * (0.5 - roi_top_width_fraction / 2)), roi_top),
            (int(width * (0.5 + roi_top_width_fraction / 2)), roi_top),
            (width, roi_bottom)
        ]], dtype=np.int32)

        self.canny_low = 50
        self.canny_high = 150
        self.hough_threshold = 30
        self.hough_min_line_length = 40
        self.hough_max_line_gap = 150
        self.min_segment_length = 30

        # last known lane positions - used as fallback when detection fails
        self.left_lane_memory  = deque(maxlen=8)
        self.right_lane_memory = deque(maxlen=8)

        self.left_confidence  = 0.0
        self.right_confidence = 0.0

    def process_frame(self, frame):
        frame = cv2.resize(frame, (self.width, self.height))

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, self.canny_low, self.canny_high)
        masked_edges = self._apply_roi(edges)

        lines = cv2.HoughLinesP(
            masked_edges, rho=2, theta=np.pi / 180,
            threshold=self.hough_threshold,
            minLineLength=self.hough_min_line_length,
            maxLineGap=self.hough_max_line_gap
        )

        left_fit, right_fit = self._separate_lanes(lines)
        lane_center, left_x, right_x = self._compute_center(left_fit, right_fit)

        frame_center = self.width // 2
        raw_error = (frame_center - lane_center) if lane_center is not None else 0

        self.error_history.append(raw_error)
        steering_error = int(np.mean(self.error_history))

        debug_frame = self._draw_overlay(
            frame, lines, left_fit, right_fit,
            left_x, right_x, lane_center, steering_error, masked_edges
        )
        return steering_error, debug_frame

    def _apply_roi(self, edges):
        mask = np.zeros_like(edges)
        cv2.fillPoly(mask, self.roi_vertices, 255)
        return cv2.bitwise_and(edges, mask)

    def _separate_lanes(self, lines):
        """
        Split detected lines into left/right by slope and position.
        Angle filter is 20-80 degrees - relaxed from the original 45 cap
        so gentle curves aren't thrown away.
        """
        left_lines  = []
        right_lines = []

        if lines is None:
            return None, None

        cx = self.width / 2

        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 == x1:
                continue  # vertical line, skip

            seg_len = np.hypot(x2 - x1, y2 - y1)
            if seg_len < self.min_segment_length:
                continue  # too short, probably noise

            slope = (y2 - y1) / (x2 - x1)
            angle = abs(np.degrees(np.arctan(slope)))

            if angle < 20 or angle > 80:
                continue

            mid_x = (x1 + x2) / 2

            if slope < 0 and mid_x < cx:
                left_lines.append(line[0])
            elif slope > 0 and mid_x > cx:
                right_lines.append(line[0])

        left_fit  = self._fit_lane(left_lines)
        right_fit = self._fit_lane(right_lines)
        return left_fit, right_fit

    def _fit_lane(self, line_segments):
        """Fit a single line to a set of segments. Returns (slope, intercept) or None."""
        if not line_segments:
            return None
        pts = []
        for x1, y1, x2, y2 in line_segments:
            pts.extend([(x1, y1), (x2, y2)])
        xs = np.array([p[0] for p in pts], dtype=np.float32)
        ys = np.array([p[1] for p in pts], dtype=np.float32)
        try:
            # fit x = f(y) rather than y = f(x) - more numerically stable for near-vertical lines
            fit = np.polyfit(ys, xs, 1)
            return fit
        except np.linalg.LinAlgError:
            return None

    def _compute_center(self, left_fit, right_fit):
        """
        Compute lane center from fitted lines.
        Falls back to memory when one side is missing.
        """
        y_eval = int(self.height * 0.75)

        left_x  = self._eval_fit(left_fit,  y_eval)
        right_x = self._eval_fit(right_fit, y_eval)

        if left_x is not None:
            self.left_lane_memory.append(left_x)
            self.left_confidence = min(1.0, self.left_confidence + 0.2)
        else:
            self.left_confidence = max(0.0, self.left_confidence - 0.15)

        if right_x is not None:
            self.right_lane_memory.append(right_x)
            self.right_confidence = min(1.0, self.right_confidence + 0.2)
        else:
            self.right_confidence = max(0.0, self.right_confidence - 0.15)

        if left_x is None and self.left_lane_memory:
            left_x = int(np.mean(self.left_lane_memory))
        if right_x is None and self.right_lane_memory:
            right_x = int(np.mean(self.right_lane_memory))

        if left_x is not None and right_x is not None:
            center = (left_x + right_x) // 2
        elif left_x is not None:
            # only left lane visible - estimate right side assuming ~250px lane width
            center = left_x + 125
        elif right_x is not None:
            center = right_x - 125
        else:
            center = None

        return center, left_x, right_x

    def _eval_fit(self, fit, y):
        """Evaluate x = slope*y + intercept. Returns None if fit is None or result is out of frame."""
        if fit is None:
            return None
        x = int(fit[0] * y + fit[1])
        if x < -50 or x > self.width + 50:
            return None
        return x

    def _make_lane_points(self, fit):
        """Return (x_bottom, x_top) for drawing a fitted lane line."""
        if fit is None:
            return None
        y_bottom = self.height
        y_top = int(self.height * 0.55)
        x_bottom = int(fit[0] * y_bottom + fit[1])
        x_top    = int(fit[0] * y_top    + fit[1])
        return (x_bottom, y_bottom), (x_top, y_top)

    def _draw_overlay(self, frame, raw_lines, left_fit, right_fit,
                      left_x, right_x, lane_center, error, edges):
        debug = frame.copy()

        left_pts  = self._make_lane_points(left_fit)
        right_pts = self._make_lane_points(right_fit)

        if left_pts:
            cv2.line(debug, left_pts[0], left_pts[1], (0, 255, 255), 4)   # cyan
        if right_pts:
            cv2.line(debug, right_pts[0], right_pts[1], (255, 128, 0), 4) # orange

        if left_pts and right_pts:
            poly = np.array([
                left_pts[0], left_pts[1], right_pts[1], right_pts[0]
            ], dtype=np.int32)
            overlay = debug.copy()
            cv2.fillPoly(overlay, [poly], (0, 255, 0))
            cv2.addWeighted(overlay, 0.15, debug, 0.85, 0, debug)

        frame_center = self.width // 2
        cv2.line(debug, (frame_center, 0), (frame_center, self.height), (255, 0, 0), 2)
        cv2.putText(debug, "Target", (frame_center + 10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

        if lane_center is not None:
            cv2.line(debug, (lane_center, 0), (lane_center, self.height), (0, 0, 255), 2)
            cv2.putText(debug, f"Lane: {lane_center}px",
                        (lane_center + 10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        cv2.polylines(debug, self.roi_vertices, True, (0, 255, 255), 1)

        abs_err = abs(error)
        if error > 10:
            direction, color = "LEFT",     (0, 165, 255)
        elif error < -10:
            direction, color = "RIGHT",    (255, 0, 255)
        else:
            direction, color = "STRAIGHT", (0, 255, 0)

        cv2.putText(debug, f"{abs_err}px -> {direction}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        self._draw_confidence_bars(debug)

        # small edge preview in top-left corner
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        edges_small   = cv2.resize(edges_colored, (120, 90))
        debug[10:100, 10:130] = edges_small

        return debug

    def _draw_confidence_bars(self, frame):
        h, w = frame.shape[:2]
        bar_w, bar_h = 30, 150
        lx = w - 110
        rx = w - 55
        by = h - 60

        for (bx, conf, label) in [(lx, self.left_confidence, "L"), (rx, self.right_confidence, "R")]:
            cv2.rectangle(frame, (bx, by - bar_h), (bx + bar_w, by), (50, 50, 50), -1)
            fill = int(bar_h * conf)
            color = (0, 165, 255) if conf > 0.5 else (0, 0, 200)
            if fill > 0:
                cv2.rectangle(frame, (bx, by - fill), (bx + bar_w, by), color, -1)
            cv2.putText(frame, label, (bx + 7, by - bar_h - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            cv2.putText(frame, f"{conf:.2f}", (bx - 2, by + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    def reset_smoothing(self):
        self.error_history.clear()
        self.left_lane_memory.clear()
        self.right_lane_memory.clear()
        self.left_confidence  = 0.0
        self.right_confidence = 0.0
