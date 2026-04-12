"""
perception.py - Premium "Tesla-Style" Lane Detection
Optimized for Raspberry Pi 3.
Features: Color-based lane filtering, EMA smoothing, neon-glow UI, safety cutoffs.
FIXED: Removed hardcoded 180° rotation — caller (mainv2.py) handles rotation
       based on source type (webcam vs video file).
"""

import cv2
import numpy as np
from collections import deque


class SmoothValue:
    """Utility class for smoothly interpolating any numeric value over time."""
    def __init__(self, initial=0.0, alpha=0.15):
        self.value = initial
        self.alpha = alpha

    def update(self, target):
        self.value += self.alpha * (target - self.value)
        return self.value

    def set_immediate(self, val):
        self.value = val


class LaneDetector:
    def __init__(self, width=640, height=480, smoothing_window=7):
        self.width = width
        self.height = height

        # ── TESLA-STYLE SMOOTHING (EMA) ──────────────────────────────
        self.ema_alpha = 0.10
        self.ema_left_fit = None
        self.ema_right_fit = None

        self.smoothing_window = smoothing_window
        self.error_history = deque(maxlen=smoothing_window)

        # ── SMOOTH UI VALUES ─────────────────────────────────────────
        self.smooth_lane_center = SmoothValue(width // 2, alpha=0.12)
        self.smooth_error = SmoothValue(0.0, alpha=0.12)
        self.smooth_left_conf = SmoothValue(0.0, alpha=0.08)
        self.smooth_right_conf = SmoothValue(0.0, alpha=0.08)

        # ── DIRECTION HYSTERESIS ─────────────────────────────────────
        self._direction_label = "STRAIGHT"
        self._direction_color = (0, 255, 0)
        self._direction_hold = 0

        # ── ROI — trapezoid covering bottom 55% of frame ────────────
        roi_bottom = height
        roi_top = int(height * 0.45)
        roi_top_width_fraction = 0.60

        self.roi_vertices = np.array([[
            (int(width * 0.02), roi_bottom),
            (int(width * (0.5 - roi_top_width_fraction / 2)), roi_top),
            (int(width * (0.5 + roi_top_width_fraction / 2)), roi_top),
            (int(width * 0.98), roi_bottom)
        ]], dtype=np.int32)

        # ── VISION TUNING ────────────────────────────────────────────
        self.canny_low = 50
        self.canny_high = 150
        self.hough_threshold = 35
        self.hough_min_line_length = 35
        self.hough_max_line_gap = 120
        self.min_segment_length = 35

        self.left_confidence = 0.0
        self.right_confidence = 0.0

        # ── LANE FIT VALIDATION ──────────────────────────────────────
        self._prev_left_slope = None
        self._prev_right_slope = None
        self._max_slope_change = 0.30   # allow reasonable inter-frame slope variation

        # ── CACHED DRAW POINTS ───────────────────────────────────────
        self._prev_left_pts = None
        self._prev_right_pts = None

    # ─────────────────────────────────────────────────────────────────
    #  COLOR-BASED LANE MASK
    # ─────────────────────────────────────────────────────────────────
    def _extract_lane_colors(self, frame):
        """
        Extract white and yellow lane markings using HSV + HLS color spaces.
        Dramatically reduces false detections from non-lane objects.
        """
        # --- WHITE LANES (HLS: high lightness + low saturation) ---
        hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
        white_lower = np.array([0,   160, 0],  dtype=np.uint8)
        white_upper = np.array([255, 255, 70], dtype=np.uint8)
        white_mask  = cv2.inRange(hls, white_lower, white_upper)

        # --- YELLOW LANES (HSV: hue range + saturation) ---
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        yellow_lower = np.array([15,  80, 120], dtype=np.uint8)
        yellow_upper = np.array([35, 255, 255], dtype=np.uint8)
        yellow_mask  = cv2.inRange(hsv, yellow_lower, yellow_upper)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Bright-on-dark contrast mask (catches white tape on dark floors)
        _, bright_mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

        # Combine
        combined = cv2.bitwise_or(white_mask, yellow_mask)
        combined = cv2.bitwise_or(combined, bright_mask)

        # Cleanup
        kernel = np.ones((3, 3), np.uint8)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        combined = cv2.dilate(combined, kernel, iterations=1)

        return combined, gray

    # ─────────────────────────────────────────────────────────────────
    #  FAILSAFE HELPERS
    # ─────────────────────────────────────────────────────────────────
    def _draw_warning_overlay(self, frame, message):
        debug = frame.copy()
        overlay = np.zeros_like(debug, dtype=np.uint8)
        cv2.rectangle(overlay, (0, 0), (self.width, self.height), (0, 0, 255), -1)
        cv2.addWeighted(overlay, 0.4, debug, 0.6, 0, debug)
        cv2.putText(debug, message, (20, self.height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
        return debug

    # ─────────────────────────────────────────────────────────────────
    #  MAIN PIPELINE
    # ─────────────────────────────────────────────────────────────────
    def process_frame(self, frame):
        """
        Process a single frame for lane detection.
        IMPORTANT: Frame must already be correctly oriented BEFORE calling this.
                   Rotation is handled by the caller (mainv2.py) based on source type.
        """
        frame = cv2.resize(frame, (self.width, self.height))
        # NOTE: No rotation here — the caller rotates if needed (webcam/Pi camera)

        # ── COLOR EXTRACTION ─────────────────────────────────────────
        color_mask, gray = self._extract_lane_colors(frame)

        # Apply color mask BEFORE edge detection
        masked_gray = cv2.bitwise_and(gray, color_mask)
        blur = cv2.GaussianBlur(masked_gray, (5, 5), 0)
        edges = cv2.Canny(blur, self.canny_low, self.canny_high)
        masked_edges = self._apply_roi(edges)

        lines = cv2.HoughLinesP(
            masked_edges, rho=2, theta=np.pi / 180,
            threshold=self.hough_threshold,
            minLineLength=self.hough_min_line_length,
            maxLineGap=self.hough_max_line_gap
        )

        raw_left_fit, raw_right_fit = self._separate_lanes(lines)

        # ── VALIDATE FITS ────────────────────────────────────────────
        raw_left_fit = self._validate_fit(raw_left_fit, self._prev_left_slope)
        raw_right_fit = self._validate_fit(raw_right_fit, self._prev_right_slope)
        if raw_left_fit is not None:
            self._prev_left_slope = raw_left_fit[0]
        if raw_right_fit is not None:
            self._prev_right_slope = raw_right_fit[0]

        self._update_confidence(raw_left_fit, raw_right_fit)

        # ── FAILSAFE 2: LOST LANES ───────────────────────────────────
        # Only trigger if BOTH lanes are lost (AND), not just one
        if self.left_confidence < 0.15 and self.right_confidence < 0.15:
            self.ema_left_fit = None
            self.ema_right_fit = None
            return None, self._draw_warning_overlay(frame, "WARNING: LOST LANES")

        # EMA Smoothing
        self.ema_left_fit = self._update_ema(self.ema_left_fit, raw_left_fit, self.left_confidence)
        self.ema_right_fit = self._update_ema(self.ema_right_fit, raw_right_fit, self.right_confidence)

        # Steering error
        lane_center, left_x, right_x = self._compute_center(self.ema_left_fit, self.ema_right_fit)
        frame_center = self.width // 2
        raw_error = (frame_center - lane_center) if lane_center is not None else 0

        self.error_history.append(raw_error)
        weights = np.linspace(0.5, 1.0, len(self.error_history))
        steering_error = int(np.average(list(self.error_history), weights=weights))

        # Smooth display values
        display_center = self.smooth_lane_center.update(
            lane_center if lane_center is not None else frame_center
        )
        display_error = self.smooth_error.update(float(steering_error))

        # Draw overlay
        debug_frame = self._draw_premium_overlay(
            frame, self.ema_left_fit, self.ema_right_fit,
            int(display_center), int(display_error), masked_edges
        )

        return steering_error, debug_frame

    # ─────────────────────────────────────────────────────────────────
    #  FIT VALIDATION
    # ─────────────────────────────────────────────────────────────────
    def _validate_fit(self, fit, prev_slope):
        if fit is None:
            return None
        slope = fit[0]
        if abs(slope) < 0.1 or abs(slope) > 5.0:
            return None
        if prev_slope is not None:
            if abs(slope - prev_slope) > self._max_slope_change:
                return None
        return fit

    # ─────────────────────────────────────────────────────────────────
    #  EMA + CONFIDENCE
    # ─────────────────────────────────────────────────────────────────
    def _update_ema(self, ema_fit, new_fit, confidence):
        if new_fit is None:
            if confidence <= 0.1:
                return None
            return ema_fit
        if ema_fit is None:
            return new_fit
        adaptive_alpha = self.ema_alpha * (0.5 + 0.5 * confidence)
        return ema_fit * (1.0 - adaptive_alpha) + new_fit * adaptive_alpha

    def _update_confidence(self, left_fit, right_fit):
        if left_fit is not None:
            self.left_confidence = min(1.0, self.left_confidence + 0.15)
        else:
            self.left_confidence = max(0.0, self.left_confidence - 0.05)
        if right_fit is not None:
            self.right_confidence = min(1.0, self.right_confidence + 0.15)
        else:
            self.right_confidence = max(0.0, self.right_confidence - 0.05)

    # ─────────────────────────────────────────────────────────────────
    #  GEOMETRY
    # ─────────────────────────────────────────────────────────────────
    def _apply_roi(self, edges):
        mask = np.zeros_like(edges)
        cv2.fillPoly(mask, self.roi_vertices, 255)
        return cv2.bitwise_and(edges, mask)

    def _separate_lanes(self, lines):
        left_lines = []
        right_lines = []
        if lines is None:
            return None, None

        cx = self.width / 2
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 == x1:
                continue
            seg_len = np.hypot(x2 - x1, y2 - y1)
            if seg_len < self.min_segment_length:
                continue
            slope = (y2 - y1) / (x2 - x1)
            angle = abs(np.degrees(np.arctan(slope)))

            if angle < 20 or angle > 75:
                continue

            mid_x = (x1 + x2) / 2
            if slope < 0 and mid_x < cx:
                left_lines.append(line[0])
            elif slope > 0 and mid_x > cx:
                right_lines.append(line[0])

        return self._fit_lane(left_lines), self._fit_lane(right_lines)

    def _fit_lane(self, line_segments):
        if not line_segments or len(line_segments) < 1:
            return None
        pts = []
        for x1, y1, x2, y2 in line_segments:
            pts.extend([(x1, y1), (x2, y2)])
        xs = np.array([p[0] for p in pts], dtype=np.float32)
        ys = np.array([p[1] for p in pts], dtype=np.float32)
        try:
            fit = np.polyfit(ys, xs, 1)
            predicted = np.polyval(fit, ys)
            residual = np.mean(np.abs(xs - predicted))
            if residual > 25:   # tightened: reject noisy/scattered fits
                return None
            return fit
        except np.linalg.LinAlgError:
            return None

    def _compute_center(self, left_fit, right_fit):
        y_eval = int(self.height * 0.75)
        left_x = self._eval_fit(left_fit, y_eval)
        right_x = self._eval_fit(right_fit, y_eval)

        if left_x is not None and right_x is not None:
            if right_x - left_x < 80:
                return None, left_x, right_x
            center = (left_x + right_x) // 2
        elif left_x is not None:
            center = left_x + 160
        elif right_x is not None:
            center = right_x - 160
        else:
            center = None

        return center, left_x, right_x

    def _eval_fit(self, fit, y):
        if fit is None:
            return None
        x = int(fit[0] * y + fit[1])
        if x < -300 or x > self.width + 300:
            return None
        return x

    def _make_lane_points(self, fit):
        if fit is None:
            return None
        y_bottom = self.height
        y_top = int(self.height * 0.45)   # matches ROI top
        x_bottom = int(fit[0] * y_bottom + fit[1])
        x_top = int(fit[0] * y_top + fit[1])
        return (x_bottom, y_bottom), (x_top, y_top)

    def _smooth_points(self, new_pts, prev_pts, alpha=0.2):
        if prev_pts is None or new_pts is None:
            return new_pts
        p0 = (int(prev_pts[0][0] + alpha * (new_pts[0][0] - prev_pts[0][0])),
              int(prev_pts[0][1] + alpha * (new_pts[0][1] - prev_pts[0][1])))
        p1 = (int(prev_pts[1][0] + alpha * (new_pts[1][0] - prev_pts[1][0])),
              int(prev_pts[1][1] + alpha * (new_pts[1][1] - prev_pts[1][1])))
        return p0, p1

    # ─────────────────────────────────────────────────────────────────
    #  DIRECTION LABEL WITH HYSTERESIS
    # ─────────────────────────────────────────────────────────────────
    def _get_direction(self, error):
        abs_err = abs(error)
        if abs_err < 8:
            new_dir, new_col = "STRAIGHT", (0, 255, 0)
        elif error > 15:
            new_dir, new_col = "LEFT", (0, 165, 255)
        elif error < -15:
            new_dir, new_col = "RIGHT", (255, 0, 255)
        else:
            return self._direction_label, self._direction_color

        if new_dir != self._direction_label:
            self._direction_hold += 1
            if self._direction_hold >= 4:
                self._direction_label = new_dir
                self._direction_color = new_col
                self._direction_hold = 0
        else:
            self._direction_hold = 0

        return self._direction_label, self._direction_color

    # ─────────────────────────────────────────────────────────────────
    #  DRAWING
    # ─────────────────────────────────────────────────────────────────
    def _draw_premium_overlay(self, frame, left_fit, right_fit, lane_center, error, edges):
        debug = frame.copy()
        h, w = debug.shape[:2]
        overlay = np.zeros_like(debug, dtype=np.uint8)

        raw_left_pts = self._make_lane_points(left_fit)
        raw_right_pts = self._make_lane_points(right_fit)

        left_pts = self._smooth_points(raw_left_pts, self._prev_left_pts, alpha=0.25)
        right_pts = self._smooth_points(raw_right_pts, self._prev_right_pts, alpha=0.25)
        self._prev_left_pts = left_pts
        self._prev_right_pts = right_pts

        # ── LANE POLYGON (green tinted driving corridor) ─────────────
        if left_pts and right_pts:
            poly = np.array([left_pts[0], left_pts[1], right_pts[1], right_pts[0]], dtype=np.int32)
            cv2.fillPoly(overlay, [poly], (0, 200, 80))
            cv2.addWeighted(overlay, 0.25, debug, 1.0, 0, debug)

        # ── NEON LANE LINES (green glow + white core) ────────────────
        def draw_neon_line(img, p1, p2, glow_color):
            if p1 and p2:
                cv2.line(img, p1, p2, glow_color, 10)
                cv2.line(img, p1, p2, (200, 255, 200), 2)

        draw_neon_line(debug,
                       left_pts[0] if left_pts else None,
                       left_pts[1] if left_pts else None,
                       (0, 255, 100))
        draw_neon_line(debug,
                       right_pts[0] if right_pts else None,
                       right_pts[1] if right_pts else None,
                       (0, 255, 100))

        # ── CENTER LINES (frame center = blue, lane center = red) ────
        frame_center = w // 2
        cv2.line(debug, (frame_center, h - 100),
                 (frame_center, h), (255, 150, 0), 2)
        if lane_center is not None:
            cv2.line(debug, (lane_center, h - 100),
                     (lane_center, h), (0, 0, 255), 3)
            # Horizontal error line
            cv2.line(debug, (frame_center, h - 50),
                     (lane_center, h - 50), (0, 0, 255), 2)
            # Target crosshair
            cv2.circle(debug, (lane_center, h - 50), 6, (0, 0, 255), 2)
            cv2.circle(debug, (frame_center, h - 50), 6, (255, 150, 0), 2)

        # ── DIRECTION INDICATOR (top area with arrow) ────────────────
        direction, dir_color = self._get_direction(error)
        abs_err = abs(error)
        arrow = "<--" if direction == "LEFT" else "-->" if direction == "RIGHT" else "| |"

        # Dark banner behind direction text
        cv2.rectangle(debug, (135, 8), (420, 40), (0, 0, 0), -1)
        cv2.rectangle(debug, (135, 8), (420, 40), dir_color, 1)
        cv2.putText(debug, f"{abs_err}px {arrow} {direction}",
                    (145, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, dir_color, 2)

        # Lane center value
        if lane_center is not None:
            cv2.putText(debug, f"Lane: {lane_center}px",
                        (145, 58), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 180, 255), 1)

        # ── EDGE MINIMAP (top-left) ──────────────────────────────────
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        edges_small = cv2.resize(edges_colored, (120, 90))
        debug[10:100, 10:130] = edges_small
        cv2.rectangle(debug, (10, 10), (130, 100), (100, 100, 100), 1)
        cv2.putText(debug, "Edge View", (12, 112),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 120, 120), 1)

        # ── CONFIDENCE BARS (right side) ─────────────────────────────
        self._draw_confidence_bars(debug)

        return debug

    def _draw_confidence_bars(self, frame):
        disp_left = self.smooth_left_conf.update(self.left_confidence)
        disp_right = self.smooth_right_conf.update(self.right_confidence)

        h, w = frame.shape[:2]
        bar_w, bar_h = 30, 150
        lx = w - 110
        rx = w - 55
        by = h - 60

        for (bx, conf, label) in [(lx, disp_left, "L"), (rx, disp_right, "R")]:
            # Background
            cv2.rectangle(frame, (bx, by - bar_h), (bx + bar_w, by), (30, 30, 30), -1)
            fill = int(bar_h * conf)
            # Color: green > 0.5, orange 0.25-0.5, red < 0.25
            if conf > 0.5:
                color = (0, 230, 100)
            elif conf > 0.25:
                color = (0, 180, 255)
            else:
                color = (0, 0, 255)
            if fill > 0:
                cv2.rectangle(frame, (bx, by - fill), (bx + bar_w, by), color, -1)
            cv2.rectangle(frame, (bx, by - bar_h), (bx + bar_w, by), (80, 80, 80), 1)
            # Label
            cv2.putText(frame, label, (bx + 8, by - bar_h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            # Value
            cv2.putText(frame, f"{conf:.2f}",
                        (bx - 2, by + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    def reset_smoothing(self):
        self.error_history.clear()
        self.ema_left_fit = None
        self.ema_right_fit = None
        self.left_confidence = 0.0
        self.right_confidence = 0.0
        self.smooth_lane_center.set_immediate(self.width // 2)
        self.smooth_error.set_immediate(0.0)
        self.smooth_left_conf.set_immediate(0.0)
        self.smooth_right_conf.set_immediate(0.0)
        self._prev_left_pts = None
        self._prev_right_pts = None
        self._prev_left_slope = None
        self._prev_right_slope = None
        self._direction_label = "STRAIGHT"
        self._direction_hold = 0