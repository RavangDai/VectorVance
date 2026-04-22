"""
perception.py — Single center-line detection via sliding-window + degree-2 polynomial.
White tape on dark floor. Draws the actual curve shape, not a straight line.
"""

import cv2
import numpy as np
from collections import deque


class SmoothValue:
    def __init__(self, initial=0.0, alpha=0.15):
        self.value = initial
        self.alpha = alpha

    def update(self, target):
        self.value += self.alpha * (target - self.value)
        return self.value

    def set_immediate(self, val):
        self.value = val


class LaneDetector:
    def __init__(self, width=640, height=480):
        self.width  = width
        self.height = height

        # EMA over 3-element polynomial [A, B, C] where x = A*y² + B*y + C
        self.ema_alpha  = 0.25
        self.ema_fit    = None
        self.confidence = 0.0

        self.error_history = deque(maxlen=5)

        self.smooth_center    = SmoothValue(width // 2, alpha=0.15)
        self.smooth_error     = SmoothValue(0.0,        alpha=0.15)
        self.smooth_conf_disp = SmoothValue(0.0,        alpha=0.08)

        self._direction_label = "STRAIGHT"
        self._direction_color = (0, 255, 0)
        self._direction_hold  = 0

        # ROI trapezoid — bottom 55%, wide top to match 130° FOV
        roi_bottom = height
        roi_top    = int(height * 0.45)
        roi_twf    = 0.70

        self.roi_vertices = np.array([[
            (int(width * 0.01),               roi_bottom),
            (int(width * (0.5 - roi_twf / 2)), roi_top),
            (int(width * (0.5 + roi_twf / 2)), roi_top),
            (int(width * 0.99),               roi_bottom),
        ]], dtype=np.int32)

        # Sliding window
        self._n_windows = 9
        self._margin    = 70
        self._min_pix   = 20

    # ── White tape mask ───────────────────────────────────────────────
    def _extract_white(self, frame):
        hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
        white_mask = cv2.inRange(hls,
            np.array([0,   170,  0], dtype=np.uint8),
            np.array([255, 255, 50], dtype=np.uint8))

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, bright_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

        combined = cv2.bitwise_or(white_mask, bright_mask)
        kernel   = np.ones((3, 3), np.uint8)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        combined = cv2.dilate(combined, kernel, iterations=1)
        return combined

    # ── ROI ───────────────────────────────────────────────────────────
    def _apply_roi(self, binary):
        mask = np.zeros_like(binary)
        cv2.fillPoly(mask, self.roi_vertices, 255)
        return cv2.bitwise_and(binary, mask)

    # ── Sliding window — single center line ───────────────────────────
    def _find_line_pixels(self, binary):
        h = self.height

        # Find tape starting x from histogram of bottom 40%
        histogram = np.sum(binary[int(h * 0.60):, :], axis=0)
        peak_x    = int(np.argmax(histogram))

        if histogram[peak_x] < 30:
            return None, None

        roi_height    = int(h * 0.55)
        window_height = max(1, roi_height // self._n_windows)

        nonzeroy = np.asarray(binary.nonzero()[0], dtype=np.int32)
        nonzerox = np.asarray(binary.nonzero()[1], dtype=np.int32)

        xs, ys = [], []
        cur_x  = peak_x

        for w in range(self._n_windows):
            y_low  = h - (w + 1) * window_height
            y_high = h - w * window_height

            good = np.where(
                (nonzeroy >= y_low)  & (nonzeroy < y_high) &
                (nonzerox >= cur_x - self._margin) &
                (nonzerox <  cur_x + self._margin)
            )[0]

            xs.extend(nonzerox[good])
            ys.extend(nonzeroy[good])

            if len(good) > self._min_pix:
                cur_x = int(np.mean(nonzerox[good]))

        xs = np.array(xs, dtype=np.float32)
        ys = np.array(ys, dtype=np.float32)
        return (xs, ys) if len(xs) >= 40 else (None, None)

    # ── Degree-2 polynomial fit ───────────────────────────────────────
    def _fit_poly(self, xs, ys):
        if xs is None:
            return None
        try:
            fit = np.polyfit(ys, xs, 2)
        except np.linalg.LinAlgError:
            return None
        # Sanity: bottom-of-frame x must land within frame bounds
        x_bot = fit[0] * self.height**2 + fit[1] * self.height + fit[2]
        if not (-self.width * 0.5 < x_bot < self.width * 1.5):
            return None
        return fit

    def _eval_poly(self, fit, y):
        if fit is None:
            return None
        return float(fit[0] * y**2 + fit[1] * y + fit[2])

    def _make_curve_points(self, fit):
        """Sample 30 points along the polynomial within the ROI for polylines drawing."""
        if fit is None:
            return None
        ys = np.linspace(self.height, int(self.height * 0.45), 30)
        xs = fit[0] * ys**2 + fit[1] * ys + fit[2]
        pts = [(int(x), int(y)) for x, y in zip(xs, ys) if 0 <= x < self.width]
        return pts if len(pts) >= 5 else None

    # ── EMA + confidence ──────────────────────────────────────────────
    def _update_ema(self, new_fit):
        if new_fit is None:
            if self.confidence <= 0.1:
                self.ema_fit = None
            return
        alpha = self.ema_alpha * (0.5 + 0.5 * self.confidence)
        if self.ema_fit is None:
            self.ema_fit = new_fit
        else:
            self.ema_fit = self.ema_fit * (1.0 - alpha) + new_fit * alpha

    def _update_confidence(self, fit):
        if fit is not None:
            self.confidence = min(1.0, self.confidence + 0.20)
        else:
            self.confidence = max(0.0, self.confidence - 0.05)

    # ── Main pipeline ─────────────────────────────────────────────────
    def process_frame(self, frame):
        frame  = cv2.resize(frame, (self.width, self.height))
        binary = self._apply_roi(self._extract_white(frame))

        xs, ys  = self._find_line_pixels(binary)
        raw_fit = self._fit_poly(xs, ys)

        self._update_confidence(raw_fit)
        self._update_ema(raw_fit)

        if self.ema_fit is None:
            return None, self._draw_warning_overlay(frame, "LOST LANE")

        # Steering: error = how far the line is from frame center
        control_y = int(self.height * 0.75)
        line_x    = self._eval_poly(self.ema_fit, control_y)
        raw_error = int(self.width // 2 - line_x) if line_x is not None else 0

        self.error_history.append(raw_error)
        weights = np.linspace(0.5, 1.0, len(self.error_history))
        steering_error = int(np.average(list(self.error_history), weights=weights))

        display_x   = int(self.smooth_center.update(line_x if line_x else self.width // 2))
        display_err = int(self.smooth_error.update(float(steering_error)))

        debug = self._draw_overlay(frame, binary, display_x, display_err)
        return steering_error, debug

    # ── Direction label ───────────────────────────────────────────────
    def _get_direction(self, error):
        abs_err = abs(error)
        if abs_err < 8:
            new_dir, new_col = "STRAIGHT", (0, 255, 0)
        elif error > 15:
            new_dir, new_col = "LEFT",     (0, 165, 255)
        elif error < -15:
            new_dir, new_col = "RIGHT",    (255, 0, 255)
        else:
            return self._direction_label, self._direction_color

        if new_dir != self._direction_label:
            self._direction_hold += 1
            if self._direction_hold >= 4:
                self._direction_label = new_dir
                self._direction_color = new_col
                self._direction_hold  = 0
        else:
            self._direction_hold = 0

        return self._direction_label, self._direction_color

    # ── Drawing ───────────────────────────────────────────────────────
    def _draw_overlay(self, frame, binary, line_x, error):
        debug = frame.copy()
        h, w  = debug.shape[:2]

        # FOV boundary
        cv2.polylines(debug, self.roi_vertices, isClosed=True,
                      color=(0, 220, 255), thickness=1)

        # Curved lane line
        pts = self._make_curve_points(self.ema_fit)
        if pts:
            arr = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(debug, [arr], isClosed=False, color=(0, 255, 80),   thickness=8)
            cv2.polylines(debug, [arr], isClosed=False, color=(200, 255, 200), thickness=2)

        # Frame center (gold) and line position (red)
        cx = w // 2
        cv2.line(debug, (cx,     h - 120), (cx,     h), (255, 150, 0), 2)
        cv2.line(debug, (line_x, h - 120), (line_x, h), (0, 0, 255),   3)

        # Error bar
        cv2.line(debug,   (cx, h - 60), (line_x, h - 60), (0, 0, 255), 2)
        cv2.circle(debug, (line_x, h - 60), 6, (0, 0, 255), 2)
        cv2.circle(debug, (cx,     h - 60), 6, (255, 150, 0), 2)

        # Direction banner
        direction, dir_color = self._get_direction(error)
        arrow = "<--" if direction == "LEFT" else "-->" if direction == "RIGHT" else "| |"
        cv2.rectangle(debug, (135, 8), (420, 40), (0, 0, 0),     -1)
        cv2.rectangle(debug, (135, 8), (420, 40), dir_color,      1)
        cv2.putText(debug, f"{abs(error)}px {arrow} {direction}",
                    (145, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, dir_color, 2)

        # Binary mask minimap (top-left)
        mini = cv2.resize(cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR), (120, 90))
        debug[10:100, 10:130] = mini
        cv2.rectangle(debug, (10, 10), (130, 100), (100, 100, 100), 1)
        cv2.putText(debug, "Mask", (12, 112),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 120, 120), 1)

        # Confidence bar (top-right)
        self._draw_confidence_bar(debug)

        return debug

    def _draw_confidence_bar(self, frame):
        h, w  = frame.shape[:2]
        conf  = self.smooth_conf_disp.update(self.confidence)
        bx    = w - 38
        by    = 115
        bar_h = 70
        color = (0, 230, 100) if conf > 0.5 else (0, 180, 255) if conf > 0.25 else (0, 0, 255)
        cv2.rectangle(frame, (bx, by - bar_h), (bx + 22, by), (30, 30, 30), -1)
        fill = int(bar_h * conf)
        if fill > 0:
            cv2.rectangle(frame, (bx, by - fill), (bx + 22, by), color, -1)
        cv2.rectangle(frame, (bx, by - bar_h), (bx + 22, by), (80, 80, 80), 1)
        cv2.putText(frame, "C", (bx + 6, by - bar_h - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
        cv2.putText(frame, f"{conf:.2f}", (bx - 2, by + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)

    def _draw_warning_overlay(self, frame, message):
        debug   = frame.copy()
        overlay = np.zeros_like(debug, dtype=np.uint8)
        cv2.rectangle(overlay, (0, 0), (self.width, self.height), (0, 0, 255), -1)
        cv2.addWeighted(overlay, 0.4, debug, 0.6, 0, debug)
        cv2.putText(debug, message, (20, self.height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
        return debug

    def reset_smoothing(self):
        self.ema_fit    = None
        self.confidence = 0.0
        self.error_history.clear()
        self.smooth_center.set_immediate(self.width // 2)
        self.smooth_error.set_immediate(0.0)
        self.smooth_conf_disp.set_immediate(0.0)
        self._direction_label = "STRAIGHT"
        self._direction_hold  = 0
