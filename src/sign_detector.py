"""
sign_detector.py - STOP sign detection.

Detection requires all of:
  - Red color (HSV mask, two hue ranges to cover the wraparound at 0/180)
  - Octagon-like shape (6-12 sides after polygon approximation)
  - Square-ish aspect ratio (0.75-1.35)
  - High solidity (>0.75, i.e. a filled shape not an outline)
  - White pixels in the interior (>5%, representing the "STOP" text)
  - High circularity (>0.55, octagons are fairly round)
  - Detected in at least 2 consecutive frames to filter single-frame noise
"""

import cv2
import numpy as np
from collections import deque


class TrafficSignDetector:
    def __init__(self):
        # two hue ranges because red wraps around 0 in HSV
        self.red_lower1 = np.array([0,   50, 50])
        self.red_upper1 = np.array([10,  255, 255])
        self.red_lower2 = np.array([170, 50, 50])
        self.red_upper2 = np.array([180, 255, 255])

        self.min_sign_area  = 400
        self.max_sign_area  = 60000
        self.min_confidence = 0.45

        self.detection_history = deque(maxlen=8)
        self.min_consecutive_detections = 2

        self.detected_signs  = []
        self.confirmed_signs = []
        self.debug_mask = None

        # ROI covers most of the frame - signs can appear anywhere
        self.roi_top_fraction    = 0.0
        self.roi_bottom_fraction = 0.90

    def detect_signs(self, frame):
        h, w = frame.shape[:2]

        roi_top    = int(h * self.roi_top_fraction)
        roi_bottom = int(h * self.roi_bottom_fraction)
        roi_frame  = frame[roi_top:roi_bottom, :]

        hsv = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)
        self.debug_mask = self._get_red_mask(hsv)

        candidates = self._detect_stop_candidates(roi_frame, hsv, roi_top)
        self.detected_signs = candidates

        current_centers = [(x + w//2, y + h//2) for (_, (x, y, w, h), _) in candidates]
        self.detection_history.append(current_centers)

        self.confirmed_signs = []
        for sign_type, (x, y, w, h), conf in candidates:
            center = (x + w//2, y + h//2)
            consecutive = self._count_consecutive_near(center)

            if consecutive >= self.min_consecutive_detections:
                self.confirmed_signs.append((sign_type, (x, y, w, h), conf))

        return self.confirmed_signs

    def _count_consecutive_near(self, center, threshold=50):
        """Count recent frames that had a detection within threshold pixels of center."""
        count = 0
        for hist_centers in reversed(self.detection_history):
            found = any(
                np.sqrt((center[0] - hc[0])**2 + (center[1] - hc[1])**2) < threshold
                for hc in hist_centers
            )
            if found:
                count += 1
            else:
                break  # must be consecutive
        return count

    def _get_red_mask(self, hsv):
        mask1 = cv2.inRange(hsv, self.red_lower1, self.red_upper1)
        mask2 = cv2.inRange(hsv, self.red_lower2, self.red_upper2)
        return cv2.bitwise_or(mask1, mask2)

    def _detect_stop_candidates(self, frame, hsv, y_offset):
        mask = self._get_red_mask(hsv)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detected = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (self.min_sign_area < area < self.max_sign_area):
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            y_actual = y + y_offset

            aspect = w / max(h, 1)
            if not (0.75 < aspect < 1.35):
                continue

            peri   = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.025 * peri, True)
            sides  = len(approx)
            if not (6 <= sides <= 12):
                continue

            hull      = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity  = area / max(hull_area, 1)
            if solidity < 0.75:
                continue

            white_ratio = self._check_white_interior(frame, x, y, w, h)
            if white_ratio < 0.05:
                continue

            circularity = (4 * np.pi * area) / max(peri * peri, 1)
            if circularity < 0.55:
                continue

            confidence = self._calculate_confidence(sides, solidity, circularity, white_ratio, area)

            if confidence >= self.min_confidence:
                detected.append(("STOP", (x, y_actual, w, h), confidence))

        # occasionally log why the largest red blob failed, useful during tuning
        if len(contours) > 0 and len(detected) == 0:
            if contours:
                largest = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest)
                if area > 200:
                    import random
                    if random.random() < 0.03:
                        x, y, w, h = cv2.boundingRect(largest)
                        aspect  = w / max(h, 1)
                        peri    = cv2.arcLength(largest, True)
                        approx  = cv2.approxPolyDP(largest, 0.025 * peri, True)
                        sides   = len(approx)
                        hull    = cv2.convexHull(largest)
                        solidity = area / max(cv2.contourArea(hull), 1)
                        circ    = (4 * np.pi * area) / max(peri * peri, 1)
                        white   = self._check_white_interior(frame, x, y, w, h)
                        print(f"[sign debug] largest red blob: area={area:.0f}, aspect={aspect:.2f}, "
                              f"sides={sides}, solid={solidity:.2f}, circ={circ:.2f}, white={white:.2f}")

        return detected

    def _check_white_interior(self, frame, x, y, w, h):
        """Check for white text inside the sign bounding box. Returns ratio of bright pixels."""
        # sample the center 50% of the box to avoid the red border
        margin_x = int(w * 0.25)
        margin_y = int(h * 0.25)

        x1 = max(0, x + margin_x)
        y1 = max(0, y + margin_y)
        x2 = min(frame.shape[1], x + w - margin_x)
        y2 = min(frame.shape[0], y + h - margin_y)

        if x2 <= x1 or y2 <= y1:
            return 0.0

        center_region = frame[y1:y2, x1:x2]
        if center_region.size == 0:
            return 0.0

        gray = cv2.cvtColor(center_region, cv2.COLOR_BGR2GRAY)
        white_pixels = np.sum(gray > 160)
        return white_pixels / max(gray.size, 1)

    def _calculate_confidence(self, sides, solidity, circularity, white_ratio, area):
        side_score     = max(0.0, 1.0 - abs(sides - 8) * 0.15)  # 8 sides = perfect
        solidity_score = min(1.0, solidity / 0.9)
        circ_score     = min(1.0, circularity / 0.8)
        white_score    = min(1.0, white_ratio / 0.25)
        area_score     = 1.0 if 1500 < area < 20000 else 0.7

        conf = (side_score     * 0.25 +
                solidity_score * 0.20 +
                circ_score     * 0.15 +
                white_score    * 0.30 +
                area_score     * 0.10)

        return round(max(0.0, min(1.0, conf)), 2)

    def get_action(self):
        if self.confirmed_signs:
            return "STOP", None
        return None, None

    def draw_overlay(self, frame):
        h_frame = frame.shape[0]

        if self.debug_mask is not None:
            mask_small   = cv2.resize(self.debug_mask, (100, 75))
            mask_colored = cv2.cvtColor(mask_small, cv2.COLOR_GRAY2BGR)
            mask_colored[:, :, 2] = mask_small  # tint red channel
            frame[h_frame-85:h_frame-10, 10:110] = mask_colored
            cv2.putText(frame, "RED", (15, h_frame-88),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)

        # thin yellow box = candidate (passed checks but not yet confirmed)
        for sign_type, (x, y, w, h), conf in self.detected_signs:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 1)
            cv2.putText(frame, f"?{conf:.0%}", (x, y-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        # thick red box = confirmed STOP sign
        for sign_type, (x, y, w, h), conf in self.confirmed_signs:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 4)

            corner = 15
            for cx, cy in [(x, y), (x+w, y), (x, y+h), (x+w, y+h)]:
                dx = corner if cx == x else -corner
                dy = corner if cy == y else -corner
                cv2.line(frame, (cx, cy), (cx + dx, cy), (0, 0, 255), 4)
                cv2.line(frame, (cx, cy), (cx, cy + dy), (0, 0, 255), 4)

            label = f"STOP {conf*100:.0f}%"
            ly = y - 15 if y > 40 else y + h + 25
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(frame, (x, ly - th - 6), (x + tw + 12, ly + 6), (0, 0, 255), -1)
            cv2.putText(frame, label, (x + 6, ly),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        return frame

    def reset(self):
        self.detected_signs  = []
        self.confirmed_signs = []
        self.detection_history.clear()
        self.debug_mask = None
