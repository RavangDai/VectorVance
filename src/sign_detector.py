"""
sign_detector.py - STOP Sign Detection (STRICT)
-------------------------------------------------
VectorVance — Only detects REAL stop signs

Requirements for detection:
  1. RED color (high saturation, not washed out)
  2. OCTAGON shape (7-9 sides)
  3. Square-ish aspect ratio (0.8 - 1.2)
  4. White text inside (>15% of interior is bright)
  5. Must appear in 2+ consecutive frames
"""

import cv2
import numpy as np
from collections import deque


class TrafficSignDetector:
    def __init__(self):
        # ── Red HSV (BALANCED - catches real signs, rejects noise) ────
        self.red_lower1 = np.array([0,   50, 50])
        self.red_upper1 = np.array([10,  255, 255])
        self.red_lower2 = np.array([170, 50, 50])
        self.red_upper2 = np.array([180, 255, 255])

        # ── Thresholds ───────────────────────────────────────────────
        self.min_sign_area  = 400
        self.max_sign_area  = 60000
        self.min_confidence = 0.45

        # ── Temporal filtering ───────────────────────────────────────
        self.detection_history = deque(maxlen=8)
        self.min_consecutive_detections = 2

        self.detected_signs = []
        self.confirmed_signs = []
        self.debug_mask = None

        # ── ROI ──────────────────────────────────────────────────────
        self.roi_top_fraction = 0.0
        self.roi_bottom_fraction = 0.90

        print("🛑 STOP Sign Detector Initialized (STRICT MODE + ROTATED 180)")
        print("   Requires: Octagon + Red + White text")

    def detect_signs(self, frame):
        """Detect STOP signs with strict validation."""
        h, w = frame.shape[:2]
        
        # Apply ROI
        roi_top = int(h * self.roi_top_fraction)
        roi_bottom = int(h * self.roi_bottom_fraction)
        roi_frame = frame[roi_top:roi_bottom, :]
        roi_h = roi_bottom - roi_top
        
        # ── THE FIX: Rotate the ROI 180 degrees so the detector processes it right-side up ──
        roi_frame_rotated = cv2.rotate(roi_frame, cv2.ROTATE_180)
        hsv = cv2.cvtColor(roi_frame_rotated, cv2.COLOR_BGR2HSV)
        
        # Store debug mask (this will display right-side up in your corner)
        self.debug_mask = self._get_red_mask(hsv)
        
        # Detect candidates using the ROTATED frame
        candidates = self._detect_stop_candidates(roi_frame_rotated, hsv)
        
        # ── Map the detected coordinates BACK to your upside-down video feed ──
        mapped_candidates = []
        for sign_type, (rx, ry, rw, rh), conf in candidates:
            # Un-rotate the bounding box coordinates
            orig_x = w - rx - rw
            orig_y = roi_h - ry - rh + roi_top
            mapped_candidates.append((sign_type, (orig_x, orig_y, rw, rh), conf))
            
        self.detected_signs = mapped_candidates
        
        # ── Temporal filtering ───────────────────────────────────────
        current_centers = [(x + w//2, y + h//2) for (_, (x, y, w, h), _) in self.detected_signs]
        self.detection_history.append(current_centers)
        
        self.confirmed_signs = []
        for sign_type, (x, y, w, h), conf in self.detected_signs:
            center = (x + w//2, y + h//2)
            consecutive = self._count_consecutive_near(center)
            
            if consecutive >= self.min_consecutive_detections:
                self.confirmed_signs.append((sign_type, (x, y, w, h), conf))
        
        return self.confirmed_signs

    def _count_consecutive_near(self, center, threshold=50):
        """Count how many recent frames had a detection near this center."""
        count = 0
        for hist_centers in reversed(self.detection_history):
            found = False
            for hc in hist_centers:
                dist = np.sqrt((center[0] - hc[0])**2 + (center[1] - hc[1])**2)
                if dist < threshold:
                    found = True
                    break
            if found:
                count += 1
            else:
                break
        return count

    def _get_red_mask(self, hsv):
        """Generate red mask for debugging."""
        mask1 = cv2.inRange(hsv, self.red_lower1, self.red_upper1)
        mask2 = cv2.inRange(hsv, self.red_lower2, self.red_upper2)
        return cv2.bitwise_or(mask1, mask2)

    def _detect_stop_candidates(self, frame_rotated, hsv):
        """Detect STOP sign candidates with strict filtering."""
        mask = self._get_red_mask(hsv)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        detected = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (self.min_sign_area < area < self.max_sign_area):
                continue

            x, y, w, h = cv2.boundingRect(cnt)

            # STRICT CHECK 1: Aspect ratio must be nearly square
            aspect = w / max(h, 1)
            if not (0.75 < aspect < 1.35):
                continue

            # STRICT CHECK 2: Must be octagon-like (6-12 sides)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.025 * peri, True)
            sides = len(approx)
            
            if not (6 <= sides <= 12):
                continue

            # STRICT CHECK 3: Solidity must be high (filled shape)
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = area / max(hull_area, 1)
            
            if solidity < 0.75:
                continue

            # STRICT CHECK 4: Must have white/bright interior (STOP text)
            white_ratio = self._check_white_interior(frame_rotated, x, y, w, h)
            
            if white_ratio < 0.05:
                continue

            # STRICT CHECK 5: Circularity (octagon is fairly circular)
            circularity = (4 * np.pi * area) / max(peri * peri, 1)
            
            if circularity < 0.55:
                continue

            # Calculate confidence
            confidence = self._calculate_confidence(
                sides, solidity, circularity, white_ratio, area
            )
            
            if confidence >= self.min_confidence:
                detected.append(("STOP", (x, y, w, h), confidence))

        return detected

    def _check_white_interior(self, frame_rotated, x, y, w, h):
        """Check for white text inside the sign."""
        margin_x = int(w * 0.25)
        margin_y = int(h * 0.25)
        
        x1 = max(0, x + margin_x)
        y1 = max(0, y + margin_y)
        x2 = min(frame_rotated.shape[1], x + w - margin_x)
        y2 = min(frame_rotated.shape[0], y + h - margin_y)
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
            
        center_region = frame_rotated[y1:y2, x1:x2]
        
        if center_region.size == 0:
            return 0.0
        
        gray = cv2.cvtColor(center_region, cv2.COLOR_BGR2GRAY)
        white_pixels = np.sum(gray > 160)
        total_pixels = gray.size
        
        return white_pixels / max(total_pixels, 1)

    def _calculate_confidence(self, sides, solidity, circularity, white_ratio, area):
        """Calculate confidence score."""
        side_score = max(0.0, 1.0 - abs(sides - 8) * 0.15)
        solidity_score = min(1.0, solidity / 0.9)
        circ_score = min(1.0, circularity / 0.8)
        white_score = min(1.0, white_ratio / 0.25)
        
        if 1500 < area < 20000:
            area_score = 1.0
        else:
            area_score = 0.7

        conf = (side_score * 0.25 + 
                solidity_score * 0.20 + 
                circ_score * 0.15 + 
                white_score * 0.30 +
                area_score * 0.10)
        
        return round(max(0.0, min(1.0, conf)), 2)

    def get_action(self):
        """Return action - only STOP or nothing."""
        if self.confirmed_signs:
            return "STOP", None
        return None, None

    def draw_overlay(self, frame):
        """Draw detection overlay."""
        h_frame = frame.shape[0]
        w_frame = frame.shape[1]
        
        # Debug: show red mask in TOP RIGHT corner to avoid overlapping UI
        if self.debug_mask is not None:
            mask_small = cv2.resize(self.debug_mask, (100, 75))
            mask_colored = cv2.cvtColor(mask_small, cv2.COLOR_GRAY2BGR)
            mask_colored[:, :, 2] = mask_small
            
            # Positioned in Top Right
            frame[10:85, w_frame-110:w_frame-10] = mask_colored
            cv2.putText(frame, "RED", (w_frame-105, 98),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        # Draw raw detections
        for sign_type, (x, y, w, h), conf in self.detected_signs:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 1)
            cv2.putText(frame, f"?{conf:.0%}", (x, y-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        # Draw CONFIRMED STOP signs
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
        self.detected_signs = []
        self.confirmed_signs = []
        self.detection_history.clear()
        self.debug_mask = None

    def red_detected(self) -> bool:
        """True if any red region was found this frame (used as DNN hint)."""
        return len(self.detected_signs) > 0


# ── ArUco Speed Limit Detector ────────────────────────────────────────────────
#
#  Print markers from: https://chev.me/arucogen/
#  Dictionary: 4x4_50
#
#  Marker ID → Speed limit:
#    ID 1  →  10 km/h  (SLOW)
#    ID 2  →  20 km/h  (AVG)
#    ID 3  →  30 km/h  (NORMAL)
#    ID 4  →  50 km/h  (MAX / madmax)
#
# ─────────────────────────────────────────────────────────────────────────────

ARUCO_SPEED_MAP = {
    1: 10,
    2: 20,
    3: 30,
    4: 50,
}

SPEED_MODE_LABEL = {10: "SLOW", 20: "AVG", 30: "NORMAL", 50: "MAX"}


class ArucoSpeedDetector:
    """
    Detects printed ArUco markers and maps them to speed limits.
    Zero ML — pure OpenCV marker detection, runs in < 1 ms.

    detect(frame)       → int | None  (speed limit in km/h, or None)
    get_speed_limit()   → int | None
    get_speed_fraction()→ float | None  (motor fraction for the limit)
    draw_overlay(frame) → annotated frame
    reset()             → clear state
    """

    # Motor fractions matching SPEED_LIMIT_MAP in dnn_detector
    SPEED_FRACTIONS = {10: 0.30, 20: 0.50, 30: 0.75, 50: 1.00}

    HOLD_FRAMES = 45   # keep limit active for N frames after last sighting

    def __init__(self):
        aruco_dict   = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        aruco_params = cv2.aruco.DetectorParameters()
        try:
            # OpenCV 4.7+
            self._detector     = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
            self._use_new_api  = True
        except AttributeError:
            # OpenCV ≤ 4.6
            self._aruco_dict   = aruco_dict
            self._aruco_params = aruco_params
            self._use_new_api  = False

        self._speed_limit   = None
        self._hold_counter  = 0
        self._last_ids      = []

        print("[ArUco] Speed-limit detector ready (DICT_4X4_50)")
        print("[ArUco] IDs: 1=10 km/h  2=20 km/h  3=30 km/h  4=50 km/h")

    def detect(self, frame: np.ndarray) -> "int | None":
        """Run ArUco detection. Returns speed limit km/h or None."""
        if self._use_new_api:
            corners, ids, _ = self._detector.detectMarkers(frame)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(
                frame, self._aruco_dict, parameters=self._aruco_params
            )

        self._last_ids = []
        if ids is not None:
            for marker_id in ids.flatten().tolist():
                self._last_ids.append(marker_id)
                if marker_id in ARUCO_SPEED_MAP:
                    self._speed_limit  = ARUCO_SPEED_MAP[marker_id]
                    self._hold_counter = self.HOLD_FRAMES

        # Count down hold timer
        if self._hold_counter > 0:
            self._hold_counter -= 1
        else:
            self._speed_limit = None

        return self._speed_limit

    def get_speed_limit(self) -> "int | None":
        return self._speed_limit

    def get_speed_fraction(self) -> "float | None":
        if self._speed_limit is None:
            return None
        return self.SPEED_FRACTIONS.get(self._speed_limit)

    def draw_overlay(self, frame: np.ndarray) -> np.ndarray:
        if self._speed_limit is not None:
            label = SPEED_MODE_LABEL.get(self._speed_limit, "")
            text  = f"LIMIT {self._speed_limit} km/h [{label}]"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (8, 200), (tw + 18, 226), (0, 80, 120), -1)
            cv2.putText(frame, text, (12, 220),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)
        return frame

    def reset(self):
        self._speed_limit  = None
        self._hold_counter = 0
        self._last_ids     = []