"""
color_sign_detector.py - VectorVance Color Sign Detection
──────────────────────────────────────────────────────────
Detects GREEN and BLUE colored signs/markers at track intersections.
The web dashboard tells the car which color = shortest path.
This module spots the signs and reports which side they're on.

Physical signs: Print or tape a solid GREEN rectangle and a solid BLUE
rectangle (about 8x5cm each) and place them at the fork — one on each side.

USAGE:
    from color_sign_detector import ColorSignDetector

    detector = ColorSignDetector()

    # In your frame loop:
    detections = detector.detect(frame)
    # Returns: [{"color": "GREEN", "x": 120, "y": 200, "w": 60, "h": 40, "side": "LEFT", "area": 2400}, ...]

    # Get steering recommendation:
    decision = detector.get_steering_decision("GREEN")
    # Returns: "LEFT" | "RIGHT" | "UNKNOWN"
"""

import cv2
import numpy as np
from collections import deque


class ColorSignDetector:
    """Detects green and blue colored signs from the camera feed."""

    def __init__(self):
        # ── GREEN HSV RANGE ──────────────────────────────────────────
        # Covers bright green paper/cardstock/tape
        self.green_lower = np.array([35, 80, 80], dtype=np.uint8)
        self.green_upper = np.array([85, 255, 255], dtype=np.uint8)

        # ── BLUE HSV RANGE ───────────────────────────────────────────
        # Covers bright blue paper/cardstock/tape
        self.blue_lower = np.array([95, 80, 80], dtype=np.uint8)
        self.blue_upper = np.array([130, 255, 255], dtype=np.uint8)

        # ── DETECTION THRESHOLDS ─────────────────────────────────────
        self.min_area = 300          # minimum contour area in pixels
        self.max_area = 50000        # maximum (reject huge blobs)
        self.min_aspect = 0.3        # min width/height ratio
        self.max_aspect = 4.0        # max width/height ratio
        self.min_solidity = 0.6      # how "filled" the shape is

        # ── TEMPORAL SMOOTHING ───────────────────────────────────────
        self.history_size = 6
        self.green_history = deque(maxlen=self.history_size)
        self.blue_history = deque(maxlen=self.history_size)
        self.min_consecutive = 3     # need 3 of last 6 frames to confirm

        # ── STATE ────────────────────────────────────────────────────
        self.detections = []          # current frame detections
        self.confirmed_green = None   # confirmed green sign info
        self.confirmed_blue = None    # confirmed blue sign info

        # ── DEBUG ────────────────────────────────────────────────────
        self.debug_mask_green = None
        self.debug_mask_blue = None

        print("Color sign detector initialized (GREEN + BLUE)")

    def detect(self, frame):
        """
        Detect green and blue signs in the frame.

        Args:
            frame: BGR image from camera (any size, will be resized)

        Returns:
            List of detections: [{"color", "x", "y", "w", "h", "side", "area", "confidence"}, ...]
        """
        frame = cv2.resize(frame, (640, 480))
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, w = frame.shape[:2]
        center_x = w // 2

        self.detections = []

        # Detect each color
        green_dets = self._detect_color(hsv, "GREEN", self.green_lower, self.green_upper, center_x)
        blue_dets = self._detect_color(hsv, "BLUE", self.blue_lower, self.blue_upper, center_x)

        self.detections = green_dets + blue_dets

        # ── TEMPORAL FILTERING ───────────────────────────────────────
        # Track the best detection per color per frame
        best_green = max(green_dets, key=lambda d: d["area"], default=None)
        best_blue = max(blue_dets, key=lambda d: d["area"], default=None)

        self.green_history.append(best_green)
        self.blue_history.append(best_blue)

        # Confirm if seen in enough recent frames
        green_count = sum(1 for d in self.green_history if d is not None)
        blue_count = sum(1 for d in self.blue_history if d is not None)

        self.confirmed_green = best_green if green_count >= self.min_consecutive else None
        self.confirmed_blue = best_blue if blue_count >= self.min_consecutive else None

        return self.detections

    def _detect_color(self, hsv, color_name, lower, upper, center_x):
        """Detect contours of a specific color."""
        mask = cv2.inRange(hsv, lower, upper)

        # Morphological cleanup
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Store debug masks
        if color_name == "GREEN":
            self.debug_mask_green = mask
        else:
            self.debug_mask_blue = mask

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (self.min_area < area < self.max_area):
                continue

            x, y, w, h = cv2.boundingRect(cnt)

            # Aspect ratio check
            aspect = w / max(h, 1)
            if not (self.min_aspect < aspect < self.max_aspect):
                continue

            # Solidity check (how rectangular/filled is it)
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = area / max(hull_area, 1)
            if solidity < self.min_solidity:
                continue

            # Determine which side of frame
            sign_center_x = x + w // 2
            side = "LEFT" if sign_center_x < center_x else "RIGHT"

            # Confidence based on area and solidity
            confidence = min(1.0, (area / 2000) * solidity)

            detections.append({
                "color": color_name,
                "x": x, "y": y, "w": w, "h": h,
                "side": side,
                "area": area,
                "confidence": round(confidence, 2),
                "center_x": sign_center_x,
            })

        # Sort by area (largest = most likely the real sign)
        detections.sort(key=lambda d: d["area"], reverse=True)

        # Keep at most 2 per color (filter noise)
        return detections[:2]

    def get_steering_decision(self, target_color):
        """
        Given the target color from the web dashboard, return which
        direction to steer at the fork.

        Args:
            target_color: "GREEN" or "BLUE"

        Returns:
            "LEFT" | "RIGHT" | "UNKNOWN"
        """
        if target_color == "GREEN" and self.confirmed_green:
            return self.confirmed_green["side"]
        elif target_color == "BLUE" and self.confirmed_blue:
            return self.confirmed_blue["side"]
        return "UNKNOWN"

    def get_both_signs(self):
        """
        Return info about both confirmed signs (for the dashboard overlay).

        Returns:
            {"green": {"side": "LEFT", ...} or None, "blue": {"side": "RIGHT", ...} or None}
        """
        return {
            "green": self.confirmed_green,
            "blue": self.confirmed_blue,
        }

    def is_fork_visible(self):
        """True if both color signs are visible (meaning we're at the fork)."""
        return self.confirmed_green is not None and self.confirmed_blue is not None

    def draw_overlay(self, frame):
        """Draw detection overlay on the debug frame."""
        h, w = frame.shape[:2]

        # Draw all raw detections (thin border)
        for det in self.detections:
            color_bgr = (0, 255, 0) if det["color"] == "GREEN" else (255, 150, 0)
            cv2.rectangle(frame,
                          (det["x"], det["y"]),
                          (det["x"] + det["w"], det["y"] + det["h"]),
                          color_bgr, 1)

        # Draw confirmed signs (thick border + label)
        for label, det in [("GREEN", self.confirmed_green), ("BLUE", self.confirmed_blue)]:
            if det is None:
                continue
            color_bgr = (0, 255, 0) if label == "GREEN" else (255, 150, 0)

            # Thick bounding box
            cv2.rectangle(frame,
                          (det["x"], det["y"]),
                          (det["x"] + det["w"], det["y"] + det["h"]),
                          color_bgr, 3)

            # Label with background
            text = f"{label} ({det['side']})"
            ty = det["y"] - 8 if det["y"] > 30 else det["y"] + det["h"] + 20
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (det["x"], ty - th - 4),
                          (det["x"] + tw + 8, ty + 4), color_bgr, -1)
            cv2.putText(frame, text, (det["x"] + 4, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Status bar at top
        g_status = f"GREEN: {self.confirmed_green['side']}" if self.confirmed_green else "GREEN: ---"
        b_status = f"BLUE: {self.confirmed_blue['side']}" if self.confirmed_blue else "BLUE: ---"

        cv2.putText(frame, g_status, (10, h - 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        cv2.putText(frame, b_status, (10, h - 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 150, 0), 1)

        if self.is_fork_visible():
            cv2.putText(frame, "FORK DETECTED", (10, h - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        # Debug: color masks in corners
        if self.debug_mask_green is not None:
            small_g = cv2.resize(self.debug_mask_green, (80, 60))
            small_g_c = cv2.cvtColor(small_g, cv2.COLOR_GRAY2BGR)
            small_g_c[:, :, 0] = 0   # remove blue channel
            small_g_c[:, :, 2] = 0   # remove red channel — keep green
            y_off = h - 70
            frame[y_off:y_off+60, w-90:w-10] = small_g_c
            cv2.putText(frame, "G", (w - 88, y_off - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)

        if self.debug_mask_blue is not None:
            small_b = cv2.resize(self.debug_mask_blue, (80, 60))
            small_b_c = cv2.cvtColor(small_b, cv2.COLOR_GRAY2BGR)
            small_b_c[:, :, 1] = 0   # remove green channel
            small_b_c[:, :, 2] = 0   # remove red channel — keep blue
            y_off = h - 140
            frame[y_off:y_off+60, w-90:w-10] = small_b_c
            cv2.putText(frame, "B", (w - 88, y_off - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 150, 0), 1)

        return frame

    def reset(self):
        self.detections = []
        self.confirmed_green = None
        self.confirmed_blue = None
        self.green_history.clear()
        self.blue_history.clear()
        self.debug_mask_green = None
        self.debug_mask_blue = None


# ─────────────────────────────────────────────────────────────────────
#  STANDALONE TEST (webcam)
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    detector = ColorSignDetector()

    source = 0
    if len(sys.argv) > 1:
        source = sys.argv[1]
        try:
            source = int(source)
        except ValueError:
            pass

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Cannot open: {source}")
        exit(1)

    print(f"Color sign detector test — source: {source}")
    print("Hold up a GREEN and/or BLUE object in front of the camera.")
    print("Press Q to quit.\n")

    target = "GREEN"  # simulate web dashboard saying "follow green"

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect(frame)
        frame = detector.draw_overlay(frame)

        # Show steering decision
        decision = detector.get_steering_decision(target)
        dec_color = (0, 255, 0) if decision != "UNKNOWN" else (100, 100, 100)
        cv2.putText(frame, f"Target: {target} -> Steer: {decision}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, dec_color, 2)

        if detector.is_fork_visible():
            cv2.putText(frame, "BOTH SIGNS VISIBLE — AT FORK!",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        cv2.imshow("Color Sign Detector", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('g'):
            target = "GREEN"
            print("Target: GREEN")
        elif key == ord('b'):
            target = "BLUE"
            print("Target: BLUE")

    cap.release()
    cv2.destroyAllWindows()
