"""
yolo_detector.py - YOLO-based detection for VectorVance
────────────────────────────────────────────────────────
Uses YOLOv8n (ultralytics, COCO pretrained) to detect:
  - Stop signs     (COCO class 11)
  - Traffic lights (COCO class 9)   ← new
  - Persons        (class 0)  ← pedestrian safety
  - Bicycles       (class 1)
  - Cars           (class 2)
  - Motorcycles    (class 3)
  - Buses          (class 5)
  - Trucks         (class 7)

Stop signs feed the STOP action (replaces color+shape detector).
Persons/vehicles feed the obstacle speed modifier.
Zone-aware: only objects in the center zone (car's path) trigger braking.

Install: pip install ultralytics
"""

import cv2
import numpy as np

try:
    from ultralytics import YOLO
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False


# ── COCO class IDs we track ───────────────────────────────────────────────────
STOP_SIGN_CLASS    = 11
TRAFFIC_LIGHT_CLASS = 9
OBSTACLE_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}
ALL_TRACKED = {
    STOP_SIGN_CLASS:     "stop sign",
    TRAFFIC_LIGHT_CLASS: "traffic light",
    **OBSTACLE_CLASSES,
}

# BGR colors per class for the HUD
_CLASS_COLORS = {
    11: (0,   0,   255),   # stop sign     — red
    9:  (0,   255, 0  ),   # traffic light — green
    0:  (0,   165, 255),   # person        — orange
    1:  (0,   255, 255),   # bicycle       — yellow
    2:  (0,   200, 0  ),   # car           — green
    3:  (255, 0,   255),   # motorcycle    — magenta
    5:  (255, 255, 0  ),   # bus           — cyan
    7:  (255, 128, 0  ),   # truck         — blue-orange
}

# Proximity weight per class (persons are highest risk)
_CLASS_WEIGHTS = {
    0: 1.5,   # person — highest priority
    1: 1.2,   # bicycle
    2: 1.0,   # car
    3: 1.2,   # motorcycle
    5: 1.0,   # bus
    7: 1.0,   # truck
}

# Temporal stop-sign filtering: confirm if seen in ≥2 of last 5 frames
_STOP_HISTORY_LEN   = 5
_STOP_CONFIRM_COUNT = 2

# Zone fractions: frame divided into left | center | right thirds
_ZONE_LEFT_END   = 1 / 3
_ZONE_RIGHT_START = 2 / 3


class YoloDetector:
    """
    Drop-in replacement for TrafficSignDetector that also provides
    obstacle proximity for adaptive speed control.

    Improvements over v1:
      - Zone-aware: obstacles in left/right zones don't trigger braking
      - Better proximity using bottom-Y position + bbox area combined
      - Class-weighted danger (persons > vehicles)
      - Traffic light tracked as an informational detection
      - Optional ByteTrack object tracking for stable IDs across frames
      - get_danger_level() → "CLEAR" | "CAUTION" | "DANGER" | "STOP"
    """

    def __init__(self, model_name: str = "yolov8n.pt",
                 conf_threshold: float = 0.45,
                 frame_width: int = 640,
                 frame_height: int = 480,
                 skip_frames: int = 1,
                 use_tracking: bool = False):
        """
        skip_frames  : run YOLO every N frames (1 = every frame for PC/GPU,
                       3-5 recommended for Raspberry Pi CPU).
        use_tracking : use ByteTrack (.track()) instead of detect-only (.predict())
                       for stable object IDs — slightly slower.
        """
        self.available       = _YOLO_AVAILABLE
        self.conf_threshold  = conf_threshold
        self.frame_width     = frame_width
        self.frame_height    = frame_height
        self.skip_frames     = max(1, skip_frames)
        self.use_tracking    = use_tracking
        self._frame_counter  = 0

        # Detection results (populated each inference frame)
        self.stop_signs:     list = []   # [(bbox, conf), ...]
        self.traffic_lights: list = []   # [(bbox, conf), ...]
        self.obstacles:      list = []   # [(cls_id, label, bbox, conf, proximity, zone), ...]
        self.all_detections: list = []   # [(cls_id, label, bbox, conf), ...]

        # Temporal stop-sign filtering
        self._stop_history:   list[bool] = []
        self._stop_confirmed: bool       = False

        # Most recent danger assessment (updated on inference frames)
        self._danger_level: str   = "CLEAR"   # "CLEAR" | "CAUTION" | "DANGER" | "STOP"
        self._speed_modifier: float = 1.0

        if not self.available:
            print("[YOLO] ultralytics not installed — running in disabled mode")
            print("[YOLO]   pip install ultralytics")
            return

        print(f"[YOLO] Loading {model_name} ...")
        self.model = YOLO(model_name)
        mode = "tracking" if use_tracking else "detection"
        skip_info = f"every {skip_frames} frame(s)" if skip_frames > 1 else "every frame"
        tracked = ", ".join(ALL_TRACKED.values())
        print(f"[YOLO] Ready — {mode}, {skip_info}")
        print(f"[YOLO] Tracking: {tracked}")

    # ── public API ────────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> list:
        """
        Run inference on frame. Must be called once per frame.
        Returns all_detections list (may be empty if YOLO unavailable).
        Skips inference and returns cached results on non-inference frames.
        """
        self._frame_counter += 1

        if not self.available:
            return []

        # Return cached results on skipped frames
        if self._frame_counter % self.skip_frames != 0:
            return self.all_detections

        self.stop_signs     = []
        self.traffic_lights = []
        self.obstacles      = []
        self.all_detections = []

        if self.use_tracking:
            results = self.model.track(
                frame,
                conf=self.conf_threshold,
                classes=list(ALL_TRACKED.keys()),
                verbose=False,
                persist=True,
            )
        else:
            results = self.model(
                frame,
                conf=self.conf_threshold,
                classes=list(ALL_TRACKED.keys()),
                verbose=False,
            )

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                if cls_id not in ALL_TRACKED:
                    continue
                conf  = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                bbox  = (x1, y1, x2 - x1, y2 - y1)   # (x, y, w, h)
                label = ALL_TRACKED[cls_id]
                self.all_detections.append((cls_id, label, bbox, conf))

                if cls_id == STOP_SIGN_CLASS:
                    self.stop_signs.append((bbox, conf))
                elif cls_id == TRAFFIC_LIGHT_CLASS:
                    self.traffic_lights.append((bbox, conf))
                elif cls_id in OBSTACLE_CLASSES:
                    zone      = self._get_zone(bbox)
                    proximity = self._estimate_proximity(bbox, y2)
                    self.obstacles.append(
                        (cls_id, label, bbox, conf, proximity, zone)
                    )

        # Temporal stop-sign filtering
        self._stop_history.append(len(self.stop_signs) > 0)
        if len(self._stop_history) > _STOP_HISTORY_LEN:
            self._stop_history.pop(0)
        self._stop_confirmed = (
            sum(self._stop_history[-_STOP_HISTORY_LEN:]) >= _STOP_CONFIRM_COUNT
        )

        # Update cached danger level and speed modifier
        self._danger_level   = self._compute_danger_level()
        self._speed_modifier = self._compute_speed_modifier()

        return self.all_detections

    def get_action(self) -> tuple[str | None, None]:
        """Mirror TrafficSignDetector.get_action() interface."""
        if self._stop_confirmed:
            return "STOP", None
        return None, None

    def get_speed_modifier(self) -> float:
        """
        Returns speed multiplier based on closest in-path obstacle.
          1.0  — clear path
          0.6  — obstacle approaching
          0.3  — obstacle close
          0.0  — obstacle very close / stop
        """
        return self._speed_modifier

    def get_danger_level(self) -> str:
        """
        Returns a human-readable danger level string:
          "CLEAR"   — no in-path obstacles
          "CAUTION" — obstacle detected in path, far
          "DANGER"  — obstacle close
          "STOP"    — obstacle very close or stop sign confirmed
        """
        return self._danger_level

    def get_zone_threats(self) -> dict[str, list]:
        """
        Returns obstacles grouped by zone: {"left": [...], "center": [...], "right": [...]}.
        Each item: (cls_id, label, bbox, conf, proximity).
        Useful for steering decisions (e.g. swerve around left-zone obstacle).
        """
        zones: dict[str, list] = {"left": [], "center": [], "right": []}
        for cls_id, label, bbox, conf, proximity, zone in self.obstacles:
            zones[zone].append((cls_id, label, bbox, conf, proximity))
        return zones

    def get_closest_obstacle(self) -> tuple | None:
        """
        Returns the in-path obstacle with the highest weighted proximity,
        or None if no center-zone obstacles exist.
        Returns: (cls_id, label, bbox, conf, proximity) or None.
        """
        center_obs = [
            (cls_id, label, bbox, conf, proximity)
            for cls_id, label, bbox, conf, proximity, zone in self.obstacles
            if zone == "center"
        ]
        if not center_obs:
            return None
        return max(center_obs, key=lambda o: o[4] * _CLASS_WEIGHTS.get(o[0], 1.0))

    def stop_sign_detected(self) -> bool:
        return self._stop_confirmed

    def draw_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Draw bounding boxes + zone lines + danger bar onto frame."""
        fh, fw = frame.shape[:2]

        # Zone dividers (subtle vertical lines)
        z1 = int(fw * _ZONE_LEFT_END)
        z2 = int(fw * _ZONE_RIGHT_START)
        cv2.line(frame, (z1, fh // 2), (z1, fh), (80, 80, 80), 1)
        cv2.line(frame, (z2, fh // 2), (z2, fh), (80, 80, 80), 1)

        # Bounding boxes
        for cls_id, label, (x, y, w, h), conf in self.all_detections:
            color     = _CLASS_COLORS.get(cls_id, (200, 200, 200))
            thickness = 3 if cls_id == STOP_SIGN_CLASS else 2

            # Dim out-of-path obstacles slightly
            zone = self._get_zone((x, y, w, h))
            if zone != "center" and cls_id in OBSTACLE_CLASSES:
                color = tuple(int(c * 0.55) for c in color)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)

            text = f"{label} {conf:.0%}"
            (tw, th), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
            )
            cv2.rectangle(frame, (x, y - th - 8), (x + tw + 8, y), color, -1)
            cv2.putText(frame, text, (x + 4, y - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

        # Bottom warning bar
        danger = self._danger_level
        if danger != "CLEAR":
            bar_colors = {
                "CAUTION": (0, 200, 255),
                "DANGER":  (0, 100, 255),
                "STOP":    (0, 0, 255),
            }
            bar_color = bar_colors.get(danger, (0, 120, 255))
            closest = self.get_closest_obstacle()
            obs_name = closest[1].upper() if closest else "OBSTACLE"
            warn_txt = f"{danger}: {obs_name} IN PATH"
            if self._stop_confirmed:
                warn_txt = "STOP SIGN CONFIRMED"
            cv2.rectangle(frame, (0, fh - 28), (fw, fh), bar_color, -1)
            cv2.putText(frame, warn_txt, (10, fh - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Speed modifier indicator
            mod_txt = f"speed x{self._speed_modifier:.1f}"
            cv2.putText(frame, mod_txt, (fw - 120, fh - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return frame

    def reset(self):
        self.stop_signs     = []
        self.traffic_lights = []
        self.obstacles      = []
        self.all_detections = []
        self._stop_history  = []
        self._stop_confirmed = False
        self._danger_level  = "CLEAR"
        self._speed_modifier = 1.0
        self._frame_counter  = 0

    # ── internal ──────────────────────────────────────────────────────────────

    def _get_zone(self, bbox: tuple) -> str:
        """Classify bbox into 'left', 'center', or 'right' zone by x-center."""
        x, y, w, h = bbox
        cx = x + w / 2
        rel = cx / self.frame_width
        if rel < _ZONE_LEFT_END:
            return "left"
        if rel >= _ZONE_RIGHT_START:
            return "right"
        return "center"

    def _estimate_proximity(self, bbox: tuple, y2: int) -> float:
        """
        Improved proximity (0 = far, 1 = very close) combining:
          - bottom-Y position: lower in frame → closer to car
          - bbox fill ratio:   larger bbox → closer to car
        """
        _, _, w, h = bbox
        fill_ratio = (w * h) / (self.frame_width * self.frame_height)
        y_score    = y2 / self.frame_height                        # 0..1
        area_score = min(1.0, fill_ratio / 0.20)                   # saturates at 20% fill
        return min(1.0, 0.5 * y_score + 0.5 * area_score)

    def _compute_speed_modifier(self) -> float:
        """Speed multiplier from the worst in-path (center-zone) obstacle."""
        center_obs = [
            (proximity * _CLASS_WEIGHTS.get(cls_id, 1.0),)
            for cls_id, label, bbox, conf, proximity, zone in self.obstacles
            if zone == "center"
        ]
        if not center_obs:
            return 1.0

        max_weighted = max(v[0] for v in center_obs)
        if max_weighted > 0.80:
            return 0.0
        if max_weighted > 0.55:
            return 0.3
        if max_weighted > 0.30:
            return 0.6
        return 1.0

    def _compute_danger_level(self) -> str:
        """Compute danger level string from current state."""
        if self._stop_confirmed:
            return "STOP"
        modifier = self._compute_speed_modifier()
        if modifier == 0.0:
            return "STOP"
        if modifier <= 0.3:
            return "DANGER"
        if modifier <= 0.6:
            return "CAUTION"
        return "CLEAR"
