"""
dnn_detector.py — OpenCV DNN + SSD MobileNet v2 COCO detector
────────────────────────────────────────────────────────────────
Uses only cv2.dnn — no PyTorch, no ultralytics.
Works on Raspberry Pi 3 (ARMv7/ARM64).

Detects: stop signs, traffic lights, persons, bicycles, cars,
         motorcycles, buses, trucks.

Model files required in src/ (run once to download):
  ssd_mobilenet_v2_coco.pb      ~67 MB  — frozen TF inference graph
  ssd_mobilenet_v2_coco.pbtxt   ~8  KB  — OpenCV DNN graph config

  Download both automatically:
    python dnn_detector.py --download

  Or manually:
    # pbtxt (small, from OpenCV extras)
    wget https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/ssd_mobilenet_v2_coco_2018_03_29.pbtxt \\
         -O ssd_mobilenet_v2_coco.pbtxt

    # pb (from TF model zoo — 67 MB tarball, extract frozen_inference_graph.pb)
    wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz
    tar -xzf ssd_mobilenet_v2_coco_2018_03_29.tar.gz \\
        ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb \\
        --strip-components=1
    mv frozen_inference_graph.pb ssd_mobilenet_v2_coco.pb

Expected performance on Raspberry Pi 3:
  ~3–6 FPS inference at 300×300 input
  With skip_frames=5 → ~15–25 FPS effective (cached results on skipped frames)
"""

import os
import cv2
import numpy as np

# ── TF COCO class IDs (1-indexed; background = 0) ────────────────────────────
_TF_STOP_SIGN       = 13
_TF_TRAFFIC_LIGHT   = 10
_TF_OBSTACLE_CLASSES = {1, 2, 3, 4, 6, 8}   # person, bicycle, car, moto, bus, truck

# Map TF class ID → (internal_id, label)
# Internal IDs use COCO 0-indexed numbering consistent with the rest of the project.
_TF_TO_INTERNAL = {
    1:  (0,  "person"),
    2:  (1,  "bicycle"),
    3:  (2,  "car"),
    4:  (3,  "motorcycle"),
    6:  (5,  "bus"),
    8:  (7,  "truck"),
    10: (9,  "traffic light"),
    13: (11, "stop sign"),
}
_ALL_TF_IDS = set(_TF_TO_INTERNAL.keys())

# Internal class IDs
STOP_SIGN_CLASS     = 11
TRAFFIC_LIGHT_CLASS = 9
OBSTACLE_CLASSES = {0: "person", 1: "bicycle", 2: "car",
                    3: "motorcycle", 5: "bus", 7: "truck"}

# BGR colours per internal class for the HUD overlay
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

# Proximity weight per class (persons highest risk)
_CLASS_WEIGHTS = {
    0: 1.5,   # person
    1: 1.2,   # bicycle
    2: 1.0,   # car
    3: 1.2,   # motorcycle
    5: 1.0,   # bus
    7: 1.0,   # truck
}

# Temporal stop-sign filter: confirm if seen in ≥2 of last 5 frames
_STOP_HISTORY_LEN   = 5
_STOP_CONFIRM_COUNT = 2

# Zone fractions: frame divided into left | center | right thirds
_ZONE_LEFT_END    = 1 / 3
_ZONE_RIGHT_START = 2 / 3

# MobileNet SSD input size (must be 300×300 for this model)
_INPUT_SIZE = (300, 300)


class DNNDetector:
    """
    OpenCV DNN object detector using SSD MobileNet v2 COCO.
    No PyTorch or ultralytics required — works on Raspberry Pi 3.

    detect()               → run inference (cached on skipped frames)
    get_action()           → ("STOP", None) when stop sign confirmed
    get_speed_modifier()   → 0.0–1.0 speed multiplier from obstacle proximity
    get_danger_level()     → "CLEAR" | "CAUTION" | "DANGER" | "STOP"
    get_zone_threats()     → obstacles grouped by zone
    get_closest_obstacle() → highest-weighted in-path obstacle
    stop_sign_detected()   → bool
    draw_overlay()         → annotated frame
    reset()                → clear state
    """

    def __init__(self,
                 model_name: str   = "ssd_mobilenet_v2_coco.pb",
                 conf_threshold: float = 0.45,
                 frame_width: int  = 640,
                 frame_height: int = 480,
                 skip_frames: int  = 1,
                 use_tracking: bool = False):   # kept for API compat; unused
        """
        model_name     : path to the .pb file; .pbtxt derived from same base name
        conf_threshold : minimum detection confidence (default 0.45)
        skip_frames    : run DNN every N frames (1=every frame; 3–5 recommended on Pi 3)
        use_tracking   : ignored (retained for API compatibility)
        """
        self.conf_threshold  = conf_threshold
        self.frame_width     = frame_width
        self.frame_height    = frame_height
        self.skip_frames     = max(1, skip_frames)
        self._frame_counter  = 0
        self.available       = False

        # Detection results
        self.stop_signs:     list = []
        self.traffic_lights: list = []
        self.obstacles:      list = []
        self.all_detections: list = []

        # Temporal stop-sign filtering
        self._stop_history:   list = []
        self._stop_confirmed: bool = False

        # Cached danger / speed state
        self._danger_level:   str   = "CLEAR"
        self._speed_modifier: float = 1.0

        pb_path    = model_name
        pbtxt_path = os.path.splitext(model_name)[0] + ".pbtxt"

        if not os.path.isfile(pb_path):
            print(f"[DNN] Model not found: {pb_path}")
            print("[DNN] Run:  python dnn_detector.py --download")
            print("[DNN] Running in disabled mode — no object detection")
            return

        if not os.path.isfile(pbtxt_path):
            print(f"[DNN] Config not found: {pbtxt_path}")
            print("[DNN] Run:  python dnn_detector.py --download")
            print("[DNN] Running in disabled mode — no object detection")
            return

        print(f"[DNN] Loading SSD MobileNet v2 from {pb_path} ...")
        try:
            self._net = cv2.dnn.readNetFromTensorflow(pb_path, pbtxt_path)
            self._net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self._net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            self.available = True
            skip_info = f"every {skip_frames} frame(s)" if skip_frames > 1 else "every frame"
            print(f"[DNN] Ready — SSD MobileNet v2 COCO, {skip_info}")
            print(f"[DNN] Tracking: stop sign, traffic light, person, bicycle, "
                  f"car, motorcycle, bus, truck")
        except cv2.error as e:
            print(f"[DNN] Failed to load model: {e}")
            print("[DNN] Running in disabled mode")

    # ── Public API ────────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> list:
        """
        Run inference on frame. Must be called once per frame.
        Returns all_detections list (empty if DNN unavailable).
        Returns cached results on non-inference frames.
        """
        self._frame_counter += 1

        if not self.available:
            return []

        if self._frame_counter % self.skip_frames != 0:
            return self.all_detections   # cached

        self.stop_signs     = []
        self.traffic_lights = []
        self.obstacles      = []
        self.all_detections = []

        blob = cv2.dnn.blobFromImage(
            frame,
            scalefactor = 1 / 127.5,
            size        = _INPUT_SIZE,
            mean        = (127.5, 127.5, 127.5),
            swapRB      = True,
            crop        = False,
        )
        self._net.setInput(blob)
        raw = self._net.forward()   # shape: (1, 1, N, 7)

        for i in range(raw.shape[2]):
            conf   = float(raw[0, 0, i, 2])
            if conf < self.conf_threshold:
                continue
            tf_cls = int(raw[0, 0, i, 1])
            if tf_cls not in _ALL_TF_IDS:
                continue

            cls_id, label = _TF_TO_INTERNAL[tf_cls]

            x1 = int(raw[0, 0, i, 3] * self.frame_width)
            y1 = int(raw[0, 0, i, 4] * self.frame_height)
            x2 = int(raw[0, 0, i, 5] * self.frame_width)
            y2 = int(raw[0, 0, i, 6] * self.frame_height)

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(self.frame_width - 1, x2), min(self.frame_height - 1, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            bbox = (x1, y1, x2 - x1, y2 - y1)   # (x, y, w, h)
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

        self._danger_level   = self._compute_danger_level()
        self._speed_modifier = self._compute_speed_modifier()

        return self.all_detections

    def get_action(self) -> tuple:
        if self._stop_confirmed:
            return "STOP", None
        return None, None

    def get_speed_modifier(self) -> float:
        return self._speed_modifier

    def get_danger_level(self) -> str:
        return self._danger_level

    def get_zone_threats(self) -> dict:
        zones: dict = {"left": [], "center": [], "right": []}
        for cls_id, label, bbox, conf, proximity, zone in self.obstacles:
            zones[zone].append((cls_id, label, bbox, conf, proximity))
        return zones

    def get_closest_obstacle(self) -> tuple | None:
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
        """Draw bounding boxes, zone lines, and danger bar onto frame."""
        fh, fw = frame.shape[:2]

        z1 = int(fw * _ZONE_LEFT_END)
        z2 = int(fw * _ZONE_RIGHT_START)
        cv2.line(frame, (z1, fh // 2), (z1, fh), (80, 80, 80), 1)
        cv2.line(frame, (z2, fh // 2), (z2, fh), (80, 80, 80), 1)

        for cls_id, label, (x, y, w, h), conf in self.all_detections:
            color     = _CLASS_COLORS.get(cls_id, (200, 200, 200))
            thickness = 3 if cls_id == STOP_SIGN_CLASS else 2

            zone = self._get_zone((x, y, w, h))
            if zone != "center" and cls_id in OBSTACLE_CLASSES:
                color = tuple(int(c * 0.55) for c in color)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)

            text = f"{label} {conf:.0%}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(frame, (x, y - th - 8), (x + tw + 8, y), color, -1)
            cv2.putText(frame, text, (x + 4, y - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

        danger = self._danger_level
        if danger != "CLEAR":
            bar_colors = {
                "CAUTION": (0, 200, 255),
                "DANGER":  (0, 100, 255),
                "STOP":    (0, 0,   255),
            }
            bar_color = bar_colors.get(danger, (0, 120, 255))
            closest   = self.get_closest_obstacle()
            obs_name  = closest[1].upper() if closest else "OBSTACLE"
            warn_txt  = f"{danger}: {obs_name} IN PATH"
            if self._stop_confirmed:
                warn_txt = "STOP SIGN CONFIRMED"
            cv2.rectangle(frame, (0, fh - 28), (fw, fh), bar_color, -1)
            cv2.putText(frame, warn_txt, (10, fh - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            mod_txt = f"speed x{self._speed_modifier:.1f}"
            cv2.putText(frame, mod_txt, (fw - 120, fh - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return frame

    def reset(self):
        self.stop_signs      = []
        self.traffic_lights  = []
        self.obstacles       = []
        self.all_detections  = []
        self._stop_history   = []
        self._stop_confirmed = False
        self._danger_level   = "CLEAR"
        self._speed_modifier = 1.0
        self._frame_counter  = 0

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_zone(self, bbox: tuple) -> str:
        x, y, w, h = bbox
        rel = (x + w / 2) / self.frame_width
        if rel < _ZONE_LEFT_END:
            return "left"
        if rel >= _ZONE_RIGHT_START:
            return "right"
        return "center"

    def _estimate_proximity(self, bbox: tuple, y2: int) -> float:
        _, _, w, h = bbox
        fill_ratio = (w * h) / (self.frame_width * self.frame_height)
        y_score    = y2 / self.frame_height
        area_score = min(1.0, fill_ratio / 0.20)
        return min(1.0, 0.5 * y_score + 0.5 * area_score)

    def _compute_speed_modifier(self) -> float:
        center_obs = [
            proximity * _CLASS_WEIGHTS.get(cls_id, 1.0)
            for cls_id, _, _, _, proximity, zone in self.obstacles
            if zone == "center"
        ]
        if not center_obs:
            return 1.0
        m = max(center_obs)
        if m > 0.80: return 0.0
        if m > 0.55: return 0.3
        if m > 0.30: return 0.6
        return 1.0

    def _compute_danger_level(self) -> str:
        if self._stop_confirmed:
            return "STOP"
        modifier = self._compute_speed_modifier()
        if modifier == 0.0: return "STOP"
        if modifier <= 0.3:  return "DANGER"
        if modifier <= 0.6:  return "CAUTION"
        return "CLEAR"


# ── Model download helper ─────────────────────────────────────────────────────

def download_models(dest_dir: str = "."):
    """Download SSD MobileNet v2 COCO model files into dest_dir."""
    import urllib.request
    import tarfile

    pbtxt_url  = (
        "https://raw.githubusercontent.com/opencv/opencv_extra/"
        "master/testdata/dnn/ssd_mobilenet_v2_coco_2018_03_29.pbtxt"
    )
    pb_url     = (
        "http://download.tensorflow.org/models/object_detection/"
        "ssd_mobilenet_v2_coco_2018_03_29.tar.gz"
    )
    pb_path    = os.path.join(dest_dir, "ssd_mobilenet_v2_coco.pb")
    pbtxt_path = os.path.join(dest_dir, "ssd_mobilenet_v2_coco.pbtxt")

    if not os.path.isfile(pbtxt_path):
        print("Downloading config (~8 KB)...")
        urllib.request.urlretrieve(pbtxt_url, pbtxt_path)
        print(f"  Saved → {pbtxt_path}")
    else:
        print(f"  Config already exists: {pbtxt_path}")

    if not os.path.isfile(pb_path):
        tar_path = os.path.join(dest_dir, "_tmp_mobilenet.tar.gz")
        print("Downloading model (~67 MB) — this may take a minute...")
        urllib.request.urlretrieve(pb_url, tar_path)
        print("  Extracting...")
        with tarfile.open(tar_path) as tar:
            for member in tar.getmembers():
                if member.name.endswith("frozen_inference_graph.pb"):
                    member.name = "frozen_inference_graph.pb"
                    tar.extract(member, dest_dir)
                    extracted = os.path.join(dest_dir, "frozen_inference_graph.pb")
                    os.rename(extracted, pb_path)
                    break
        os.remove(tar_path)
        print(f"  Saved → {pb_path}")
    else:
        print(f"  Model already exists: {pb_path}")

    print("\nAll model files ready.  Run: python main.py")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--download", action="store_true",
                   help="Download SSD MobileNet v2 COCO model files")
    args = p.parse_args()
    if args.download:
        download_models()
    else:
        p.print_help()
