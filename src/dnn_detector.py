"""
dnn_detector.py — DNN stop-sign confirmation layer
────────────────────────────────────────────────────
Two-stage detection pipeline:
  Stage 1 (every frame, < 1 ms):
      TrafficSignDetector in sign_detector.py
      → HSV red mask + octagon shape → candidate found?

  Stage 2 (this file — only when Stage 1 flags red, every 15 frames):
      SSD MobileNet v2 COCO → confirms it really is a stop sign
      → DNN never runs on an empty scene → near-zero FPS cost

Why keep DNN at all?
  • Satisfies the AI/ML coursework requirement
  • Adds a true neural-network second opinion, reducing false positives
  • "We use a hybrid sensor-fusion pipeline: classical CV for real-time
    candidate detection, and a pretrained SSD MobileNet v2 DNN for
    confirmation — the DNN is gated by the CV stage so it only
    activates on frames that already contain a sign candidate."

Model download (one-time):
    python dnn_detector.py --download
"""

import os
import cv2
import numpy as np

# TF COCO class ID for stop sign (1-indexed)
_TF_STOP_SIGN       = 13
_STOP_HISTORY_LEN   = 5
_STOP_CONFIRM_COUNT = 2

_INPUT_SIZE = (300, 300)


class StopSignConfirmer:
    """
    DNN confirmation layer for stop signs.

    Runs SSD MobileNet v2 ONLY when the fast CV stage
    has already flagged a red octagon in the frame.
    This keeps average DNN cost near zero.

    detect(frame, red_hint)   → list of detections
    stop_sign_detected()      → bool (confirmed by DNN)
    draw_overlay(frame)       → annotated frame
    reset()                   → clear state
    """

    def __init__(self,
                 model_name: str       = "ssd_mobilenet_v2_coco.pb",
                 conf_threshold: float = 0.55,
                 frame_width: int      = 640,
                 frame_height: int     = 480,
                 skip_frames: int      = 15):
        """
        skip_frames  : DNN runs at most every N frames
        red_hint     : when False AND not on an inference frame, skip entirely
                       → DNN load is (1/skip_frames) × (fraction of frames with red)
        """
        self.conf_threshold = conf_threshold
        self.frame_width    = frame_width
        self.frame_height   = frame_height
        self.skip_frames    = max(1, skip_frames)
        self._frame_counter = 0
        self.available      = False

        self.all_detections: list = []   # (label, bbox, conf)
        self._stop_history:  list = []
        self._stop_confirmed: bool = False

        pb_path    = model_name
        pbtxt_path = os.path.splitext(model_name)[0] + ".pbtxt"

        if not os.path.isfile(pb_path) or not os.path.isfile(pbtxt_path):
            print(f"[DNN] Model not found: {pb_path}")
            print("[DNN] Run: python dnn_detector.py --download")
            print("[DNN] Running in disabled mode — CV-only stop sign detection")
            return

        print(f"[DNN] Loading SSD MobileNet v2 (stop-sign confirmer) ...")
        try:
            self._net = cv2.dnn.readNetFromTensorflow(pb_path, pbtxt_path)
            self._net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self._net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            self.available = True
            print(f"[DNN] Ready — confirming stop signs every {skip_frames} frames "
                  f"(gated by CV red-hint)")
        except cv2.error as e:
            print(f"[DNN] Failed to load model: {e}")
            print("[DNN] Running in disabled mode")

    # ── Public API ────────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray, red_hint: bool = True) -> list:
        """
        Run DNN inference.

        red_hint : True when the CV stage already found a red region this frame.
                   When False, the DNN skips unless it's scheduled inference frame.
                   This dramatically reduces CPU load when no signs are present.

        Returns cached detections on skipped frames.
        """
        self._frame_counter += 1

        if not self.available:
            self._update_stop_history(False)
            return []

        # Only run inference if:
        #   a) it's a scheduled inference frame  (every skip_frames)
        #   b) OR the CV stage flagged something red (immediate confirmation)
        is_scheduled = (self._frame_counter % self.skip_frames == 0)
        if not is_scheduled and not red_hint:
            self._update_stop_history(False)
            return self.all_detections   # cached

        self.all_detections = []
        found_stop = False

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
            tf_cls = int(raw[0, 0, i, 1])

            if conf < self.conf_threshold or tf_cls != _TF_STOP_SIGN:
                continue

            x1 = int(raw[0, 0, i, 3] * self.frame_width)
            y1 = int(raw[0, 0, i, 4] * self.frame_height)
            x2 = int(raw[0, 0, i, 5] * self.frame_width)
            y2 = int(raw[0, 0, i, 6] * self.frame_height)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(self.frame_width - 1, x2), min(self.frame_height - 1, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            bbox = (x1, y1, x2 - x1, y2 - y1)
            self.all_detections.append(("stop sign", bbox, conf))
            found_stop = True

        self._update_stop_history(found_stop)
        return self.all_detections

    def stop_sign_detected(self) -> bool:
        return self._stop_confirmed

    def draw_overlay(self, frame: np.ndarray) -> np.ndarray:
        for label, (x, y, w, h), conf in self.all_detections:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 200), 2)
            text = f"DNN:{label} {conf:.0%}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x, y - th - 6), (x + tw + 6, y), (0, 0, 180), -1)
            cv2.putText(frame, text, (x + 3, y - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return frame

    def reset(self):
        self.all_detections  = []
        self._stop_history   = []
        self._stop_confirmed = False
        self._frame_counter  = 0

    # ── Internal ──────────────────────────────────────────────────────────────

    def _update_stop_history(self, found: bool):
        self._stop_history.append(found)
        if len(self._stop_history) > _STOP_HISTORY_LEN:
            self._stop_history.pop(0)
        self._stop_confirmed = (
            sum(self._stop_history[-_STOP_HISTORY_LEN:]) >= _STOP_CONFIRM_COUNT
        )


# ── Model download ────────────────────────────────────────────────────────────

def download_models(dest_dir: str = "."):
    import urllib.request, tarfile

    pbtxt_url = (
        "https://raw.githubusercontent.com/opencv/opencv_extra/"
        "master/testdata/dnn/ssd_mobilenet_v2_coco_2018_03_29.pbtxt"
    )
    pb_url    = (
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
        print(f"  Config exists: {pbtxt_path}")

    if not os.path.isfile(pb_path):
        tar_path = os.path.join(dest_dir, "_tmp_mobilenet.tar.gz")
        print("Downloading SSD MobileNet v2 (~67 MB)...")
        urllib.request.urlretrieve(pb_url, tar_path)
        with tarfile.open(tar_path) as tar:
            for member in tar.getmembers():
                if member.name.endswith("frozen_inference_graph.pb"):
                    member.name = "frozen_inference_graph.pb"
                    tar.extract(member, dest_dir)
                    os.rename(
                        os.path.join(dest_dir, "frozen_inference_graph.pb"), pb_path
                    )
                    break
        os.remove(tar_path)
        print(f"  Saved → {pb_path}")
    else:
        print(f"  Model exists: {pb_path}")

    print("\nReady. Run: python main.py")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--download", action="store_true")
    args = p.parse_args()
    if args.download:
        download_models()
    else:
        p.print_help()
