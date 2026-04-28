"""
vision.py — Autonomous-car perception demo  (runs on LAPTOP, not on Pi)
=========================================================================
Simulates how a self-driving car sees the road: real-time object detection
with class-coloured bounding boxes, confidence labels, and a driving HUD.

Usage:
    python vision.py                        # default webcam (cam 0)
    python vision.py --source 0             # explicit webcam index
    python vision.py --source traffic.mp4   # video file (loops)
    python vision.py --model yolov8s        # larger = more accurate
    python vision.py --conf 0.40            # raise confidence threshold
    python vision.py --all-classes          # show every COCO class

Keys while running:
    Q / ESC   quit
    P         pause / resume
    S         save screenshot to working dir
    +  /  -   raise / lower confidence on-the-fly

Requirements (one-time):
    pip install ultralytics opencv-python
    The first run auto-downloads the YOLOv8 weights (~6 MB for nano).
"""

import argparse
import sys
import time
import cv2
import numpy as np
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    print("[vision] ultralytics not installed.")
    print("         Run:  pip install ultralytics opencv-python")
    sys.exit(1)


# ── Per-class colours (BGR) — mirrors the reference image palette ─────────────
_CLASS_COLORS: dict[str, tuple] = {
    "car":           (0,   255, 255),   # yellow
    "truck":         (0,   255,   0),   # green
    "bus":           (0,   165, 255),   # orange
    "motorcycle":    (255, 100,   0),   # blue
    "bicycle":       (0,   255, 255),   # yellow
    "person":        (0,   0,   255),   # red
    "traffic light": (255, 255,   0),   # cyan
    "stop sign":     (80,  80,  220),   # dark red
    "fire hydrant":  (0,   128, 255),   # orange
}
_DEFAULT_COLOR = (180, 180, 180)

# Only these classes are shown by default (everything a driver cares about)
_DRIVING_CLASSES = frozenset({
    "car", "truck", "bus", "motorcycle", "bicycle",
    "person", "traffic light", "stop sign", "fire hydrant",
})


def _color(label: str) -> tuple:
    return _CLASS_COLORS.get(label.lower(), _DEFAULT_COLOR)


# ── Drawing helpers ───────────────────────────────────────────────────────────

def _draw_box(frame: np.ndarray,
              label: str, conf: float,
              x1: int, y1: int, x2: int, y2: int,
              color: tuple) -> None:
    """Bounding box + filled label pill above it (matches reference style)."""
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    text = f"{label} {conf:.0%}"
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thick)
    pad = 4
    lx1, ly1 = x1, max(0, y1 - th - baseline - pad * 2)
    lx2, ly2 = x1 + tw + pad * 2, y1
    cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), color, -1)
    cv2.putText(frame, text, (lx1 + pad, ly2 - baseline - pad),
                font, scale, (0, 0, 0), thick, cv2.LINE_AA)


def _draw_hud(frame: np.ndarray,
              fps: float,
              counts: dict,
              conf_thresh: float,
              source_name: str,
              paused: bool) -> None:
    """Semi-transparent top bar (FPS / source) + right-side detection panel."""
    h, w = frame.shape[:2]

    # ── top bar ───────────────────────────────────────────────────────────────
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 40), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    cv2.putText(frame, f"FPS {fps:4.1f}", (10, 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 120), 2, cv2.LINE_AA)
    cv2.putText(frame, f"SRC: {source_name}", (130, 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(frame, f"CONF >{conf_thresh:.0%}", (w - 170, 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

    # ── right-side detection panel ────────────────────────────────────────────
    if counts:
        panel_w = 170
        rows = len(counts)
        row_h = 22
        panel_h = rows * row_h + 10
        overlay2 = frame.copy()
        cv2.rectangle(overlay2, (w - panel_w, 44), (w, 44 + panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay2, 0.55, frame, 0.45, 0, frame)
        for i, (cls, cnt) in enumerate(sorted(counts.items())):
            clr = _color(cls)
            cv2.putText(frame, f"{cls}: {cnt}",
                        (w - panel_w + 6, 44 + 16 + i * row_h),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, clr, 1, cv2.LINE_AA)

    # ── bottom label ─────────────────────────────────────────────────────────
    cv2.putText(frame, "AUTONOMOUS VISION  [P pause  S screenshot  Q quit]",
                (8, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 120), 1, cv2.LINE_AA)

    # ── PAUSED overlay ────────────────────────────────────────────────────────
    if paused:
        cv2.putText(frame, "PAUSED",
                    (w // 2 - 80, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 200, 255), 4, cv2.LINE_AA)


# ── Main loop ─────────────────────────────────────────────────────────────────

def run(source: str,
        model_name: str,
        conf: float,
        filter_driving: bool) -> None:

    print(f"[vision] Loading {model_name} …  (auto-downloads on first run)")
    model = YOLO(f"{model_name}.pt")

    is_webcam = source.isdigit()
    cap_src   = int(source) if is_webcam else source
    src_label = f"webcam({source})" if is_webcam else Path(source).name

    cap = cv2.VideoCapture(cap_src)
    if not cap.isOpened():
        print(f"[vision] ERROR: cannot open source '{source}'")
        sys.exit(1)

    cv2.namedWindow("Autonomous Vision", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Autonomous Vision", 1280, 720)
    print(f"[vision] Running on '{src_label}'  — Q/ESC quit  P pause  S save  +/- confidence")

    paused     = False
    frame      = None
    frame_n    = 0
    fps        = 0.0
    t_fps      = time.time()
    shot_n     = 0

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                if not is_webcam:           # loop video file
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break

            results  = model(frame, conf=conf, verbose=False)[0]
            counts: dict[str, int] = {}

            for box in results.boxes:
                cls_id = int(box.cls[0])
                label  = model.names[cls_id]
                if filter_driving and label not in _DRIVING_CLASSES:
                    continue
                c_val        = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                _draw_box(frame, label, c_val, x1, y1, x2, y2, _color(label))
                counts[label] = counts.get(label, 0) + 1

            frame_n += 1
            if frame_n % 15 == 0:
                now  = time.time()
                fps  = 15.0 / max(now - t_fps, 1e-6)
                t_fps = now

            _draw_hud(frame, fps, counts, conf, src_label, paused)

        if frame is not None:
            cv2.imshow("Autonomous Vision", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('p'):
            paused = not paused
        elif key == ord('s') and frame is not None:
            fname = f"vision_shot_{shot_n:03d}.jpg"
            cv2.imwrite(fname, frame)
            print(f"[vision] Screenshot saved → {fname}")
            shot_n += 1
        elif key == ord('+') or key == ord('='):
            conf = min(0.95, round(conf + 0.05, 2))
            print(f"[vision] Confidence threshold → {conf:.0%}")
        elif key == ord('-'):
            conf = max(0.05, round(conf - 0.05, 2))
            print(f"[vision] Confidence threshold → {conf:.0%}")

    cap.release()
    cv2.destroyAllWindows()
    print("[vision] Done.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="Autonomous-car vision demo — object detection on video or webcam",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python vision.py
  python vision.py --source traffic.mp4
  python vision.py --source 0 --model yolov8s --conf 0.45
  python vision.py --source dashcam.mp4 --all-classes
""")
    p.add_argument("--source", default="0",
                   help="0 for webcam, or path to a video file (default: 0)")
    p.add_argument("--model",
                   default="yolov8m",
                   choices=["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"],
                   help="YOLOv8 model size: n=nano (fastest) … x=xlarge (most accurate)")
    p.add_argument("--conf", type=float, default=0.35,
                   help="Detection confidence threshold (default 0.35)")
    p.add_argument("--all-classes", action="store_true",
                   help="Show all 80 COCO classes (default: driving-relevant only)")
    args = p.parse_args()

    run(
        source=args.source,
        model_name=args.model,
        conf=args.conf,
        filter_driving=not args.all_classes,
    )


if __name__ == "__main__":
    main()
