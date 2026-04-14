"""
camera_viewer.py — Live camera feed viewer for alignment and calibration
─────────────────────────────────────────────────────────────────────────
Shows two windows side by side:
  LEFT  — raw feed straight from camera
  RIGHT — corrected feed (180° rotation + fisheye undistortion)
           This is exactly what the car sees during autonomous mode.

Use this to:
  • Physically adjust camera angle/position
  • Verify the 180° rotation looks correct
  • Check undistortion is not over/under-correcting

KEYBOARD
  Q / ESC   quit
  R         toggle 180° rotation on/off
  U         toggle fisheye undistortion on/off
  G         toggle calibration grid overlay
  S         save a snapshot pair (raw + corrected)
  +  /  -   adjust FOV (affects undistortion strength)
  0         reset FOV to default (130°)
  B         cycle balance  0.0 → 0.3 → 0.5 → 0.7 → 1.0 → 0.0

USAGE
  python camera_viewer.py
  python camera_viewer.py --cam-index 1
  python camera_viewer.py --fov 120
  python camera_viewer.py --calibration calibration.npz
  python camera_viewer.py --no-undistort
"""

import cv2
import numpy as np
import argparse
import time
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from camera import FisheyeCamera


# ── Grid overlay helper ───────────────────────────────────────────────────────

def draw_grid(frame, rows=6, cols=8, color=(0, 80, 0)):
    h, w = frame.shape[:2]
    for r in range(1, rows):
        y = int(h * r / rows)
        cv2.line(frame, (0, y), (w, y), color, 1)
    for c in range(1, cols):
        x = int(w * c / cols)
        cv2.line(frame, (x, 0), (x, h), color, 1)
    # centre crosshair
    cx, cy = w // 2, h // 2
    cv2.line(frame, (cx - 20, cy), (cx + 20, cy), (0, 255, 0), 2)
    cv2.line(frame, (cx, cy - 20), (cx, cy + 20), (0, 255, 0), 2)
    # horizon reference (40% from top — approximate lane ROI start)
    hy = int(h * 0.40)
    cv2.line(frame, (0, hy), (w, hy), (0, 180, 255), 1)
    cv2.putText(frame, "horizon ref", (4, hy - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 180, 255), 1)
    return frame


def draw_hud(frame, label, extras=None):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 22), (0, 0, 0), -1)
    cv2.putText(frame, label, (6, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    if extras:
        cv2.putText(frame, extras, (w - 200, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
    return frame


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Camera alignment & calibration viewer")
    p.add_argument("--cam-index",    type=int,   default=-1,
                   help="Camera device index (default: auto-detect 0-3)")
    p.add_argument("--fov",          type=float, default=130.0,
                   help="Camera FOV in degrees (default: 130)")
    p.add_argument("--no-undistort", action="store_true",
                   help="Start with undistortion OFF")
    p.add_argument("--calibration",  type=str,   default=None,
                   help="Path to calibration .npz file")
    args = p.parse_args()

    # ── Open camera ──────────────────────────────────────────────────────
    cap = None
    search = [args.cam_index] if args.cam_index >= 0 else range(4)
    for idx in search:
        c = cv2.VideoCapture(idx)
        if c.isOpened():
            cap = c
            print(f"[Camera] Found at index {idx}")
            break
        c.release()

    if cap is None:
        print("[Camera] No camera found. Check connection.")
        print("         Try: python camera_viewer.py --cam-index 0")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # ── Load undistortion model ───────────────────────────────────────────
    fov        = args.fov
    undistort  = not args.no_undistort   # start OFF so you see raw first
    show_grid  = False
    rotate_180 = True
    balance    = 0.5
    balance_steps = [0.0, 0.3, 0.5, 0.7, 1.0]

    def make_camera(fov_deg, bal):
        if args.calibration:
            return FisheyeCamera.from_file(args.calibration, balance=bal)
        return FisheyeCamera(fov_deg=fov_deg, balance=bal)

    fisheye = make_camera(fov, balance)

    snap_count = 0

    print("\n── Controls ─────────────────────────────────")
    print("  Q / ESC  quit")
    print("  R        toggle 180° rotation")
    print("  U        toggle undistortion  ← start here, compare raw vs corrected")
    print("  G        toggle grid overlay")
    print("  S        save snapshot")
    print("  + / -    adjust FOV (currently %.0f°)" % fov)
    print("  0        reset FOV to 130°")
    print("  B        cycle balance (crop vs keep borders)")
    print("─────────────────────────────────────────────")
    print("TIP: start with U=OFF. If raw looks straight, skip undistortion.")
    print("     If lanes curve inward/outward, enable U and press +/- to tune FOV.\n")

    prev_time = time.time()
    fps_display = 0.0

    while True:
        ret, raw = cap.read()
        if not ret:
            print("[Camera] Frame grab failed — retrying...")
            continue

        # ── Build corrected frame ─────────────────────────────────────
        corrected = raw.copy()

        if rotate_180:
            corrected = cv2.rotate(corrected, cv2.ROTATE_180)

        if undistort:
            corrected = fisheye.undistort(corrected)

        if show_grid:
            draw_grid(raw.copy())   # grid on raw too
            draw_grid(corrected)

        # ── FPS ───────────────────────────────────────────────────────
        now      = time.time()
        fps_display = 0.9 * fps_display + 0.1 * (1.0 / max(now - prev_time, 1e-6))
        prev_time = now

        # ── HUD labels ────────────────────────────────────────────────
        raw_disp = raw.copy()
        if show_grid:
            draw_grid(raw_disp)
        draw_hud(raw_disp, "RAW  (straight from camera)",
                 f"FPS {fps_display:.1f}")

        rot_str   = "ROT:ON" if rotate_180 else "ROT:OFF"
        und_str   = f"UND:ON fov={fov:.0f} bal={balance:.1f}" if undistort else "UND:OFF"
        corr_disp = corrected.copy()
        if show_grid:
            draw_grid(corr_disp)
        draw_hud(corr_disp, "CORRECTED  (what the car sees)",
                 f"{rot_str}  {und_str}")

        # ── Stack side by side ────────────────────────────────────────
        combined = np.hstack([raw_disp, corr_disp])
        cv2.putText(combined,
                    "Q=quit  R=rotate  U=undistort  G=grid  S=snap  +/-=FOV  0=reset  B=balance",
                    (6, combined.shape[0] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1)

        cv2.imshow("VectorVance Camera Viewer", combined)

        # ── Key handling ─────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'), 27):   # Q or ESC
            break

        elif key == ord('r'):
            rotate_180 = not rotate_180
            print(f"[R] 180° rotation: {'ON' if rotate_180 else 'OFF'}")

        elif key == ord('u'):
            undistort = not undistort
            print(f"[U] Undistortion: {'ON' if undistort else 'OFF'}")

        elif key == ord('g'):
            show_grid = not show_grid
            print(f"[G] Grid: {'ON' if show_grid else 'OFF'}")

        elif key == ord('s'):
            ts = time.strftime("%H%M%S")
            raw_path  = f"snap_{ts}_raw.jpg"
            corr_path = f"snap_{ts}_corrected.jpg"
            cv2.imwrite(raw_path,  raw)
            cv2.imwrite(corr_path, corrected)
            snap_count += 1
            print(f"[S] Saved: {raw_path}  {corr_path}")

        elif key == ord('+') or key == ord('='):
            fov = min(fov + 5, 220)
            fisheye = make_camera(fov, balance)
            print(f"[FOV] {fov:.0f}°")

        elif key == ord('-'):
            fov = max(fov - 5, 60)
            fisheye = make_camera(fov, balance)
            print(f"[FOV] {fov:.0f}°")

        elif key == ord('0'):
            fov = 130.0
            fisheye = make_camera(fov, balance)
            print(f"[FOV] reset to {fov:.0f}°")

        elif key == ord('b'):
            idx     = balance_steps.index(balance) if balance in balance_steps else 0
            balance = balance_steps[(idx + 1) % len(balance_steps)]
            fisheye = make_camera(fov, balance)
            print(f"[B] balance={balance:.1f}  "
                  f"(0.0=crop tight, 0.5=balanced, 1.0=keep all pixels)")

    cap.release()
    cv2.destroyAllWindows()
    print(f"Done. Snapshots saved: {snap_count}")


if __name__ == "__main__":
    main()
