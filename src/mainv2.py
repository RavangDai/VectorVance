"""
mainv2.py - VectorVance Autonomous Car (PC TESTING VERSION)
────────────────────────────────────────────────────────────
Same logic as main.py but uses webcam or video file instead
of Picamera2. No GPIO / motor / ultrasonic hardware required.

USAGE:
  python mainv2.py                               → interactive menu
  python mainv2.py --webcam                      → webcam directly
  python mainv2.py --video path/to/video.mp4     → video file
  python mainv2.py --dnn --target-color GREEN    → DNN + green tape

CONTROLS:
  Q           Quit
  SPACE       Toggle autonomous
  R           Reset all
  S           Snapshot
  D           Debug detectors
  G / B / E   Set target tape colour Green / Blue / rEd
  P           Pause / Resume  (video only)
  LEFT/RIGHT  Skip ±5 s      (video only)
"""

import cv2
import sys
import os
import glob
import time
import argparse

from perception import LaneDetector, SmoothValue
from controller import PIDController
from speed_controller import AdaptiveSpeedController, draw_speed_indicator
from safety import ObstacleDetector
from sign_detector import TrafficSignDetector
from dnn_detector import DNNDetector
from intersection_detector import IntersectionDetector
from color_sign_detector import ColorSignDetector


# ── Video source picker ───────────────────────────────────────────────────────
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm'}
_COLOR_FOLLOW_FRAMES = 45


def find_videos(directory):
    videos = []
    if not os.path.isdir(directory):
        return videos
    for ext in VIDEO_EXTENSIONS:
        videos.extend(glob.glob(os.path.join(directory, f'*{ext}')))
        videos.extend(glob.glob(os.path.join(directory, f'**/*{ext}'), recursive=True))
    return sorted(set(videos))


def pick_source_interactive(default_video_dir='test_videos'):
    print("=" * 60)
    print("   VectorVance — Input Source Selector")
    print("=" * 60)
    print("\n  [0]  Webcam (default camera)\n")

    videos = find_videos(default_video_dir)
    for ext in VIDEO_EXTENSIONS:
        videos.extend(glob.glob(f'*{ext}'))
    videos = sorted(set(videos))

    if videos:
        print(f"  Found {len(videos)} video(s):")
        for i, v in enumerate(videos, start=1):
            size_mb = os.path.getsize(v) / (1024 * 1024)
            print(f"  [{i}]  {v}  ({size_mb:.1f} MB)")
    else:
        print(f"  No videos found in '{default_video_dir}/' or current dir.")

    print("\n  [p]  Enter a custom path")
    print("  [q]  Quit\n")

    while True:
        choice = input("  Select source: ").strip().lower()
        if choice == 'q':
            sys.exit(0)
        elif choice == 'p':
            path = input("  Path: ").strip().strip('"\'')
            if os.path.isfile(path):
                return 'video', path
            print(f"  Not found: {path}")
        elif choice in ('0', ''):
            return 'webcam', 0
        else:
            try:
                idx = int(choice)
                if 1 <= idx <= len(videos):
                    return 'video', videos[idx - 1]
                print(f"  Pick 0–{len(videos)}")
            except ValueError:
                print("  Invalid choice.")


# ── Autonomous vehicle (PC version — no hardware) ─────────────────────────────
class AutonomousVehicle:

    def __init__(self, max_speed=0.8, enable_dnn=True,
                 dnn_model="ssd_mobilenet_v2_coco.pb", target_color="GREEN"):

        self.perception    = LaneDetector(width=640, height=480)
        self.steering      = PIDController(Kp=0.003, Ki=0.0001, Kd=0.001)
        self.speed_control = AdaptiveSpeedController(min_speed=0.2, max_speed=max_speed)
        self.safety        = ObstacleDetector(emergency_distance=20, warning_distance=50)
        self.intersection_detector = IntersectionDetector()

        # ── Sign / obstacle detector ──────────────────────────────────
        self.dnn_enabled = enable_dnn
        if enable_dnn:
            self.detector = DNNDetector(model_name=dnn_model, skip_frames=3)
        else:
            self.detector = TrafficSignDetector()
            print("[Detector] Classic stop-sign detector (--dnn to enable DNN)")

        # ── Colour tape navigator ─────────────────────────────────────
        self.color_detector       = ColorSignDetector(frame_width=640, frame_height=480)
        self.color_detector.set_target(target_color)
        self._color_follow_frames = 0

        # ── Runtime state ─────────────────────────────────────────────
        self.autonomous_enabled  = True
        self.current_speed_limit = max_speed
        self.stop_sign_timer     = 0
        self.stop_sign_cooldown  = 0
        self.frame_count         = 0
        self.total_error         = 0
        self.stop_signs_detected = 0

        # ── Smooth display values ─────────────────────────────────────
        self.smooth_left       = SmoothValue(0.0, alpha=0.18)
        self.smooth_right      = SmoothValue(0.0, alpha=0.18)
        self.smooth_base_speed = SmoothValue(0.0, alpha=0.15)
        self.smooth_pid        = SmoothValue(0.0, alpha=0.12)

        # ── Status hold ──────────────────────────────────────────────
        self._display_status     = "READY"
        self._status_hold_frames = 0
        self._STATUS_MIN_HOLD    = 6
        self._source_label       = ""

    # ── Status hold ───────────────────────────────────────────────────────────

    def _update_display_status(self, new_status):
        if new_status == self._display_status:
            self._status_hold_frames = self._STATUS_MIN_HOLD
            return self._display_status
        self._status_hold_frames -= 1
        if self._status_hold_frames <= 0:
            self._display_status     = new_status
            self._status_hold_frames = self._STATUS_MIN_HOLD
        return self._display_status

    # ── Frame processing ──────────────────────────────────────────────────────

    def process_frame(self, frame):
        self.frame_count += 1
        steering_error, vision_frame = self.perception.process_frame(frame)

        if steering_error is None:
            self.smooth_left.update(0.0)
            self.smooth_right.update(0.0)
            self.smooth_base_speed.update(0.0)
            self.smooth_pid.update(0.0)
            return (
                self._create_debug_frame(
                    vision_frame, 0, 0.0, 0.0, 0.0, 0.0, "EMERGENCY STOP", "NONE"
                ),
                (0.0, 0.0, "EMERGENCY STOP"),
            )

        self.total_error += abs(steering_error)

        # ── Detection ─────────────────────────────────────────────────
        if self.dnn_enabled:
            self.detector.detect(frame)
            obstacle_modifier = self.detector.get_speed_modifier()
        else:
            self.detector.detect_signs(frame)
            obstacle_modifier = 1.0

        sign_action, _ = self.detector.get_action()

        if self.stop_sign_cooldown > 0:
            self.stop_sign_cooldown -= 1
        if sign_action == "STOP" and self.stop_sign_cooldown == 0:
            if self.stop_sign_timer == 0:
                self.stop_sign_timer    = 60
                self.stop_sign_cooldown = 120
                self.stop_signs_detected += 1
                print("STOP SIGN — holding 2 s")

        # ── Colour tape ───────────────────────────────────────────────
        self.color_detector.detect(frame)

        # ── Intersection detection ────────────────────────────────────
        num_raw_lines = 0
        lane_width    = None
        gray  = cv2.cvtColor(cv2.resize(frame, (640, 480)), cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
        lines = cv2.HoughLinesP(edges, 2, 3.14159 / 180, 30,
                                minLineLength=40, maxLineGap=150)
        if lines is not None:
            num_raw_lines = len(lines)
        if (self.perception.ema_left_fit is not None and
                self.perception.ema_right_fit is not None):
            y_eval  = int(480 * 0.75)
            left_x  = self.perception._eval_fit(self.perception.ema_left_fit,  y_eval)
            right_x = self.perception._eval_fit(self.perception.ema_right_fit, y_eval)
            if left_x is not None and right_x is not None:
                lane_width = right_x - left_x

        fork_detected = self.intersection_detector.update(
            num_lines        = num_raw_lines,
            left_confidence  = self.perception.left_confidence,
            right_confidence = self.perception.right_confidence,
            lane_width       = lane_width,
            left_fit         = self.perception.ema_left_fit,
            right_fit        = self.perception.ema_right_fit,
        )

        if fork_detected:
            if self.color_detector.target_visible():
                self._color_follow_frames = _COLOR_FOLLOW_FRAMES
                print(f"[Fork] → {self.color_detector.target_color} "
                      f"({self.color_detector.target_det['side']})")
            else:
                print("[Fork] Detected — target tape not visible, continuing")

        # ── Speed ─────────────────────────────────────────────────────
        if self.stop_sign_timer > 0:
            base_speed = 0.0
            self.stop_sign_timer -= 1
            status = f"STOPPED ({self.stop_sign_timer})"
        else:
            base_speed = self.speed_control.calculate_speed(steering_error, obstacle_modifier)
            base_speed = min(base_speed, self.current_speed_limit)
            status     = self.speed_control.get_speed_category(
                abs(steering_error)
            ).replace("_", " ")

        # ── Steering ─────────────────────────────────────────────────
        if self.autonomous_enabled and base_speed > 0:
            if self._color_follow_frames > 0:
                self._color_follow_frames -= 1
                offset = self.color_detector.get_steering_offset()
                if offset is not None:
                    steer      = offset * 0.40
                    left_speed  = max(0.0, min(1.0, base_speed + steer))
                    right_speed = max(0.0, min(1.0, base_speed - steer))
                    pid_output  = steer
                    status = (f"COLOR: {self.color_detector.target_color} "
                              f"→ {self.color_detector.target_det['side']}")
                else:
                    pid_output  = self.steering.compute(steering_error)
                    left_speed  = max(0.0, min(1.0, base_speed * 0.5 + pid_output))
                    right_speed = max(0.0, min(1.0, base_speed * 0.5 - pid_output))
                    status      = "COLOR: SCANNING..."
            else:
                pid_output  = self.steering.compute(steering_error)
                left_speed  = max(0.0, min(1.0, base_speed + pid_output))
                right_speed = max(0.0, min(1.0, base_speed - pid_output))
        else:
            pid_output  = 0.0
            left_speed  = 0.0
            right_speed = 0.0

        self.smooth_left.update(left_speed)
        self.smooth_right.update(right_speed)
        self.smooth_base_speed.update(base_speed)
        self.smooth_pid.update(pid_output)

        debug_frame = self._create_debug_frame(
            vision_frame, steering_error, pid_output,
            base_speed, left_speed, right_speed, status, sign_action
        )
        return debug_frame, (left_speed, right_speed, status)

    # ── HUD ───────────────────────────────────────────────────────────────────

    def _create_debug_frame(self, vision_frame, error, pid_output,
                            base_speed, left_speed, right_speed, status, sign_action):
        frame = vision_frame.copy()
        frame = self.safety.draw_overlay(frame)
        frame = self.detector.draw_overlay(frame)
        frame = self.color_detector.draw_overlay(frame)
        frame = self._draw_motor_bars(
            frame,
            self.smooth_left.value,
            self.smooth_right.value,
            self.smooth_pid.value,
        )
        frame = draw_speed_indicator(
            frame,
            self.smooth_base_speed.value,
            self.speed_control.target_speed,
            self.speed_control.get_speed_category(abs(error)),
        )

        display_status = self._update_display_status(status)
        status_color = (
            (0,   0,   255) if "STOP"   in display_status else
            (0,   200, 255) if "COLOR:" in display_status else
            (255, 255, 255)
        )
        cv2.putText(frame, f"Status: {display_status}",
                    (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

        if sign_action == "STOP":
            cv2.putText(frame, "STOP DETECTED!",
                        (10, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
        elif self.dnn_enabled:
            danger = self.detector.get_danger_level()
            danger_colors = {
                "CLEAR":   (100, 200, 100),
                "CAUTION": (0,   200, 255),
                "DANGER":  (0,   100, 255),
                "STOP":    (0,   0,   255),
            }
            cv2.putText(frame, f"DNN: {danger}",
                        (10, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        danger_colors.get(danger, (150, 150, 150)), 1)
        else:
            cv2.putText(frame, "Scanning...",
                        (10, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

        conf = self.intersection_detector.fork_confidence
        if conf > 0.2:
            fork_color = (0, 255, 255) if conf > 0.45 else (100, 200, 200)
            cv2.putText(frame, f"Fork: {conf:.0%}",
                        (10, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.4, fork_color, 1)

        if self._source_label:
            cv2.putText(frame, self._source_label,
                        (frame.shape[1] - 250, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1)

        return frame

    def _draw_motor_bars(self, frame, left_speed, right_speed, pid_output):
        h, w        = frame.shape[:2]
        bar_width   = 40
        bar_height  = 200
        bar_x_left  = w - 120
        bar_x_right = w - 60
        bar_y       = h - bar_height - 50

        for bx in (bar_x_left, bar_x_right):
            cv2.rectangle(frame, (bx, bar_y),
                          (bx + bar_width, bar_y + bar_height), (50, 50, 50), -1)

        left_fill  = int(bar_height * max(0.0, min(1.0, left_speed)))
        right_fill = int(bar_height * max(0.0, min(1.0, right_speed)))
        bar_color  = (0, 255, 0) if abs(pid_output) < 0.1 else (0, 165, 255)

        if left_fill:
            cv2.rectangle(frame,
                          (bar_x_left,  bar_y + bar_height - left_fill),
                          (bar_x_left  + bar_width, bar_y + bar_height), bar_color, -1)
        if right_fill:
            cv2.rectangle(frame,
                          (bar_x_right, bar_y + bar_height - right_fill),
                          (bar_x_right + bar_width, bar_y + bar_height), bar_color, -1)

        for bx, lbl in ((bar_x_left, "L"), (bar_x_right, "R")):
            cv2.putText(frame, lbl, (bx + 12, bar_y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"{left_speed:.2f}",
                    (bar_x_left,  bar_y + bar_height + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"{right_speed:.2f}",
                    (bar_x_right, bar_y + bar_height + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return frame

    # ── Reset ─────────────────────────────────────────────────────────────────

    def _reset_all(self):
        self.steering.reset()
        self.speed_control.reset()
        self.safety.reset()
        self.perception.reset_smoothing()
        self.detector.reset()
        self.color_detector.reset()
        self.intersection_detector.reset()
        self.current_speed_limit  = 0.8
        self.stop_sign_timer      = 0
        self._color_follow_frames = 0
        self.smooth_left.set_immediate(0.0)
        self.smooth_right.set_immediate(0.0)
        self.smooth_base_speed.set_immediate(0.0)
        self.smooth_pid.set_immediate(0.0)
        self._display_status = "READY"
        print("All systems reset")

    # ── Run loop ──────────────────────────────────────────────────────────────

    def run(self, source_type='webcam', source_value=0, rotate_frame=False):
        is_video   = (source_type == 'video')
        total_frames = 1

        if is_video:
            if not os.path.isfile(source_value):
                print(f"File not found: {source_value}")
                return
            cap         = cv2.VideoCapture(source_value)
            video_fps   = cap.get(cv2.CAP_PROP_FPS) or 30
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self._source_label = f"FILE: {os.path.basename(source_value)}"
            print(f"Video: {source_value} | FPS:{video_fps:.0f} | Frames:{total_frames}")
        else:
            cap = cv2.VideoCapture(source_value)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            video_fps          = 30
            self._source_label = "WEBCAM"

        if not cap.isOpened():
            print("Could not open source.")
            return

        time.sleep(0.5)
        print(f"Source ready | target tape: {self.color_detector.target_color}")
        print("Keys: [Q] Quit  [SPACE] Auto  [R] Reset  [S] Snap  "
              "[D] Debug  [G/B/E] Tape  "
              + ("[P] Pause  [←/→] Skip" if is_video else ""))

        start_time = time.time()
        paused     = False
        loop_video = True

        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    if is_video and loop_video:
                        print("End of video — looping")
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        self._reset_all()
                        continue
                    break

                if rotate_frame:
                    frame = cv2.rotate(frame, cv2.ROTATE_180)

                debug_frame, (left, right, status) = self.process_frame(frame)

                # Video progress bar
                if is_video:
                    curr = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                    pct  = curr / max(total_frames, 1)
                    h, w = debug_frame.shape[:2]
                    cv2.rectangle(debug_frame, (0, h - 8), (w, h), (30, 30, 30), -1)
                    cv2.rectangle(debug_frame, (0, h - 8),
                                  (int(w * pct), h), (0, 200, 255), -1)

            title = ("VectorVance  "
                     + ("AUTO" if self.autonomous_enabled else "MANUAL")
                     + (" [DNN]" if self.dnn_enabled else "")
                     + f"  tape:{self.color_detector.target_color}")
            cv2.imshow(title, debug_frame)

            if self.frame_count % 30 == 0 and not paused:
                elapsed = time.time() - start_time
                fps     = self.frame_count / max(elapsed, 0.1)
                avg_err = self.total_error / max(self.frame_count, 1)
                print(f"Frame {self.frame_count:04d} | "
                      f"L:{left:.2f} R:{right:.2f} | "
                      f"{status:35s} | FPS:{fps:.1f} | AvgErr:{avg_err:.1f}px")

            wait_ms = int(1000 / video_fps) if is_video and not paused else 1
            key = cv2.waitKey(wait_ms) & 0xFF

            if   key == ord('q'):
                break
            elif key == ord(' '):
                self.autonomous_enabled = not self.autonomous_enabled
                print(f"Autonomous: {'ON' if self.autonomous_enabled else 'OFF'}")
            elif key == ord('r'):
                self._reset_all()
                if is_video:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            elif key == ord('s'):
                fname = f"snap_{self.frame_count:04d}.jpg"
                cv2.imwrite(fname, debug_frame)
                print(f"Snapshot: {fname}")
            elif key == ord('d'):
                if self.dnn_enabled:
                    print(f"DNN: {len(self.detector.all_detections)} dets | "
                          f"danger={self.detector.get_danger_level()}")
                    for d in self.detector.all_detections:
                        print(f"  {d[1]} {d[3]:.0%}")
                print(f"Tape: {self.color_detector.detections}")
            elif key == ord('g'):
                self.color_detector.set_target("GREEN")
            elif key == ord('b'):
                self.color_detector.set_target("BLUE")
            elif key == ord('e'):
                self.color_detector.set_target("RED")
            elif key == ord('p') and is_video:
                paused = not paused
                print("PAUSED" if paused else "RESUMED")
            elif key == ord('l') and is_video:
                loop_video = not loop_video
                print(f"Loop: {'ON' if loop_video else 'OFF'}")
            elif key == 81 and is_video:
                pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
                cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, pos - video_fps * 5))
            elif key == 83 and is_video:
                pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
                cap.set(cv2.CAP_PROP_POS_FRAMES, min(total_frames, pos + video_fps * 5))

        cap.release()
        cv2.destroyAllWindows()

        duration = time.time() - start_time
        print("=" * 60)
        print(f"Duration    : {duration:.1f}s")
        print(f"Frames      : {self.frame_count}")
        print(f"Avg FPS     : {self.frame_count / max(duration, 0.1):.1f}")
        print(f"Avg error   : {self.total_error / max(self.frame_count, 1):.1f}px")
        print(f"Stop signs  : {self.stop_signs_detected}")
        print("=" * 60)


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description="VectorVance — PC Testing")
    p.add_argument("--webcam",        action="store_true")
    p.add_argument("--video",         type=str,   default=None)
    p.add_argument("--dir",           type=str,   default="test_videos")
    p.add_argument("--speed",         type=float, default=0.8)
    p.add_argument("--rotate",        action="store_true")
    p.add_argument("--no-rotate",     action="store_true")
    p.add_argument("--dnn",           action="store_true")
    p.add_argument("--dnn-model",     type=str,   default="ssd_mobilenet_v2_coco.pb")
    p.add_argument("--target-color",  type=str,   default="GREEN",
                   choices=["GREEN", "BLUE", "RED"],
                   help="Tape colour to follow at forks (default: GREEN)")
    args = p.parse_args()

    vehicle = AutonomousVehicle(
        max_speed    = args.speed,
        enable_dnn   = args.dnn,
        dnn_model    = args.dnn_model,
        target_color = args.target_color,
    )

    if args.video:
        vehicle.run(source_type='video', source_value=args.video,
                    rotate_frame=args.rotate)
    elif args.webcam:
        vehicle.run(source_type='webcam', source_value=0,
                    rotate_frame=not args.no_rotate)
    else:
        source_type, source_value = pick_source_interactive(args.dir)
        rotate = args.rotate or (source_type == 'webcam' and not args.no_rotate)
        vehicle.run(source_type=source_type, source_value=source_value,
                    rotate_frame=rotate)


if __name__ == "__main__":
    main()
