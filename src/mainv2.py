"""
mainv2.py - VectorVance Autonomous Car (FULL NAVIGATION VERSION)
────────────────────────────────────────────────────────────────
Lane detection + PID steering + STOP signs + PATHFINDING + NAVIGATION

USAGE:
  python mainv2.py                               → interactive menu
  python mainv2.py --webcam                      → webcam only
  python mainv2.py --video test_videos/finalt.mp4 → stock footage

NAVIGATION MODE:
  python mainv2.py --nav                         → enable pathfinding navigation
  python mainv2.py --nav --track complex         → use complex track layout
  python mainv2.py --video X.mp4 --nav           → test nav overlay on video

CONTROLS:
  Q         Quit
  SPACE     Toggle autonomous
  R         Reset all
  S         Snapshot
  D         Debug signs
  N         Toggle navigation overlay
  P         Pause/Resume (video only)
  LEFT/RIGHT  Skip 5s (video only)
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
from intersection_detector import IntersectionDetector
from track_map import build_default_track, build_complex_track
from pathfinder import find_shortest_path, get_all_routes, print_route
from navigator import Navigator, NavigatorState


# ─────────────────────────────────────────────────────────────────────
#  INPUT SOURCE PICKER
# ─────────────────────────────────────────────────────────────────────
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm'}


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
    print()
    print("  [0]  Webcam (default camera)")
    print()

    videos = find_videos(default_video_dir)
    for ext in VIDEO_EXTENSIONS:
        videos.extend(glob.glob(f'*{ext}'))
    videos = sorted(set(videos))

    if videos:
        print(f"  Stock footage found ({len(videos)} videos):")
        for i, v in enumerate(videos, start=1):
            size_mb = os.path.getsize(v) / (1024 * 1024)
            print(f"  [{i}]  {v}  ({size_mb:.1f} MB)")
    else:
        print(f"  No video files found in '{default_video_dir}/' or current dir.")

    print()
    print("  [p]  Enter a custom file path")
    print("  [q]  Quit")
    print()

    while True:
        choice = input("  Select source: ").strip().lower()
        if choice == 'q':
            sys.exit(0)
        elif choice == 'p':
            path = input("  Enter full video path: ").strip().strip('"').strip("'")
            if os.path.isfile(path):
                return 'video', path
            else:
                print(f"  File not found: {path}")
        elif choice == '0' or choice == '':
            return 'webcam', 0
        else:
            try:
                idx = int(choice)
                if 1 <= idx <= len(videos):
                    return 'video', videos[idx - 1]
                else:
                    print(f"  Pick a number 0-{len(videos)}")
            except ValueError:
                print("  Invalid choice, try again")


# ─────────────────────────────────────────────────────────────────────
#  AUTONOMOUS VEHICLE (with Navigation)
# ─────────────────────────────────────────────────────────────────────
class AutonomousVehicle:
    def __init__(self, max_speed=0.8, enable_nav=False, track_name="default"):
        self.perception = LaneDetector(width=640, height=480)
        self.steering = PIDController(Kp=0.003, Ki=0.0001, Kd=0.001)
        self.speed_control = AdaptiveSpeedController(min_speed=0.2, max_speed=max_speed)
        self.safety = ObstacleDetector(emergency_distance=20, warning_distance=50)
        self.sign_detector = TrafficSignDetector()
        self.intersection_detector = IntersectionDetector()

        self.autonomous_enabled = True
        self.current_speed_limit = max_speed
        self.stop_sign_timer = 0
        self.stop_sign_cooldown = 0
        self.frame_count = 0
        self.total_error = 0
        self.stop_signs_detected = 0

        # ── SMOOTH DISPLAY VALUES ────────────────────────────────────
        self.smooth_left = SmoothValue(0.0, alpha=0.18)
        self.smooth_right = SmoothValue(0.0, alpha=0.18)
        self.smooth_base_speed = SmoothValue(0.0, alpha=0.15)
        self.smooth_pid = SmoothValue(0.0, alpha=0.12)

        # ── STATUS HOLD ──────────────────────────────────────────────
        self._display_status = "READY"
        self._status_hold_frames = 0
        self._STATUS_MIN_HOLD = 6

        # ── NAVIGATION ───────────────────────────────────────────────
        self.nav_enabled = enable_nav
        self.nav_overlay_visible = enable_nav
        self.navigator = None
        self.track = None

        if enable_nav:
            self._init_navigation(track_name)

    def _init_navigation(self, track_name):
        """Load track, compute shortest path, initialize navigator."""
        print("\n" + "=" * 60)
        print("  NAVIGATION MODE")
        print("=" * 60)

        if track_name == "complex":
            self.track = build_complex_track()
        else:
            self.track = build_default_track()

        self.track.print_map()

        start = self.track.get_start()
        dest = self.track.get_destination()

        if not start or not dest:
            print("Error: Track must have a START and DESTINATION node")
            self.nav_enabled = False
            return

        # Find shortest path
        route = find_shortest_path(self.track, start.name, dest.name)
        print_route(route, "SHORTEST PATH (Dijkstra)")

        # Show all alternatives
        all_routes = get_all_routes(self.track, start.name, dest.name)
        if len(all_routes) > 1:
            print(f"\n  All {len(all_routes)} routes:")
            for i, r in enumerate(all_routes):
                tag = " ★ CHOSEN" if i == 0 else ""
                print(f"    {i+1}. {r['distance']:.0f}cm — "
                      f"{' → '.join(r['path'])}{tag}")

        # Initialize navigator with shortest route
        self.navigator = Navigator(route)
        print("=" * 60 + "\n")

    def _update_display_status(self, new_status):
        if new_status == self._display_status:
            self._status_hold_frames = self._STATUS_MIN_HOLD
            return self._display_status
        self._status_hold_frames -= 1
        if self._status_hold_frames <= 0:
            self._display_status = new_status
            self._status_hold_frames = self._STATUS_MIN_HOLD
        return self._display_status

    def process_frame(self, frame):
        self.frame_count += 1
        steering_error, vision_frame = self.perception.process_frame(frame)

        # ── EMERGENCY BRAKE ──────────────────────────────────────────
        if steering_error is None:
            self.smooth_left.update(0.0)
            self.smooth_right.update(0.0)
            self.smooth_base_speed.update(0.0)
            self.smooth_pid.update(0.0)
            debug_frame = self._create_debug_frame(
                vision_frame, 0, 0.0, 0.0, 0.0, 0.0,
                "EMERGENCY STOP", "NONE"
            )
            return debug_frame, (0.0, 0.0, "EMERGENCY STOP")

        self.total_error += abs(steering_error)
        obstacle_modifier = 1.0

        # ── SIGN DETECTION ───────────────────────────────────────────
        self.sign_detector.detect_signs(frame)
        sign_action, _ = self.sign_detector.get_action()

        if self.stop_sign_cooldown > 0:
            self.stop_sign_cooldown -= 1

        if sign_action == "STOP" and self.stop_sign_cooldown == 0:
            if self.stop_sign_timer == 0:
                self.stop_sign_timer = 60
                self.stop_sign_cooldown = 120
                self.stop_signs_detected += 1
                print("STOP SIGN - stopping for 2 seconds")

        # ── INTERSECTION DETECTION ───────────────────────────────────
        # Count raw hough lines for the intersection detector
        num_raw_lines = 0
        lane_width = None

        # Get raw line data from perception's last frame
        gray = cv2.cvtColor(cv2.resize(frame, (640, 480)), cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        lines = cv2.HoughLinesP(edges, 2, 3.14159 / 180, 30,
                                minLineLength=40, maxLineGap=150)
        if lines is not None:
            num_raw_lines = len(lines)

        # Compute lane width if both lanes visible
        if (self.perception.ema_left_fit is not None and
                self.perception.ema_right_fit is not None):
            y_eval = int(480 * 0.75)
            left_x = self.perception._eval_fit(self.perception.ema_left_fit, y_eval)
            right_x = self.perception._eval_fit(self.perception.ema_right_fit, y_eval)
            if left_x is not None and right_x is not None:
                lane_width = right_x - left_x

        fork_detected = self.intersection_detector.update(
            num_lines=num_raw_lines,
            left_confidence=self.perception.left_confidence,
            right_confidence=self.perception.right_confidence,
            lane_width=lane_width,
            left_fit=self.perception.ema_left_fit,
            right_fit=self.perception.ema_right_fit
        )

        # ── NAVIGATION UPDATE ────────────────────────────────────────
        turn_override = None
        if self.nav_enabled and self.navigator:
            # Estimate distance traveled (rough: speed * time per frame)
            # At ~30fps, base_speed 0.8 ≈ moving ~0.5cm/frame on the Pi
            est_speed_cm = self.smooth_base_speed.value * 0.6
            nav_state = self.navigator.update(
                distance_delta=est_speed_cm,
                intersection_detected=fork_detected
            )
            turn_override = self.navigator.get_turn_override()

        # ── SPEED CALCULATION ────────────────────────────────────────
        if self.stop_sign_timer > 0:
            base_speed = 0.0
            self.stop_sign_timer -= 1
            status = f"STOPPED (Sign: {self.stop_sign_timer})"
        elif (self.nav_enabled and self.navigator and
              self.navigator.state == NavigatorState.ARRIVED):
            base_speed = 0.0
            status = "ARRIVED AT DEST"
        else:
            base_speed = self.speed_control.calculate_speed(steering_error, obstacle_modifier)
            base_speed = min(base_speed, self.current_speed_limit)
            speed_category = self.speed_control.get_speed_category(abs(steering_error))
            status = speed_category.replace("_", " ")

        # ── STEERING (with nav turn override) ────────────────────────
        if self.autonomous_enabled and base_speed > 0:
            if turn_override == "TURN_LEFT":
                # Hard left: left motors slow, right motors fast
                left_speed = 0.15
                right_speed = 0.7
                pid_output = -0.3
                status = "NAV: TURNING LEFT"
            elif turn_override == "TURN_RIGHT":
                # Hard right: left motors fast, right motors slow
                left_speed = 0.7
                right_speed = 0.15
                pid_output = 0.3
                status = "NAV: TURNING RIGHT"
            else:
                # Normal PID lane following
                pid_output = self.steering.compute(steering_error)
                left_speed = max(0.0, min(1.0, base_speed + pid_output))
                right_speed = max(0.0, min(1.0, base_speed - pid_output))
        else:
            pid_output = 0.0
            left_speed = 0.0
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

    def _create_debug_frame(self, vision_frame, error, pid_output,
                            base_speed, left_speed, right_speed, status, sign_action):
        frame = vision_frame.copy()
        frame = self.safety.draw_overlay(frame)
        frame = self.sign_detector.draw_overlay(frame)

        # Motor bars
        frame = self._draw_motor_bars(
            frame, self.smooth_left.value,
            self.smooth_right.value, self.smooth_pid.value
        )

        # Speed indicator
        speed_category = self.speed_control.get_speed_category(abs(error))
        target_speed = self.speed_control.target_speed
        frame = draw_speed_indicator(frame, self.smooth_base_speed.value,
                                     target_speed, speed_category)

        # Status
        display_status = self._update_display_status(status)
        status_color = (0, 0, 255) if "STOP" in display_status else \
                       (0, 200, 255) if "NAV:" in display_status else \
                       (0, 255, 100) if "ARRIVED" in display_status else \
                       (255, 255, 255)
        cv2.putText(frame, f"Status: {display_status}",
                    (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

        # Sign status
        if sign_action == "STOP":
            cv2.putText(frame, "STOP DETECTED!",
                        (10, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
        else:
            cv2.putText(frame, "Scanning...",
                        (10, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

        # Intersection detector indicator
        if self.intersection_detector.fork_confidence > 0.2:
            conf = self.intersection_detector.fork_confidence
            fork_color = (0, 255, 255) if conf > 0.45 else (100, 200, 200)
            cv2.putText(frame, f"Fork: {conf:.0%}",
                        (10, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.4, fork_color, 1)

        # Source label
        if hasattr(self, '_source_label'):
            cv2.putText(frame, self._source_label, (frame.shape[1] - 250, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1)

        # ── NAVIGATION HUD ───────────────────────────────────────────
        if self.nav_overlay_visible and self.navigator:
            frame = self.navigator.draw_overlay(frame)

        return frame

    def _draw_motor_bars(self, frame, left_speed, right_speed, pid_output):
        h, w = frame.shape[:2]
        bar_width = 40
        bar_height = 200
        bar_x_left = w - 120
        bar_x_right = w - 60
        bar_y = h - bar_height - 50

        cv2.rectangle(frame, (bar_x_left, bar_y),
                      (bar_x_left + bar_width, bar_y + bar_height), (50, 50, 50), -1)
        cv2.rectangle(frame, (bar_x_right, bar_y),
                      (bar_x_right + bar_width, bar_y + bar_height), (50, 50, 50), -1)

        left_fill = int(bar_height * max(0.0, min(1.0, left_speed)))
        right_fill = int(bar_height * max(0.0, min(1.0, right_speed)))
        bar_color = (0, 255, 0) if abs(pid_output) < 0.1 else (0, 165, 255)

        if left_fill > 0:
            cv2.rectangle(frame, (bar_x_left, bar_y + bar_height - left_fill),
                          (bar_x_left + bar_width, bar_y + bar_height), bar_color, -1)
        if right_fill > 0:
            cv2.rectangle(frame, (bar_x_right, bar_y + bar_height - right_fill),
                          (bar_x_right + bar_width, bar_y + bar_height), bar_color, -1)

        cv2.putText(frame, "L", (bar_x_left + 12, bar_y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, "R", (bar_x_right + 12, bar_y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"{left_speed:.2f}", (bar_x_left, bar_y + bar_height + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"{right_speed:.2f}", (bar_x_right, bar_y + bar_height + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return frame

    def _reset_all(self):
        self.steering.reset()
        self.speed_control.reset()
        self.safety.reset()
        self.perception.reset_smoothing()
        self.sign_detector.reset()
        self.intersection_detector.reset()
        self.current_speed_limit = 0.8
        self.stop_sign_timer = 0
        self.smooth_left.set_immediate(0.0)
        self.smooth_right.set_immediate(0.0)
        self.smooth_base_speed.set_immediate(0.0)
        self.smooth_pid.set_immediate(0.0)
        self._display_status = "READY"
        if self.navigator:
            self.navigator.reset()
        print("All systems reset")

    def run(self, source_type='webcam', source_value=0, rotate_frame=False):
        is_video = (source_type == 'video')

        if is_video:
            if not os.path.isfile(source_value):
                print(f"Error: File not found: {source_value}")
                return
            cap = cv2.VideoCapture(source_value)
            video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self._source_label = f"FILE: {os.path.basename(source_value)}"
            print(f"Playing: {source_value}")
            print(f"Video FPS: {video_fps:.1f} | Frames: {total_frames}")
        else:
            cap = cv2.VideoCapture(source_value)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            video_fps = 30
            self._source_label = "WEBCAM"

        if not cap.isOpened():
            print(f"Error: Could not open {'video' if is_video else 'webcam'}.")
            return

        time.sleep(0.5)
        nav_str = " + NAVIGATION" if self.nav_enabled else ""
        print(f"Source ready! 640x480 | Rotation: {'ON' if rotate_frame else 'OFF'}{nav_str}")
        print("Controls: [Q] Quit  [SPACE] Auto  [R] Reset  [S] Snap  [D] Signs  [N] Nav overlay")
        if is_video:
            print("          [P] Pause  [LEFT/RIGHT] Skip 5s  [L] Loop")

        start_time = time.time()
        paused = False
        loop_video = True

        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    if is_video and loop_video:
                        print("End of video — looping...")
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        self._reset_all()
                        continue
                    elif is_video:
                        print("End of video.")
                        break
                    else:
                        print("Failed to grab frame. Exiting...")
                        break

                if rotate_frame:
                    frame = cv2.rotate(frame, cv2.ROTATE_180)

                debug_frame, (left, right, status) = self.process_frame(frame)

                # Video progress bar
                if is_video:
                    current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                    progress = current_frame / max(total_frames, 1)
                    h = debug_frame.shape[0]
                    w = debug_frame.shape[1]
                    bar_w = int(w * progress)
                    cv2.rectangle(debug_frame, (0, h - 8), (w, h), (30, 30, 30), -1)
                    cv2.rectangle(debug_frame, (0, h - 8), (bar_w, h), (0, 200, 255), -1)

            window_title = "VectorVance - " + \
                           ("AUTONOMOUS" if self.autonomous_enabled else "MANUAL") + \
                           (" [NAV]" if self.nav_enabled else "")
            cv2.imshow(window_title, debug_frame)

            if self.frame_count % 30 == 0 and not paused:
                elapsed = time.time() - start_time
                fps = self.frame_count / max(elapsed, 0.1)
                avg_error = self.total_error / max(self.frame_count, 1)
                nav_info = ""
                if self.navigator:
                    p = self.navigator.get_progress()
                    nav_info = f" | Nav:{p['state']}"
                print(f"Frame {self.frame_count:04d} | "
                      f"L:{left:.2f} R:{right:.2f} | "
                      f"{status:25s} | "
                      f"FPS:{fps:.1f} | "
                      f"AvgErr:{avg_error:.1f}px{nav_info}")

            wait_ms = int(1000 / video_fps) if is_video and not paused else 1
            key = cv2.waitKey(wait_ms) & 0xFF

            if key == ord('q'):
                break
            elif key == ord(' '):
                self.autonomous_enabled = not self.autonomous_enabled
                print(f"Autonomous: {'ENABLED' if self.autonomous_enabled else 'DISABLED'}")
            elif key == ord('r'):
                self._reset_all()
                if is_video:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            elif key == ord('s'):
                filename = f"snapshot_{self.frame_count:04d}.jpg"
                cv2.imwrite(filename, debug_frame)
                print(f"Snapshot saved: {filename}")
            elif key == ord('d'):
                print(f"Sign detector: {len(self.sign_detector.detected_signs)} raw, "
                      f"{len(self.sign_detector.confirmed_signs)} confirmed")
                for s in self.sign_detector.confirmed_signs:
                    print(f"  {s[0].value} at {s[1]} conf={s[2]:.2f}")
            elif key == ord('n'):
                self.nav_overlay_visible = not self.nav_overlay_visible
                print(f"Nav overlay: {'ON' if self.nav_overlay_visible else 'OFF'}")
            elif key == ord('p') and is_video:
                paused = not paused
                print(f"{'PAUSED' if paused else 'RESUMED'}")
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
        self._print_statistics(start_time)

    def _print_statistics(self, start_time):
        duration = time.time() - start_time
        avg_error = self.total_error / max(self.frame_count, 1)
        fps = self.frame_count / max(duration, 0.1)
        print("=" * 70)
        print(f"Duration:      {duration:.1f}s")
        print(f"Frames:        {self.frame_count}")
        print(f"Average FPS:   {fps:.1f}")
        print(f"Average error: {avg_error:.1f}px")
        print(f"Stop signs:    {self.stop_signs_detected}")
        if self.navigator:
            p = self.navigator.get_progress()
            print(f"Navigation:    {p['state']} — "
                  f"{p['total_traveled']:.0f}/{p['total_route']:.0f}cm")
        print("=" * 70)


# ─────────────────────────────────────────────────────────────────────
#  CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='VectorVance — Full Navigation')
    parser.add_argument('--webcam', action='store_true',
                        help='Use webcam directly')
    parser.add_argument('--video', type=str, default=None,
                        help='Path to a video file')
    parser.add_argument('--dir', type=str, default='test_videos',
                        help='Folder to scan for videos')
    parser.add_argument('--speed', type=float, default=0.8,
                        help='Max speed 0.0-1.0')
    parser.add_argument('--rotate', action='store_true',
                        help='Rotate frame 180 degrees')
    parser.add_argument('--no-rotate', action='store_true',
                        help='Force no rotation')
    parser.add_argument('--nav', action='store_true',
                        help='Enable pathfinding navigation')
    parser.add_argument('--track', type=str, default='default',
                        choices=['default', 'complex'],
                        help='Track layout to use (default or complex)')
    args = parser.parse_args()

    vehicle = AutonomousVehicle(
        max_speed=args.speed,
        enable_nav=args.nav,
        track_name=args.track
    )

    if args.video:
        rotate = args.rotate
        vehicle.run(source_type='video', source_value=args.video, rotate_frame=rotate)
    elif args.webcam:
        rotate = not args.no_rotate
        vehicle.run(source_type='webcam', source_value=0, rotate_frame=rotate)
    else:
        source_type, source_value = pick_source_interactive(args.dir)
        if args.rotate:
            rotate = True
        elif args.no_rotate:
            rotate = False
        else:
            rotate = (source_type == 'webcam')
        vehicle.run(source_type=source_type, source_value=source_value, rotate_frame=rotate)


if __name__ == "__main__":
    main()