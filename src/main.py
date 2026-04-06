"""
main.py - VectorVance Autonomous Car (PRODUCTION)
──────────────────────────────────────────────────
Full feature set for Raspberry Pi deployment.

  Hardware : Picamera2 + dual L298N motors + HC-SR04 ultrasonic
  Detection: YOLO v8n (optional) or classic color+shape detector
  Control  : PID lane-follow + adaptive speed + obstacle avoidance
  Navigation: Dijkstra pathfinding + intersection-aware turn logic
  Dashboard: Live web UI at http://<pi-ip>:5000/

USAGE:
  python main.py                         # basic lane follow
  python main.py --yolo                  # + YOLO detection
  python main.py --nav                   # + pathfinding navigation
  python main.py --yolo --nav            # full feature set
  python main.py --no-web                # disable web dashboard
  python main.py --no-display            # headless (no cv2 window)
  python main.py --help                  # all options

CONTROLS (keyboard, when display is on):
  Q      Quit
  SPACE  Toggle autonomous mode
  R      Reset all systems
  S      Save snapshot
  D      Print sign detector debug info
  N      Toggle navigation overlay
"""

import cv2
import time
import lgpio
import argparse

from gpiozero import Motor
from picamera2 import Picamera2

from perception import LaneDetector, SmoothValue
from controller import PIDController
from speed_controller import AdaptiveSpeedController, draw_speed_indicator
from safety import ObstacleDetector
from sign_detector import TrafficSignDetector
from yolo_detector import YoloDetector
from intersection_detector import IntersectionDetector
from track_map import build_default_track, build_complex_track
from pathfinder import find_shortest_path, get_all_routes, print_route
from navigator import Navigator, NavigatorState
import pi_server

# ── GPIO pin assignments ──────────────────────────────────────────────────────
# Motor Driver 1 — Front Left (IN1/IN2) and Front Right (IN3/IN4)
FL_FWD, FL_BWD = 25, 27
FR_FWD, FR_BWD = 5, 15
# Motor Driver 2 — Rear Left (IN5/IN6) and Rear Right (IN7/IN8)
RL_FWD, RL_BWD = 26, 20
RR_FWD, RR_BWD = 16, 6
# HC-SR04 ultrasonic
TRIG_PIN, ECHO_PIN = 4, 17
STOP_DISTANCE = 20   # cm — emergency stop
SLOW_DISTANCE = 50   # cm — start slowing down


# ─────────────────────────────────────────────────────────────────────────────
class AutonomousVehicle:

    def __init__(self,
                 max_speed    = 0.8,
                 enable_nav   = False,
                 track_name   = "default",
                 enable_yolo  = False,
                 yolo_model   = "yolov8n.pt",
                 yolo_skip    = 5,
                 enable_web   = True,
                 web_port     = 5000,
                 show_display = True):

        # ── Perception & control ──────────────────────────────────────
        self.perception   = LaneDetector(width=640, height=480)
        self.steering     = PIDController(Kp=0.003, Ki=0.0001, Kd=0.001)
        self.speed_control = AdaptiveSpeedController(min_speed=0.2, max_speed=max_speed)
        self.safety       = ObstacleDetector(
            emergency_distance=STOP_DISTANCE,
            warning_distance=SLOW_DISTANCE
        )
        self.intersection_detector = IntersectionDetector()

        # ── Detector: YOLO or classic fallback ───────────────────────
        self.yolo_enabled = enable_yolo
        if enable_yolo:
            self.detector = YoloDetector(
                model_name=yolo_model,
                skip_frames=yolo_skip
            )
        else:
            self.detector = TrafficSignDetector()
            print("[Detector] Classic color+shape detector active (--yolo to enable YOLO)")

        # ── Navigation ───────────────────────────────────────────────
        self.nav_enabled       = enable_nav
        self.nav_overlay_visible = enable_nav
        self.navigator         = None
        self.track             = None
        if enable_nav:
            self._init_navigation(track_name)

        # ── Motors ───────────────────────────────────────────────────
        self.front_left  = Motor(forward=FL_FWD, backward=FL_BWD)
        self.rear_left   = Motor(forward=RL_FWD, backward=RL_BWD)
        self.front_right = Motor(forward=FR_FWD, backward=FR_BWD)
        self.rear_right  = Motor(forward=RR_FWD, backward=RR_BWD)
        print("[Motors] All 4 motors OK")

        # ── Ultrasonic sensor ─────────────────────────────────────────
        self._gpio = lgpio.gpiochip_open(0)
        lgpio.gpio_claim_output(self._gpio, TRIG_PIN)
        lgpio.gpio_claim_input(self._gpio, ECHO_PIN)
        lgpio.gpio_write(self._gpio, TRIG_PIN, 0)
        time.sleep(0.1)
        print("[Ultrasonic] HC-SR04 OK")

        # ── Web dashboard ─────────────────────────────────────────────
        self.web_enabled  = enable_web
        self.web_port     = web_port
        self.show_display = show_display

        # ── Smooth display values ─────────────────────────────────────
        self.smooth_left       = SmoothValue(0.0, alpha=0.18)
        self.smooth_right      = SmoothValue(0.0, alpha=0.18)
        self.smooth_base_speed = SmoothValue(0.0, alpha=0.15)
        self.smooth_pid        = SmoothValue(0.0, alpha=0.12)

        # ── Status hold (prevents HUD flickering) ────────────────────
        self._display_status   = "READY"
        self._status_hold_frames = 0
        self._STATUS_MIN_HOLD  = 6

        # ── Runtime state ─────────────────────────────────────────────
        self.autonomous_enabled   = True
        self.current_speed_limit  = max_speed
        self.stop_sign_timer      = 0
        self.stop_sign_cooldown   = 0
        self.frame_count          = 0
        self.total_error          = 0
        self.stop_signs_detected  = 0
        self._last_distance       = 999.0
        self._last_steering_error = 0.0
        self._start_time          = 0.0   # set in run()

    # ── Navigation setup ─────────────────────────────────────────────────────

    def _init_navigation(self, track_name: str):
        print("\n" + "=" * 56)
        print("  NAVIGATION MODE")
        print("=" * 56)
        self.track = (build_complex_track() if track_name == "complex"
                      else build_default_track())
        self.track.print_map()

        start = self.track.get_start()
        dest  = self.track.get_destination()
        if not start or not dest:
            print("[Nav] ERROR: Track must have START and DESTINATION nodes")
            self.nav_enabled = False
            return

        route = find_shortest_path(self.track, start.name, dest.name)
        print_route(route, "SHORTEST PATH (Dijkstra)")

        all_routes = get_all_routes(self.track, start.name, dest.name)
        if len(all_routes) > 1:
            print(f"\n  All {len(all_routes)} routes:")
            for i, r in enumerate(all_routes):
                tag = " ★ CHOSEN" if i == 0 else ""
                print(f"    {i+1}. {r['distance']:.0f}cm — "
                      f"{' → '.join(r['path'])}{tag}")

        self.navigator = Navigator(route)
        print("=" * 56 + "\n")

    # ── Status hold ───────────────────────────────────────────────────────────

    def _update_display_status(self, new_status: str) -> str:
        if new_status == self._display_status:
            self._status_hold_frames = self._STATUS_MIN_HOLD
            return self._display_status
        self._status_hold_frames -= 1
        if self._status_hold_frames <= 0:
            self._display_status = new_status
            self._status_hold_frames = self._STATUS_MIN_HOLD
        return self._display_status

    # ── Hardware helpers ──────────────────────────────────────────────────────

    def _get_distance(self) -> float:
        """Measure distance in cm via HC-SR04. Returns 999 on timeout."""
        lgpio.gpio_write(self._gpio, TRIG_PIN, 1)
        time.sleep(0.00001)
        lgpio.gpio_write(self._gpio, TRIG_PIN, 0)

        timeout = time.time() + 0.04
        start   = time.time()
        while lgpio.gpio_read(self._gpio, ECHO_PIN) == 0:
            start = time.time()
            if time.time() > timeout:
                return 999.0

        stop    = time.time()
        timeout = time.time() + 0.04
        while lgpio.gpio_read(self._gpio, ECHO_PIN) == 1:
            stop = time.time()
            if time.time() > timeout:
                return 999.0

        return round((stop - start) * 34300 / 2, 1)

    def _drive(self, left_speed: float, right_speed: float):
        """Send PWM speeds (0.0–1.0) to all four motors."""
        left_speed  = max(0.0, min(1.0, left_speed))
        right_speed = max(0.0, min(1.0, right_speed))

        if left_speed < 0.05:
            self.front_left.stop();  self.rear_left.stop()
        else:
            self.front_left.backward(left_speed)
            self.rear_left.backward(left_speed)

        if right_speed < 0.05:
            self.front_right.stop(); self.rear_right.stop()
        else:
            self.front_right.backward(right_speed)
            self.rear_right.backward(right_speed)

    def _stop_motors(self):
        for m in (self.front_left, self.rear_left,
                  self.front_right, self.rear_right):
            m.stop()

    def _cleanup_hardware(self):
        self._stop_motors()
        for m in (self.front_left, self.rear_left,
                  self.front_right, self.rear_right):
            m.close()
        lgpio.gpiochip_close(self._gpio)
        print("[Hardware] GPIO released")

    # ── Web command handler ───────────────────────────────────────────────────

    def _handle_web_commands(self):
        cmd = pi_server.get_pending_command()
        if cmd is None:
            return
        action = cmd.get("action")
        if action == "toggle_auto":
            self.autonomous_enabled = not self.autonomous_enabled
            print(f"[Web] Autonomous: {'ON' if self.autonomous_enabled else 'OFF'}")
        elif action == "emergency_stop":
            self.autonomous_enabled = False
            self._stop_motors()
            print("[Web] Emergency stop!")
        elif action == "reset":
            self._reset_all()
        elif action == "set_speed":
            val = float(cmd.get("value", 0.8))
            self.current_speed_limit = max(0.1, min(1.0, val))
            print(f"[Web] Speed limit set to {self.current_speed_limit:.2f}")
        pi_server.clear_command()

    # ── Telemetry builder ─────────────────────────────────────────────────────

    def _build_telemetry(self, left: float, right: float, status: str) -> dict:
        elapsed = max(time.time() - self._start_time, 0.1)
        fps     = round(self.frame_count / elapsed, 1)
        yolo_dets = (
            [{"label": d[1], "conf": round(d[3], 2)}
             for d in self.detector.all_detections]
            if self.yolo_enabled else []
        )
        yolo_danger = self.detector.get_danger_level() if self.yolo_enabled else "CLEAR"
        nav_progress = None
        nav_state    = None
        nav_turn     = None
        if self.navigator:
            nav_progress = self.navigator.get_progress()
            nav_state    = self.navigator.state.value
            nav_turn     = self.navigator.get_turn_override()

        return {
            "mode":               "AUTONOMOUS" if self.autonomous_enabled else "MANUAL",
            "status":             status,
            "speed_left":         round(left, 3),
            "speed_right":        round(right, 3),
            "base_speed":         round(self.smooth_base_speed.value, 3),
            "steering_error":     round(self._last_steering_error, 1),
            "distance_cm":        self._last_distance,
            "fps":                fps,
            "frame_count":        self.frame_count,
            "stop_signs_detected":self.stop_signs_detected,
            "yolo_enabled":       self.yolo_enabled,
            "yolo_detections":    yolo_dets,
            "yolo_danger":        yolo_danger,
            "obstacle_modifier":  round(
                self.detector.get_speed_modifier() if self.yolo_enabled else 1.0, 2
            ),
            "nav_enabled":        self.nav_enabled,
            "nav_state":          nav_state,
            "nav_next_turn":      nav_turn,
            "nav_progress":       nav_progress,
            "fork_confidence":    round(self.intersection_detector.fork_confidence, 2),
        }

    # ── Main perception + decision loop ──────────────────────────────────────

    def process_frame(self, frame):
        self.frame_count += 1
        steering_error, vision_frame = self.perception.process_frame(frame)

        # ── Emergency brake: no lane detected ────────────────────────
        if steering_error is None:
            self.smooth_left.update(0.0)
            self.smooth_right.update(0.0)
            self.smooth_base_speed.update(0.0)
            self.smooth_pid.update(0.0)
            debug_frame = self._create_debug_frame(
                vision_frame, 0, 0.0, 0.0, 0.0, 0.0, "EMERGENCY STOP", "NONE"
            )
            return debug_frame, (0.0, 0.0, "EMERGENCY STOP")

        self.total_error += abs(steering_error)
        self._last_steering_error = steering_error

        # ── Ultrasonic (every 3 frames) ───────────────────────────────
        if self.frame_count % 3 == 0:
            self._last_distance = self._get_distance()

        self.safety.sensors['front']['distance'] = self._last_distance
        self.safety._check_obstacles()

        # ── Detection ─────────────────────────────────────────────────
        if self.yolo_enabled:
            self.detector.detect(frame)
            obstacle_modifier = self.detector.get_speed_modifier()
        else:
            self.detector.detect_signs(frame)
            # Merge ultrasonic obstacle modifier with sign detector
            obstacle_modifier = self.safety.get_speed_modifier()

        sign_action, _ = self.detector.get_action()

        if self.stop_sign_cooldown > 0:
            self.stop_sign_cooldown -= 1

        if sign_action == "STOP" and self.stop_sign_cooldown == 0:
            if self.stop_sign_timer == 0:
                self.stop_sign_timer = 60
                self.stop_sign_cooldown = 120
                self.stop_signs_detected += 1
                print("STOP SIGN — stopping for 2 seconds")

        # ── Intersection detection ────────────────────────────────────
        num_raw_lines = 0
        lane_width    = None

        gray  = cv2.cvtColor(cv2.resize(frame, (640, 480)), cv2.COLOR_BGR2GRAY)
        blur  = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
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
            right_fit        = self.perception.ema_right_fit
        )

        # ── Navigation update ─────────────────────────────────────────
        turn_override = None
        if self.nav_enabled and self.navigator:
            est_speed_cm = self.smooth_base_speed.value * 0.6
            self.navigator.update(
                distance_delta       = est_speed_cm,
                intersection_detected = fork_detected
            )
            turn_override = self.navigator.get_turn_override()

        # ── Speed decision ────────────────────────────────────────────
        if self.stop_sign_timer > 0:
            base_speed = 0.0
            self.stop_sign_timer -= 1
            status = f"STOPPED (Sign: {self.stop_sign_timer})"
        elif (self.nav_enabled and self.navigator and
              self.navigator.state == NavigatorState.ARRIVED):
            base_speed = 0.0
            status = "ARRIVED AT DEST"
        else:
            base_speed = self.speed_control.calculate_speed(
                steering_error, obstacle_modifier
            )
            base_speed = min(base_speed, self.current_speed_limit)
            speed_cat  = self.speed_control.get_speed_category(abs(steering_error))
            status     = speed_cat.replace("_", " ")

        # ── Steering ──────────────────────────────────────────────────
        if self.autonomous_enabled and base_speed > 0:
            if turn_override == "TURN_LEFT":
                left_speed  = 0.15
                right_speed = 0.70
                pid_output  = -0.3
                status      = "NAV: TURNING LEFT"
            elif turn_override == "TURN_RIGHT":
                left_speed  = 0.70
                right_speed = 0.15
                pid_output  = 0.3
                status      = "NAV: TURNING RIGHT"
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
                            base_speed, left_speed, right_speed,
                            status, sign_action):
        frame = vision_frame.copy()
        frame = self.safety.draw_overlay(frame)
        frame = self.detector.draw_overlay(frame)
        frame = self._draw_motor_bars(
            frame,
            self.smooth_left.value,
            self.smooth_right.value,
            self.smooth_pid.value
        )
        speed_cat    = self.speed_control.get_speed_category(abs(error))
        target_speed = self.speed_control.target_speed
        frame = draw_speed_indicator(frame, self.smooth_base_speed.value,
                                     target_speed, speed_cat)

        display_status = self._update_display_status(status)
        status_color = (
            (0,   0,   255) if "STOP"    in display_status else
            (0,   200, 255) if "NAV:"    in display_status else
            (0,   255, 100) if "ARRIVED" in display_status else
            (255, 255, 255)
        )
        cv2.putText(frame, f"Status: {display_status}",
                    (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

        if sign_action == "STOP":
            cv2.putText(frame, "STOP DETECTED!",
                        (10, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
        elif self.yolo_enabled:
            danger = self.detector.get_danger_level()
            danger_colors = {
                "CLEAR":   (100, 200, 100),
                "CAUTION": (0,   200, 255),
                "DANGER":  (0,   100, 255),
                "STOP":    (0,   0,   255),
            }
            cv2.putText(frame, f"YOLO: {danger}",
                        (10, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        danger_colors.get(danger, (150, 150, 150)), 1)
        else:
            cv2.putText(frame, "Scanning...",
                        (10, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

        if self.intersection_detector.fork_confidence > 0.2:
            conf       = self.intersection_detector.fork_confidence
            fork_color = (0, 255, 255) if conf > 0.45 else (100, 200, 200)
            cv2.putText(frame, f"Fork: {conf:.0%}",
                        (10, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.4, fork_color, 1)

        if self.nav_overlay_visible and self.navigator:
            frame = self.navigator.draw_overlay(frame)

        return frame

    def _draw_motor_bars(self, frame, left_speed, right_speed, pid_output):
        h, w       = frame.shape[:2]
        bar_width  = 40
        bar_height = 200
        bar_x_left = w - 120
        bar_x_right= w - 60
        bar_y      = h - bar_height - 50

        cv2.rectangle(frame, (bar_x_left,  bar_y),
                      (bar_x_left  + bar_width, bar_y + bar_height), (50, 50, 50), -1)
        cv2.rectangle(frame, (bar_x_right, bar_y),
                      (bar_x_right + bar_width, bar_y + bar_height), (50, 50, 50), -1)

        left_fill  = int(bar_height * max(0.0, min(1.0, left_speed)))
        right_fill = int(bar_height * max(0.0, min(1.0, right_speed)))
        bar_color  = (0, 255, 0) if abs(pid_output) < 0.1 else (0, 165, 255)

        if left_fill > 0:
            cv2.rectangle(frame,
                          (bar_x_left,  bar_y + bar_height - left_fill),
                          (bar_x_left  + bar_width, bar_y + bar_height), bar_color, -1)
        if right_fill > 0:
            cv2.rectangle(frame,
                          (bar_x_right, bar_y + bar_height - right_fill),
                          (bar_x_right + bar_width, bar_y + bar_height), bar_color, -1)

        cv2.putText(frame, "L", (bar_x_left  + 12, bar_y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, "R", (bar_x_right + 12, bar_y - 5),
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
        self.intersection_detector.reset()
        self.current_speed_limit = 0.8
        self.stop_sign_timer     = 0
        self.smooth_left.set_immediate(0.0)
        self.smooth_right.set_immediate(0.0)
        self.smooth_base_speed.set_immediate(0.0)
        self.smooth_pid.set_immediate(0.0)
        self._display_status = "READY"
        if self.navigator:
            self.navigator.reset()
        print("[System] All systems reset")

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self):
        # Start web dashboard
        if self.web_enabled:
            ok = pi_server.start_server(self.web_port)
            if not ok:
                print("[WebServer] Dashboard unavailable — continuing without it")
                self.web_enabled = False

        # Start camera
        picam2 = Picamera2()
        config = picam2.create_preview_configuration(
            main={"format": "RGB888", "size": (640, 480)}
        )
        picam2.configure(config)
        picam2.start()
        time.sleep(1)

        nav_str  = " + NAVIGATION" if self.nav_enabled  else ""
        yolo_str = " + YOLO"       if self.yolo_enabled else ""
        print(f"[Camera] 640×480 ready{nav_str}{yolo_str}")
        if self.show_display:
            print("Controls: [Q] Quit  [SPACE] Toggle auto  [R] Reset  "
                  "[S] Snapshot  [D] Signs  [N] Nav overlay")

        self._start_time = time.time()

        while True:
            frame = picam2.capture_array()
            debug_frame, (left, right, status) = self.process_frame(frame)

            # Handle commands from web dashboard
            if self.web_enabled:
                self._handle_web_commands()

            # Drive motors
            if self.autonomous_enabled:
                self._drive(left, right)
            else:
                self._stop_motors()

            # Push to web dashboard
            if self.web_enabled:
                pi_server.push_frame(debug_frame)
                pi_server.push_telemetry(
                    self._build_telemetry(left, right, status)
                )

            # Optional local display
            if self.show_display:
                title = ("VectorVance — " +
                         ("AUTONOMOUS" if self.autonomous_enabled else "MANUAL") +
                         (" [NAV]"  if self.nav_enabled  else "") +
                         (" [YOLO]" if self.yolo_enabled else ""))
                cv2.imshow(title, debug_frame)

            # Console stats every 30 frames
            if self.frame_count % 30 == 0:
                elapsed = time.time() - self._start_time
                fps     = self.frame_count / max(elapsed, 0.1)
                avg_err = self.total_error / max(self.frame_count, 1)
                nav_info = ""
                if self.navigator:
                    p = self.navigator.get_progress()
                    nav_info = f" | Nav:{p['state']}"
                print(f"Frame {self.frame_count:04d} | "
                      f"L:{left:.2f} R:{right:.2f} | "
                      f"{status:25s} | "
                      f"Dist:{self._last_distance:.0f}cm | "
                      f"FPS:{fps:.1f} | "
                      f"AvgErr:{avg_err:.1f}px{nav_info}")

            # Keyboard input
            if self.show_display:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord(' '):
                    self.autonomous_enabled = not self.autonomous_enabled
                    print(f"Autonomous: {'ENABLED' if self.autonomous_enabled else 'DISABLED'}")
                elif key == ord('r'):
                    self._reset_all()
                elif key == ord('s'):
                    fname = f"/home/pi/test/snapshot_{self.frame_count:04d}.jpg"
                    cv2.imwrite(fname, debug_frame)
                    print(f"Snapshot saved: {fname}")
                elif key == ord('d'):
                    det = self.detector
                    if hasattr(det, 'confirmed_signs'):
                        print(f"Signs: {len(det.detected_signs)} raw, "
                              f"{len(det.confirmed_signs)} confirmed")
                        for s in det.confirmed_signs:
                            print(f"  {s[0].value} conf={s[2]:.2f}")
                    else:
                        print(f"YOLO: {len(det.all_detections)} detections")
                        for d in det.all_detections:
                            print(f"  {d[1]} conf={d[3]:.2f}")
                elif key == ord('n'):
                    self.nav_overlay_visible = not self.nav_overlay_visible
                    print(f"Nav overlay: {'ON' if self.nav_overlay_visible else 'OFF'}")
            else:
                # Headless: check for Ctrl-C only (handled by Python's signal)
                pass

        picam2.stop()
        if self.show_display:
            cv2.destroyAllWindows()
        self._stop_motors()
        self._cleanup_hardware()
        self._print_statistics()

    # ── End-of-run stats ──────────────────────────────────────────────────────

    def _print_statistics(self):
        duration = time.time() - self._start_time
        avg_err  = self.total_error / max(self.frame_count, 1)
        fps      = self.frame_count / max(duration, 0.1)
        print("=" * 70)
        print(f"Duration      : {duration:.1f}s")
        print(f"Frames        : {self.frame_count}")
        print(f"Average FPS   : {fps:.1f}")
        print(f"Average error : {avg_err:.1f}px")
        print(f"Stop signs    : {self.stop_signs_detected}")
        if self.navigator:
            p = self.navigator.get_progress()
            print(f"Navigation    : {p['state']} — "
                  f"{p['total_traveled']:.0f}/{p['total_route']:.0f}cm")
        print("=" * 70)


# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='VectorVance Autonomous Car')
    parser.add_argument('--speed',       type=float, default=0.8,
                        help='Max motor speed 0.0–1.0 (default 0.8)')
    parser.add_argument('--yolo',        action='store_true',
                        help='Enable YOLOv8 detection')
    parser.add_argument('--yolo-model',  type=str,   default='yolov8n.pt',
                        help='YOLO model (default: yolov8n.pt)')
    parser.add_argument('--yolo-skip',   type=int,   default=5,
                        help='Run YOLO every N frames (default: 5, lower = more accurate but slower)')
    parser.add_argument('--nav',         action='store_true',
                        help='Enable pathfinding navigation')
    parser.add_argument('--track',       type=str,   default='default',
                        choices=['default', 'complex'],
                        help='Track layout (default or complex)')
    parser.add_argument('--port',        type=int,   default=5000,
                        help='Web dashboard port (default: 5000)')
    parser.add_argument('--no-web',      action='store_true',
                        help='Disable web dashboard')
    parser.add_argument('--no-display',  action='store_true',
                        help='Headless mode — no cv2 window (useful when no monitor connected)')
    args = parser.parse_args()

    vehicle = AutonomousVehicle(
        max_speed    = args.speed,
        enable_nav   = args.nav,
        track_name   = args.track,
        enable_yolo  = args.yolo,
        yolo_model   = args.yolo_model,
        yolo_skip    = args.yolo_skip,
        enable_web   = not args.no_web,
        web_port     = args.port,
        show_display = not args.no_display,
    )
    vehicle.run()


if __name__ == "__main__":
    main()
