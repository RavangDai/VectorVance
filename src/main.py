"""
main.py - VectorVance Autonomous Car (PRODUCTION)
──────────────────────────────────────────────────
Full feature set for Raspberry Pi deployment.

  Hardware : Picamera2 + dual L298N motors + HC-SR04 ultrasonic
  Detection: YOLO v8n (optional) or classic color+shape detector
  Control  : PID lane-follow + adaptive speed + obstacle avoidance
  Fork nav : Color tape detection — steers toward target color at forks
  Dashboard: Live web UI at http://<pi-ip>:5000/

USAGE:
  python main.py                                 # basic lane follow
  python main.py --yolo                          # + YOLO detection
  python main.py --target-color GREEN            # follow GREEN tape at forks
  python main.py --yolo --target-color BLUE      # YOLO + follow BLUE tape
  python main.py --no-web                        # disable web dashboard
  python main.py --no-display                    # headless (no cv2 window)

KEYBOARD (when display is on):
  Q        Quit
  SPACE    Toggle autonomous mode
  R        Reset all systems
  S        Save snapshot
  D        Print detector debug info
  G/B/E    Set target tape colour to Green / Blue / rEd
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
from color_sign_detector import ColorSignDetector
import pi_server

# ── GPIO pin assignments ──────────────────────────────────────────────────────
FL_FWD, FL_BWD = 25, 27    # Front Left
FR_FWD, FR_BWD = 5,  15    # Front Right
RL_FWD, RL_BWD = 26, 20    # Rear Left
RR_FWD, RR_BWD = 16, 6     # Rear Right
TRIG_PIN, ECHO_PIN = 4, 17 # HC-SR04
STOP_DISTANCE = 20          # cm — hard stop
SLOW_DISTANCE = 50          # cm — start slowing

# How many frames to steer by tape colour after a fork fires (~1.5 s at 30 fps)
_COLOR_FOLLOW_FRAMES = 45


# ─────────────────────────────────────────────────────────────────────────────
class AutonomousVehicle:

    def __init__(self,
                 max_speed    = 0.8,
                 enable_yolo  = False,
                 yolo_model   = "yolov8n.pt",
                 yolo_skip    = 5,
                 target_color = "GREEN",
                 enable_web   = True,
                 web_port     = 5000,
                 show_display = True):

        # ── Perception & control ──────────────────────────────────────
        self.perception    = LaneDetector(width=640, height=480)
        self.steering      = PIDController(Kp=0.003, Ki=0.0001, Kd=0.001)
        self.speed_control = AdaptiveSpeedController(min_speed=0.2, max_speed=max_speed)
        self.safety        = ObstacleDetector(
            emergency_distance=STOP_DISTANCE,
            warning_distance=SLOW_DISTANCE
        )
        self.intersection_detector = IntersectionDetector()

        # ── Sign / obstacle detector ──────────────────────────────────
        self.yolo_enabled = enable_yolo
        if enable_yolo:
            self.detector = YoloDetector(model_name=yolo_model, skip_frames=yolo_skip)
        else:
            self.detector = TrafficSignDetector()
            print("[Detector] Classic stop-sign detector active (use --yolo for YOLO)")

        # ── Colour tape navigator ─────────────────────────────────────
        self.color_detector       = ColorSignDetector(
            target_color=target_color, frame_width=640, frame_height=480
        )
        self._color_follow_frames = 0   # countdown after a fork fires

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
        self._display_status     = "READY"
        self._status_hold_frames = 0
        self._STATUS_MIN_HOLD    = 6

        # ── Runtime counters ──────────────────────────────────────────
        self.autonomous_enabled   = True
        self.current_speed_limit  = max_speed
        self.stop_sign_timer      = 0
        self.stop_sign_cooldown   = 0
        self.frame_count          = 0
        self.total_error          = 0
        self.stop_signs_detected  = 0
        self._last_distance       = 999.0
        self._last_steering_error = 0.0
        self._start_time          = 0.0

    # ── Status hold ───────────────────────────────────────────────────────────

    def _update_display_status(self, new_status: str) -> str:
        if new_status == self._display_status:
            self._status_hold_frames = self._STATUS_MIN_HOLD
            return self._display_status
        self._status_hold_frames -= 1
        if self._status_hold_frames <= 0:
            self._display_status     = new_status
            self._status_hold_frames = self._STATUS_MIN_HOLD
        return self._display_status

    # ── Hardware helpers ──────────────────────────────────────────────────────

    def _get_distance(self) -> float:
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
            print(f"[Web] Speed limit → {self.current_speed_limit:.2f}")
        elif action == "set_target_color":
            self.color_detector.set_target(str(cmd.get("value", "GREEN")))
        pi_server.clear_command()

    # ── Telemetry ─────────────────────────────────────────────────────────────

    def _build_telemetry(self, left: float, right: float, status: str) -> dict:
        elapsed = max(time.time() - self._start_time, 0.1)
        fps     = round(self.frame_count / elapsed, 1)

        yolo_dets = (
            [{"label": d[1], "conf": round(d[3], 2)}
             for d in self.detector.all_detections]
            if self.yolo_enabled else []
        )
        yolo_danger = self.detector.get_danger_level() if self.yolo_enabled else "CLEAR"

        color_dets = [
            {"color": c, "side": det["side"], "area": int(det["area"])}
            for c, det in self.color_detector.detections.items()
        ]

        return {
            "mode":                  "AUTONOMOUS" if self.autonomous_enabled else "MANUAL",
            "status":                status,
            "speed_left":            round(left, 3),
            "speed_right":           round(right, 3),
            "base_speed":            round(self.smooth_base_speed.value, 3),
            "steering_error":        round(self._last_steering_error, 1),
            "distance_cm":           self._last_distance,
            "fps":                   fps,
            "frame_count":           self.frame_count,
            "stop_signs_detected":   self.stop_signs_detected,
            "yolo_enabled":          self.yolo_enabled,
            "yolo_detections":       yolo_dets,
            "yolo_danger":           yolo_danger,
            "obstacle_modifier":     round(
                self.detector.get_speed_modifier() if self.yolo_enabled else 1.0, 2
            ),
            "fork_confidence":       round(self.intersection_detector.fork_confidence, 2),
            "color_target":          self.color_detector.target_color,
            "color_detections":      color_dets,
            "color_follow_active":   self._color_follow_frames > 0,
            "color_target_visible":  self.color_detector.target_visible(),
        }

    # ── Main perception + decision loop ──────────────────────────────────────

    def process_frame(self, frame):
        self.frame_count += 1
        steering_error, vision_frame = self.perception.process_frame(frame)

        # ── No lane detected — emergency stop ─────────────────────────
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

        self.total_error          += abs(steering_error)
        self._last_steering_error  = steering_error

        # ── Ultrasonic (every 3 frames to avoid blocking) ─────────────
        if self.frame_count % 3 == 0:
            self._last_distance = self._get_distance()
        self.safety.sensors['front']['distance'] = self._last_distance
        self.safety._check_obstacles()

        # ── Sign / obstacle detection ──────────────────────────────────
        if self.yolo_enabled:
            self.detector.detect(frame)
            obstacle_modifier = self.detector.get_speed_modifier()
        else:
            self.detector.detect_signs(frame)
            obstacle_modifier = self.safety.get_speed_modifier()

        sign_action, _ = self.detector.get_action()

        if self.stop_sign_cooldown > 0:
            self.stop_sign_cooldown -= 1
        if sign_action == "STOP" and self.stop_sign_cooldown == 0:
            if self.stop_sign_timer == 0:
                self.stop_sign_timer    = 60
                self.stop_sign_cooldown = 120
                self.stop_signs_detected += 1
                print("STOP SIGN — holding for 2 s")

        # ── Colour tape detection (runs every frame — cheap HSV mask) ──
        self.color_detector.detect(frame)

        # ── Intersection / fork detection ─────────────────────────────
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

        # Fork detected → arm colour-follow window
        if fork_detected:
            if self.color_detector.target_visible():
                self._color_follow_frames = _COLOR_FOLLOW_FRAMES
                print(f"[Fork] → following {self.color_detector.target_color} "
                      f"({self.color_detector.target_det['side']})")
            else:
                print("[Fork] Detected but target tape not visible — continuing lane follow")

        # ── Speed decision ────────────────────────────────────────────
        if self.stop_sign_timer > 0:
            base_speed = 0.0
            self.stop_sign_timer -= 1
            status = f"STOPPED ({self.stop_sign_timer} frames)"
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
                    # Proportional gentle turn toward the target tape
                    steer      = offset * 0.40
                    left_speed  = max(0.0, min(1.0, base_speed + steer))
                    right_speed = max(0.0, min(1.0, base_speed - steer))
                    pid_output  = steer
                    status = (f"COLOR: {self.color_detector.target_color} "
                              f"→ {self.color_detector.target_det['side']}")
                else:
                    # Tape not visible right now — slow and keep lane following
                    pid_output  = self.steering.compute(steering_error)
                    left_speed  = max(0.0, min(1.0, base_speed * 0.5 + pid_output))
                    right_speed = max(0.0, min(1.0, base_speed * 0.5 - pid_output))
                    status      = "COLOR: SCANNING..."
            else:
                # Normal PID lane following
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

        # Sign status line
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

        # Fork confidence
        conf = self.intersection_detector.fork_confidence
        if conf > 0.2:
            fork_color = (0, 255, 255) if conf > 0.45 else (100, 200, 200)
            cv2.putText(frame, f"Fork: {conf:.0%}",
                        (10, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.4, fork_color, 1)

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

        for bx, label in ((bar_x_left, "L"), (bar_x_right, "R")):
            cv2.putText(frame, label, (bx + 12, bar_y - 5),
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
        print("[System] All systems reset")

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self):
        if self.web_enabled:
            ok = pi_server.start_server(self.web_port)
            if not ok:
                print("[WebServer] Flask not installed — dashboard disabled")
                self.web_enabled = False

        picam2 = Picamera2()
        picam2.configure(picam2.create_preview_configuration(
            main={"format": "RGB888", "size": (640, 480)}
        ))
        picam2.start()
        time.sleep(1)
        print(f"[Camera] 640×480 ready"
              f"{' + YOLO' if self.yolo_enabled else ''}"
              f" | tape target: {self.color_detector.target_color}")
        if self.show_display:
            print("Keys: [Q] Quit  [SPACE] Auto  [R] Reset  [S] Snap  "
                  "[D] Debug  [G/B/E] Tape Green/Blue/rEd")

        self._start_time = time.time()

        while True:
            frame = picam2.capture_array()
            debug_frame, (left, right, status) = self.process_frame(frame)

            if self.web_enabled:
                self._handle_web_commands()

            if self.autonomous_enabled:
                self._drive(left, right)
            else:
                self._stop_motors()

            if self.web_enabled:
                pi_server.push_frame(debug_frame)
                pi_server.push_telemetry(self._build_telemetry(left, right, status))

            if self.show_display:
                title = ("VectorVance  "
                         + ("AUTO" if self.autonomous_enabled else "MANUAL")
                         + (" [YOLO]" if self.yolo_enabled else "")
                         + f"  tape:{self.color_detector.target_color}")
                cv2.imshow(title, debug_frame)

            if self.frame_count % 30 == 0:
                elapsed = time.time() - self._start_time
                fps     = self.frame_count / max(elapsed, 0.1)
                avg_err = self.total_error / max(self.frame_count, 1)
                print(f"Frame {self.frame_count:04d} | "
                      f"L:{left:.2f} R:{right:.2f} | "
                      f"{status:35s} | "
                      f"Dist:{self._last_distance:.0f}cm | "
                      f"FPS:{fps:.1f} | AvgErr:{avg_err:.1f}px")

            if self.show_display:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord(' '):
                    self.autonomous_enabled = not self.autonomous_enabled
                    print(f"Autonomous: {'ON' if self.autonomous_enabled else 'OFF'}")
                elif key == ord('r'):
                    self._reset_all()
                elif key == ord('s'):
                    fname = f"/home/pi/snap_{self.frame_count:04d}.jpg"
                    cv2.imwrite(fname, debug_frame)
                    print(f"Snapshot: {fname}")
                elif key == ord('d'):
                    if self.yolo_enabled:
                        print(f"YOLO: {len(self.detector.all_detections)} dets | "
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

        picam2.stop()
        if self.show_display:
            cv2.destroyAllWindows()
        self._stop_motors()
        self._cleanup_hardware()
        self._print_stats()

    # ── End-of-run stats ──────────────────────────────────────────────────────

    def _print_stats(self):
        duration = time.time() - self._start_time
        print("=" * 60)
        print(f"Duration    : {duration:.1f}s")
        print(f"Frames      : {self.frame_count}")
        print(f"Avg FPS     : {self.frame_count / max(duration, 0.1):.1f}")
        print(f"Avg error   : {self.total_error / max(self.frame_count, 1):.1f}px")
        print(f"Stop signs  : {self.stop_signs_detected}")
        print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description="VectorVance Autonomous Car")
    p.add_argument("--speed",         type=float, default=0.8,
                   help="Max motor speed 0–1 (default 0.8)")
    p.add_argument("--yolo",          action="store_true",
                   help="Enable YOLOv8 detection")
    p.add_argument("--yolo-model",    type=str,   default="yolov8n.pt",
                   help="YOLO weights file (default: yolov8n.pt)")
    p.add_argument("--yolo-skip",     type=int,   default=5,
                   help="Run YOLO every N frames (default: 5)")
    p.add_argument("--target-color",  type=str,   default="GREEN",
                   choices=["GREEN", "BLUE", "RED"],
                   help="Tape colour to follow at forks (default: GREEN)")
    p.add_argument("--port",          type=int,   default=5000,
                   help="Web dashboard port (default: 5000)")
    p.add_argument("--no-web",        action="store_true",
                   help="Disable web dashboard")
    p.add_argument("--no-display",    action="store_true",
                   help="Headless mode — no OpenCV window")
    args = p.parse_args()

    AutonomousVehicle(
        max_speed    = args.speed,
        enable_yolo  = args.yolo,
        yolo_model   = args.yolo_model,
        yolo_skip    = args.yolo_skip,
        target_color = args.target_color,
        enable_web   = not args.no_web,
        web_port     = args.port,
        show_display = not args.no_display,
    ).run()


if __name__ == "__main__":
    main()
