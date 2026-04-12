"""
main.py - VectorVance Autonomous Car (PRODUCTION)
──────────────────────────────────────────────────
Hardware : Innomaker 1080P USB2.0 UVC 130° wide-angle camera (rear-mounted) + dual L298N motors + HC-SR04 ultrasonic
Detection: MobileNet SSD v2 (always active)
Control  : PID lane-follow + adaptive speed + obstacle avoidance
Fork nav : Stops at fork, waits for user to pick colour on dashboard,
           then follows that tape and stops on RED destination tape.
Dashboard: Live web UI at http://<pi-ip>:5000/

USAGE:
  python main.py                    # full autonomous (MobileNet SSD always on)
  python main.py --no-web           # disable web dashboard
  python main.py --no-display       # headless (no cv2 window)

KEYBOARD (when display is on):
  Q        Quit
  SPACE    Toggle autonomous mode
  R        Reset all systems
  S        Save snapshot
  D        Print detector debug info
  G / B    Set path colour Green / Blue  (same as dashboard buttons)
"""

import cv2
import time
import argparse

from gpiozero import Motor

from camera import FisheyeCamera
from perception import LaneDetector, SmoothValue
from controller import PIDController
from speed_controller import AdaptiveSpeedController, draw_speed_indicator
from safety import ObstacleDetector
from dnn_detector import DNNDetector
from intersection_detector import IntersectionDetector
from color_sign_detector import ColorSignDetector
from rear_monitor import RearMonitor
import pi_server

# ── GPIO pin assignments ──────────────────────────────────────────────────────
FL_FWD, FL_BWD = 25, 27
FR_FWD, FR_BWD = 5,  15
RL_FWD, RL_BWD = 26, 20
RR_FWD, RR_BWD = 16, 6
TRIG_PIN, ECHO_PIN = 4, 17
STOP_DISTANCE = 20
SLOW_DISTANCE = 50

# ── Car state machine ─────────────────────────────────────────────────────────
class State:
    LANE_FOLLOW     = "LANE_FOLLOW"      # normal driving
    FORK_WAITING    = "FORK_WAITING"     # stopped at fork, waiting for user input
    COLOR_FOLLOWING = "COLOR_FOLLOWING"  # steering toward chosen tape
    ARRIVED         = "ARRIVED"          # RED tape seen, stopped at destination
    FREE_ROAM       = "FREE_ROAM"        # FSD: obstacle-avoiding free roam, no lane


# ── FSD free-roam controller ──────────────────────────────────────────────────
class FreeRoamController:
    """
    Obstacle-avoiding controller for FSD mode.
    No lane detection — drives forward and steers away from obstacles.
    Uses DNN zone threats + ultrasonic for decisions.
    """
    BASE_SPEED    = 0.45
    TURN_SPEED    = 0.52
    COMMIT_FRAMES = 28   # hold a turn decision for ~1 s at 28 fps
    BACKUP_FRAMES = 15   # frames to reverse when ultrasonic triggers

    def __init__(self):
        self.turn_direction   = 0    # -1=left, 0=straight, +1=right
        self.committed_frames = 0
        self.backing_up       = 0

    def reset(self):
        self.turn_direction   = 0
        self.committed_frames = 0
        self.backing_up       = 0

    def compute(self, detector, distance_cm: float):
        """
        Returns (left_speed, right_speed, status_str).
        Negative values = reverse — caller must use _drive_manual().
        """
        # Backup countdown
        if self.backing_up > 0:
            self.backing_up -= 1
            return -0.35, -0.35, "FSD: BACKING UP"

        # Ultrasonic emergency → initiate backup
        if distance_cm < 20:
            self.backing_up       = self.BACKUP_FRAMES
            self.turn_direction   = 1 if self.turn_direction <= 0 else -1
            self.committed_frames = self.COMMIT_FRAMES
            return -0.35, -0.35, "FSD: BACKING UP"

        # Recalculate direction when commitment expires
        if self.committed_frames > 0:
            self.committed_frames -= 1
        else:
            zones          = detector.get_zone_threats()
            center_blocked = len(zones["center"]) > 0
            left_blocked   = len(zones["left"])   > 0
            right_blocked  = len(zones["right"])  > 0
            near_obstacle  = distance_cm < 50

            if center_blocked or near_obstacle:
                if left_blocked and not right_blocked:
                    new_dir = 1    # right side clear → turn right
                elif right_blocked and not left_blocked:
                    new_dir = -1   # left side clear  → turn left
                else:
                    new_dir = 1 if self.turn_direction <= 0 else -1
                self.turn_direction   = new_dir
                self.committed_frames = self.COMMIT_FRAMES
            else:
                self.turn_direction = 0

        # to turn left:  slow left motor, fast right → left=low, right=high
        # to turn right: fast left motor, slow right → left=high, right=low
        if self.turn_direction == -1:
            return self.BASE_SPEED * 0.10, self.TURN_SPEED, "FSD: TURN LEFT"
        elif self.turn_direction == 1:
            return self.TURN_SPEED, self.BASE_SPEED * 0.10, "FSD: TURN RIGHT"
        else:
            return self.BASE_SPEED, self.BASE_SPEED, "FSD: FORWARD"


# ─────────────────────────────────────────────────────────────────────────────
class AutonomousVehicle:

    def __init__(self,
                 max_speed        = 0.8,
                 dnn_model        = "ssd_mobilenet_v2_coco.pb",
                 dnn_skip         = 5,
                 enable_web       = True,
                 web_port         = 5000,
                 show_display     = True,
                 fov_deg          = 130.0,
                 undistort        = True,
                 calibration_file = None,
                 cam_index        = -1):

        # ── Camera (wide-angle undistortion for 130° Innomaker lens) ────
        self.undistort_enabled = undistort
        if undistort:
            if calibration_file:
                self.camera = FisheyeCamera.from_file(calibration_file)
            else:
                self.camera = FisheyeCamera(fov_deg=fov_deg)
        else:
            self.camera = None
            print(f"[Camera] Undistortion disabled (FOV={fov_deg}°)")

        # ── Perception & control ──────────────────────────────────────
        self.perception    = LaneDetector(width=640, height=480)
        self.steering      = PIDController(Kp=0.003, Ki=0.0001, Kd=0.001)
        self.speed_control = AdaptiveSpeedController(min_speed=0.2, max_speed=max_speed)
        self.safety        = ObstacleDetector(
            emergency_distance=STOP_DISTANCE, warning_distance=SLOW_DISTANCE
        )
        self.intersection_detector = IntersectionDetector()

        # ── Sign / obstacle detector ──────────────────────────────────
        self.detector = DNNDetector(model_name=dnn_model, skip_frames=dnn_skip)

        # ── Colour tape detector ──────────────────────────────────────
        self.color_detector = ColorSignDetector(frame_width=640, frame_height=480)

        # ── State machine ─────────────────────────────────────────────
        self.car_state  = State.LANE_FOLLOW
        self.drive_mode = "LANE"   # "FSD" | "LANE" | "MANUAL"
        self.free_roam  = FreeRoamController()

        # ── Motors ───────────────────────────────────────────────────
        self.front_left  = Motor(forward=FL_FWD, backward=FL_BWD)
        self.rear_left   = Motor(forward=RL_FWD, backward=RL_BWD)
        self.front_right = Motor(forward=FR_FWD, backward=FR_BWD)
        self.rear_right  = Motor(forward=RR_FWD, backward=RR_BWD)
        print("[Motors] All 4 motors OK")

        # ── Rear ultrasonic + collision alerts ───────────────────────
        self.rear_monitor = RearMonitor(trig=TRIG_PIN, echo=ECHO_PIN)
        self.rear_monitor.start()

        # ── Web / display ─────────────────────────────────────────────
        self.web_enabled  = enable_web
        self.web_port     = web_port
        self.show_display = show_display

        # ── Smooth display values ─────────────────────────────────────
        self.smooth_left       = SmoothValue(0.0, alpha=0.18)
        self.smooth_right      = SmoothValue(0.0, alpha=0.18)
        self.smooth_base_speed = SmoothValue(0.0, alpha=0.15)
        self.smooth_pid        = SmoothValue(0.0, alpha=0.12)

        # ── Status hold ──────────────────────────────────────────────
        self.cam_index           = cam_index
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

    # ── Drive-mode switcher ───────────────────────────────────────────────────

    def _set_drive_mode(self, mode: str):
        """Switch between FSD / LANE / MANUAL modes safely."""
        if mode not in ("FSD", "LANE", "MANUAL"):
            print(f"[Mode] Unknown mode '{mode}' — ignored")
            return
        prev = self.drive_mode
        self.drive_mode = mode

        if mode == "MANUAL":
            self.autonomous_enabled = False
            self._stop_motors()
        elif mode == "LANE":
            self.autonomous_enabled = True
            if self.car_state == State.FREE_ROAM:
                self.car_state = State.LANE_FOLLOW
        elif mode == "FSD":
            self.autonomous_enabled = True
            self.car_state = State.FREE_ROAM
            self.free_roam.reset()

        print(f"[Mode] {prev} → {mode}")

    # ── Hardware helpers ──────────────────────────────────────────────────────

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

    def _drive_manual(self, left: float, right: float):
        """Manual drive: positive = forward, negative = backward."""
        left  = max(-1.0, min(1.0, left))
        right = max(-1.0, min(1.0, right))

        if abs(left) < 0.05:
            self.front_left.stop();  self.rear_left.stop()
        elif left > 0:
            self.front_left.backward(left);  self.rear_left.backward(left)
        else:
            self.front_left.forward(-left);  self.rear_left.forward(-left)

        if abs(right) < 0.05:
            self.front_right.stop(); self.rear_right.stop()
        elif right > 0:
            self.front_right.backward(right); self.rear_right.backward(right)
        else:
            self.front_right.forward(-right); self.rear_right.forward(-right)

    def _apply_manual_drive(self, keys: dict):
        """Translate WASD key state into motor commands."""
        spd = 0.65
        w, a, s, d = keys.get("w"), keys.get("a"), keys.get("s"), keys.get("d")
        if w:
            if a:   self._drive_manual(spd * 0.25, spd)    # forward-left
            elif d: self._drive_manual(spd, spd * 0.25)    # forward-right
            else:   self._drive_manual(spd, spd)            # straight forward
        elif s:
            if a:   self._drive_manual(-spd * 0.25, -spd)  # reverse-left
            elif d: self._drive_manual(-spd, -spd * 0.25)  # reverse-right
            else:   self._drive_manual(-spd, -spd)          # straight reverse
        elif a:     self._drive_manual(-spd * 0.5,  spd * 0.5)  # spin left
        elif d:     self._drive_manual( spd * 0.5, -spd * 0.5)  # spin right
        else:       self._stop_motors()

    def _cleanup_hardware(self):
        self._stop_motors()
        for m in (self.front_left, self.rear_left,
                  self.front_right, self.rear_right):
            m.close()
        self.rear_monitor.stop()
        print("[Hardware] GPIO released")

    # ── FSD frame processing ─────────────────────────────────────────────────

    def process_frame_fsd(self, frame):
        """Lane-free frame loop for FSD mode. Uses DNN + ultrasonic only."""
        self.frame_count += 1

        # DNN obstacle detection (internally caches on skip_frames)
        self.detector.detect(frame)

        # FreeRoam decision — no front ultrasonic (sensor is rear-facing)
        left, right, status = self.free_roam.compute(
            self.detector, 999.0
        )

        self.smooth_left.update(abs(left))
        self.smooth_right.update(abs(right))
        self.smooth_base_speed.update(self.free_roam.BASE_SPEED)
        self.smooth_pid.update(0.0)
        self._last_steering_error = 0.0

        # Build debug frame
        debug = frame.copy()
        debug = self.safety.draw_overlay(debug)
        debug = self.detector.draw_overlay(debug)
        debug = self._draw_motor_bars(
            debug, abs(left), abs(right), 0.0
        )

        fsd_color = (0, 0, 255) if "BACKUP" in status else \
                    (0, 165, 255) if "TURN" in status else (0, 212, 255)
        cv2.putText(debug, f"[FSD] {status}",
                    (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, fsd_color, 2)
        cv2.putText(debug, f"Rear: {self.rear_monitor.distance_cm:.0f} cm",
                    (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        danger = self.detector.get_danger_level()
        danger_colors = {
            "CLEAR":   (100, 200, 100), "CAUTION": (0, 200, 255),
            "DANGER":  (0, 100, 255),   "STOP":    (0, 0, 255),
        }
        cv2.putText(debug, f"DNN: {danger}",
                    (10, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    danger_colors.get(danger, (150, 150, 150)), 1)

        return debug, (left, right, status)

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
            color = str(cmd.get("value", "")).upper()
            if color in ("GREEN", "BLUE"):
                self.color_detector.set_target(color)
                # Only start moving if we are currently waiting at a fork
                if self.car_state == State.FORK_WAITING:
                    self.car_state = State.COLOR_FOLLOWING
                    print(f"[Fork] User selected {color} — resuming toward tape")
            else:
                print(f"[Web] Invalid colour '{color}'")

        elif action == "set_mode":
            mode = str(cmd.get("value", "LANE")).upper()
            self._set_drive_mode(mode)

        pi_server.clear_command()

    # ── Telemetry ─────────────────────────────────────────────────────────────

    def _build_telemetry(self, left: float, right: float, status: str) -> dict:
        elapsed   = max(time.time() - self._start_time, 0.1)
        dnn_dets = [{"label": d[1], "conf": round(d[3], 2)}
                    for d in self.detector.all_detections]
        dnn_danger = self.detector.get_danger_level()

        color_dets = [
            {"color": c, "side": det["side"]}
            for c, det in self.color_detector.detections.items()
        ]
        fork_options = self.color_detector.get_fork_options()

        return {
            "mode":                  "AUTONOMOUS" if self.autonomous_enabled else "MANUAL",
            "drive_mode":            self.drive_mode,
            "status":                status,
            "car_state":             self.car_state,
            "fork_waiting":          self.car_state == State.FORK_WAITING,
            "fork_options":          fork_options,
            "speed_left":            round(left, 3),
            "speed_right":           round(right, 3),
            "base_speed":            round(self.smooth_base_speed.value, 3),
            "steering_error":        round(self._last_steering_error, 1),
            "rear_distance_cm":       self.rear_monitor.distance_cm,
            "rear_alert_count":       self.rear_monitor.get_status()["rear_alert_count"],
            "fps":                   round(self.frame_count / elapsed, 1),
            "frame_count":           self.frame_count,
            "stop_signs_detected":   self.stop_signs_detected,
            "dnn_enabled":           True,
            "dnn_detections":        dnn_dets,
            "dnn_danger":            dnn_danger,
            "obstacle_modifier":     round(self.detector.get_speed_modifier(), 2),
            "fork_confidence":       round(self.intersection_detector.fork_confidence, 2),
            "color_target":          self.color_detector.target_color,
            "color_detections":      color_dets,
            "color_target_visible":  self.color_detector.target_visible(),
        }

    # ── Main perception + decision loop ──────────────────────────────────────

    def process_frame(self, frame):
        self.frame_count += 1
        steering_error, vision_frame = self.perception.process_frame(frame)

        # ── No lane → emergency stop ───────────────────────────────────
        if steering_error is None:
            for s in (self.smooth_left, self.smooth_right,
                      self.smooth_base_speed, self.smooth_pid):
                s.update(0.0)
            return (
                self._create_debug_frame(
                    vision_frame, 0, 0.0, 0.0, 0.0, 0.0, "EMERGENCY STOP", "NONE"
                ),
                (0.0, 0.0, "EMERGENCY STOP"),
            )

        self.total_error          += abs(steering_error)
        self._last_steering_error  = steering_error

        # ── Sign / obstacle detection ──────────────────────────────────
        self.detector.detect(frame)
        obstacle_modifier = self.detector.get_speed_modifier()

        sign_action, _ = self.detector.get_action()
        if self.stop_sign_cooldown > 0:
            self.stop_sign_cooldown -= 1
        if sign_action == "STOP" and self.stop_sign_cooldown == 0:
            if self.stop_sign_timer == 0:
                self.stop_sign_timer    = 60
                self.stop_sign_cooldown = 120
                self.stop_signs_detected += 1
                print("STOP SIGN — holding 2 s")

        # ── Colour tape detection (every frame, cheap) ─────────────────
        self.color_detector.detect(frame)

        # ── State machine transitions ──────────────────────────────────

        if self.car_state == State.LANE_FOLLOW:
            # Check for fork
            num_raw_lines, lane_width = self._measure_lines(frame)
            fork_detected = self.intersection_detector.update(
                num_lines        = num_raw_lines,
                left_confidence  = self.perception.left_confidence,
                right_confidence = self.perception.right_confidence,
                lane_width       = lane_width,
                left_fit         = self.perception.ema_left_fit,
                right_fit        = self.perception.ema_right_fit,
            )
            if fork_detected:
                options = self.color_detector.get_fork_options()
                print(f"[Fork] Detected — options visible: {options or 'NONE'}")
                print("[Fork] Car STOPPED — waiting for dashboard colour selection")
                self.car_state = State.FORK_WAITING

        elif self.car_state == State.COLOR_FOLLOWING:
            # Check if destination reached (RED tape)
            if self.color_detector.destination_reached():
                self.car_state = State.ARRIVED
                print("[ARRIVED] RED tape confirmed — destination reached!")

        # ── Speed decision ────────────────────────────────────────────
        if self.stop_sign_timer > 0:
            base_speed = 0.0
            self.stop_sign_timer -= 1
            status = f"STOPPED: sign ({self.stop_sign_timer})"

        elif self.car_state == State.FORK_WAITING:
            base_speed = 0.0
            options    = self.color_detector.get_fork_options()
            status     = f"FORK: pick {' or '.join(options) if options else 'colour'}"

        elif self.car_state == State.ARRIVED:
            base_speed = 0.0
            status     = "ARRIVED AT DESTINATION"

        else:
            base_speed = self.speed_control.calculate_speed(steering_error, obstacle_modifier)
            base_speed = min(base_speed, self.current_speed_limit)
            status     = self.speed_control.get_speed_category(
                abs(steering_error)
            ).replace("_", " ")

        # ── Steering ─────────────────────────────────────────────────
        if self.autonomous_enabled and base_speed > 0:

            if self.car_state == State.COLOR_FOLLOWING:
                offset = self.color_detector.get_steering_offset()
                if offset is not None:
                    steer      = offset * 0.40
                    left_speed  = max(0.0, min(1.0, base_speed + steer))
                    right_speed = max(0.0, min(1.0, base_speed - steer))
                    pid_output  = steer
                    target      = self.color_detector.target_color
                    side        = self.color_detector.target_det['side']
                    status      = f"FOLLOWING {target} → {side}"
                else:
                    # Tape not visible — slow down and scan with PID
                    pid_output  = self.steering.compute(steering_error)
                    left_speed  = max(0.0, min(1.0, base_speed * 0.5 + pid_output))
                    right_speed = max(0.0, min(1.0, base_speed * 0.5 - pid_output))
                    status      = f"SEEKING {self.color_detector.target_color}..."
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

        return (
            self._create_debug_frame(
                vision_frame, steering_error, pid_output,
                base_speed, left_speed, right_speed, status, sign_action
            ),
            (left_speed, right_speed, status),
        )

    # ── Line measurement helper ───────────────────────────────────────────────

    def _measure_lines(self, frame):
        gray  = cv2.cvtColor(cv2.resize(frame, (640, 480)), cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
        lines = cv2.HoughLinesP(edges, 2, 3.14159 / 180, 30,
                                minLineLength=40, maxLineGap=150)
        num   = len(lines) if lines is not None else 0
        lane_width = None
        if (self.perception.ema_left_fit is not None and
                self.perception.ema_right_fit is not None):
            y    = int(480 * 0.75)
            lx   = self.perception._eval_fit(self.perception.ema_left_fit,  y)
            rx   = self.perception._eval_fit(self.perception.ema_right_fit, y)
            if lx is not None and rx is not None:
                lane_width = rx - lx
        return num, lane_width

    # ── HUD ───────────────────────────────────────────────────────────────────

    def _create_debug_frame(self, vision_frame, error, pid_output,
                            base_speed, left_speed, right_speed,
                            status, sign_action):
        frame = vision_frame.copy()
        h, w = frame.shape[:2]
        frame = self.safety.draw_overlay(frame)
        frame = self.detector.draw_overlay(frame)
        frame = self.color_detector.draw_overlay(frame)

        # ── SPEED BADGE (top-right) ──────────────────────────────────
        speed_pct = int(self.smooth_base_speed.value * 100)
        category = self.speed_control.get_speed_category(abs(error)).replace("_", " ").upper()

        cv2.putText(frame, category, (w - 200, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 200, 150), 1)

        badge_text = f"Speed: {speed_pct}%"
        (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        bx1, by1 = w - tw - 24, 28
        bx2, by2 = w - 8, 28 + th + 14
        if speed_pct >= 70:
            badge_bg = (0, 180, 0)
        elif speed_pct >= 40:
            badge_bg = (0, 160, 255)
        else:
            badge_bg = (0, 80, 255)
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), badge_bg, -1)
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (255, 255, 255), 1)
        cv2.putText(frame, badge_text, (bx1 + 8, by2 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # ── STATUS (left side with background) ───────────────────────
        display_status = self._update_display_status(status)
        if self.car_state == State.FORK_WAITING:
            status_color, status_bg = (0, 230, 255), (0, 80, 100)
        elif self.car_state == State.ARRIVED:
            status_color, status_bg = (0, 255, 100), (0, 60, 0)
        elif "STOP" in display_status:
            status_color, status_bg = (0, 0, 255), (0, 0, 100)
        elif "SLOW" in display_status:
            status_color, status_bg = (0, 180, 255), (0, 60, 100)
        else:
            status_color, status_bg = (200, 255, 200), (0, 60, 0)

        status_text = f"Status: {display_status}"
        (stw, sth), _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (8, 115), (stw + 20, 140), status_bg, -1)
        cv2.rectangle(frame, (8, 115), (stw + 20, 140), status_color, 1)
        cv2.putText(frame, status_text,
                    (12, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

        # Big overlay banners for important states
        if self.car_state == State.FORK_WAITING:
            options = self.color_detector.get_fork_options()
            opt_str = " or ".join(options) if options else "colour"
            cv2.rectangle(frame, (0, h//2 - 30), (w, h//2 + 30), (0, 120, 200), -1)
            cv2.putText(frame, f"FORK -- SELECT: {opt_str}",
                        (w//2 - 180, h//2 + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        elif self.car_state == State.ARRIVED:
            cv2.rectangle(frame, (0, h//2 - 30), (w, h//2 + 30), (0, 160, 0), -1)
            cv2.putText(frame, "ARRIVED AT DESTINATION",
                        (w//2 - 210, h//2 + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        # DNN status
        danger = self.detector.get_danger_level()
        danger_colors = {
            "CLEAR":   (100, 220, 100),
            "CAUTION": (0,   200, 255),
            "DANGER":  (0,   100, 255),
            "STOP":    (0,   0,   255),
        }
        cv2.putText(frame, f"DNN: {danger}", (12, 162),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    danger_colors.get(danger, (150, 150, 150)), 1)

        # Fork confidence
        conf = self.intersection_detector.fork_confidence
        if conf > 0.2:
            fork_color = (0, 255, 255) if conf > 0.45 else (100, 200, 200)
            cv2.putText(frame, f"Fork: {conf:.0%}",
                        (12, 182), cv2.FONT_HERSHEY_SIMPLEX, 0.4, fork_color, 1)

        # ── BOTTOM BAR — rear distance + DNN obstacle ────────────────
        rear_dist = self.rear_monitor.distance_cm
        modifier  = self.detector.get_speed_modifier()
        if rear_dist < SLOW_DISTANCE or modifier < 1.0:
            bar_y = h - 30
            cv2.rectangle(frame, (0, bar_y), (w, h), (0, 0, 0), -1)
            if rear_dist < SLOW_DISTANCE:
                cx = w // 2
                circ_color = (0, 0, 255) if rear_dist < STOP_DISTANCE else (0, 165, 255)
                cv2.circle(frame, (cx, bar_y + 15), 10, circ_color, -1)
                cv2.putText(frame, f"REAR {rear_dist:.0f}cm", (cx - 30, bar_y + 12),
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 255), 1)
            if modifier < 1.0:
                obs_text = "OBSTACLE DETECTED -- SLOWING" if modifier > 0 else "OBSTACLE -- STOPPING"
                obs_color = (0, 165, 255) if modifier > 0 else (0, 0, 255)
                cv2.putText(frame, obs_text, (10, h - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, obs_color, 2)

        # Motor bars
        frame = self._draw_motor_bars(
            frame, self.smooth_left.value, self.smooth_right.value, self.smooth_pid.value)

        return frame

    def _draw_motor_bars(self, frame, left_speed, right_speed, pid_output):
        h, w        = frame.shape[:2]
        bar_width   = 35
        bar_height  = 160
        bar_x_left  = w - 105
        bar_x_right = w - 50
        bar_y       = h - bar_height - 55

        for bx in (bar_x_left, bar_x_right):
            cv2.rectangle(frame, (bx, bar_y),
                          (bx + bar_width, bar_y + bar_height), (30, 30, 30), -1)

        left_fill  = int(bar_height * max(0.0, min(1.0, left_speed)))
        right_fill = int(bar_height * max(0.0, min(1.0, right_speed)))
        bar_color  = (0, 230, 100) if abs(pid_output) < 0.1 else (0, 180, 255)

        if left_fill:
            cv2.rectangle(frame,
                          (bar_x_left,  bar_y + bar_height - left_fill),
                          (bar_x_left  + bar_width, bar_y + bar_height), bar_color, -1)
        if right_fill:
            cv2.rectangle(frame,
                          (bar_x_right, bar_y + bar_height - right_fill),
                          (bar_x_right + bar_width, bar_y + bar_height), bar_color, -1)

        for bx in (bar_x_left, bar_x_right):
            cv2.rectangle(frame, (bx, bar_y),
                          (bx + bar_width, bar_y + bar_height), (80, 80, 80), 1)

        for bx, lbl in ((bar_x_left, "L"), (bar_x_right, "R")):
            cv2.putText(frame, lbl, (bx + 10, bar_y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.putText(frame, f"{left_speed:.2f}",
                    (bar_x_left - 2,  bar_y + bar_height + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        cv2.putText(frame, f"{right_speed:.2f}",
                    (bar_x_right - 2, bar_y + bar_height + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
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
        self.free_roam.reset()
        self.car_state           = State.LANE_FOLLOW
        self.drive_mode          = "LANE"
        self.autonomous_enabled  = True
        self.current_speed_limit = 0.8
        self.stop_sign_timer     = 0
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

        # ── Innomaker 1080P USB2.0 130° camera (rear-mounted, single camera) ──
        self._cap = None
        search = [self.cam_index] if self.cam_index >= 0 else range(4)
        for idx in search:
            _c = cv2.VideoCapture(idx)
            if _c.isOpened():
                self._cap = _c
                print(f"[Camera] Innomaker 130° USB camera found at /dev/video{idx}")
                break
            _c.release()
        if self._cap is None:
            raise RuntimeError(
                "[Camera] Innomaker USB camera not found!\n"
                "  Check cable, then run: ls /dev/video*\n"
                "  If found at a different index, set it with: --cam-index N"
            )
        self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self._cap.set(cv2.CAP_PROP_FPS, 30)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        time.sleep(1)
        print("[Camera] 640×480 ready + MobileNet SSD")
        if self.show_display:
            print("Keys: [Q] Quit  [SPACE] Auto/Manual  [F] FSD  [R] Reset  "
                  "[S] Snap  [D] Debug  [G] Green path  [B] Blue path")

        self._start_time = time.time()

        while True:
            ret, frame = self._cap.read()
            if not ret:
                print("[Camera] Frame grab failed — retrying...")
                continue
            frame = cv2.rotate(frame, cv2.ROTATE_180)  # camera mounted upside-down
            if self.undistort_enabled and self.camera:
                frame = self.camera.undistort(frame)

            # ── Frame processing (mode-dependent) ────────────────────────
            if self.drive_mode == "FSD":
                debug_frame, (left, right, status) = self.process_frame_fsd(frame)
            else:
                debug_frame, (left, right, status) = self.process_frame(frame)

            if self.web_enabled:
                self._handle_web_commands()

            # ── Motor control (mode-dependent) ───────────────────────────
            if self.drive_mode == "FSD":
                # _drive_manual handles negative values for backup maneuver
                self._drive_manual(left, right)
            elif self.drive_mode == "LANE":
                if self.car_state not in (State.FORK_WAITING, State.ARRIVED):
                    self._drive(left, right)
                else:
                    self._stop_motors()
            else:  # MANUAL
                if self.web_enabled:
                    self._apply_manual_drive(pi_server.get_manual_keys())
                else:
                    self._stop_motors()

            if self.web_enabled:
                pi_server.push_frame(debug_frame)
                pi_server.push_telemetry(self._build_telemetry(left, right, status))

            if self.show_display:
                state_info = f"  [{self.car_state}]" if self.drive_mode == "LANE" else ""
                title = f"VectorVance  [{self.drive_mode}]{state_info}"
                cv2.imshow(title, debug_frame)

            if self.frame_count % 30 == 0:
                elapsed = time.time() - self._start_time
                fps     = self.frame_count / max(elapsed, 0.1)
                avg_err = self.total_error / max(self.frame_count, 1)
                print(f"Frame {self.frame_count:04d} | "
                      f"State:{self.car_state:18s} | "
                      f"L:{left:.2f} R:{right:.2f} | "
                      f"Rear:{self.rear_monitor.distance_cm:.0f}cm | "
                      f"FPS:{fps:.1f} | AvgErr:{avg_err:.1f}px")

            if self.show_display:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord(' '):
                    # Toggle between current auto mode and MANUAL
                    if self.drive_mode == "MANUAL":
                        self._set_drive_mode("LANE")
                    else:
                        self._set_drive_mode("MANUAL")
                elif key == ord('f'):
                    self._set_drive_mode("FSD")
                elif key == ord('r'):
                    self._reset_all()
                elif key == ord('s'):
                    fname = f"/home/pi/snap_{self.frame_count:04d}.jpg"
                    cv2.imwrite(fname, debug_frame)
                    print(f"Snapshot: {fname}")
                elif key == ord('d'):
                    print(f"State: {self.car_state}")
                    print(f"Tape detections: {self.color_detector.detections}")
                    print(f"DNN: {len(self.detector.all_detections)} dets | "
                          f"danger={self.detector.get_danger_level()}")
                elif key == ord('g'):
                    self.color_detector.set_target("GREEN")
                    if self.car_state == State.FORK_WAITING:
                        self.car_state = State.COLOR_FOLLOWING
                elif key == ord('b'):
                    self.color_detector.set_target("BLUE")
                    if self.car_state == State.FORK_WAITING:
                        self.car_state = State.COLOR_FOLLOWING

        if self._cap:
            self._cap.release()
        if self.show_display:
            cv2.destroyAllWindows()
        self._stop_motors()
        self._cleanup_hardware()
        elapsed = time.time() - self._start_time
        print("=" * 60)
        print(f"Duration    : {elapsed:.1f}s")
        print(f"Frames      : {self.frame_count}")
        print(f"Avg FPS     : {self.frame_count / max(elapsed, 0.1):.1f}")
        print(f"Stop signs  : {self.stop_signs_detected}")
        print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description="VectorVance Autonomous Car")
    p.add_argument("--speed",       type=float, default=0.8)
    p.add_argument("--dnn-model",     type=str,   default="ssd_mobilenet_v2_coco.pb")
    p.add_argument("--dnn-skip",      type=int,   default=5)
    p.add_argument("--port",          type=int,   default=5000)
    p.add_argument("--no-web",        action="store_true")
    p.add_argument("--no-display",    action="store_true")
    p.add_argument("--fov",           type=float, default=130.0,
                   help="Camera FOV in degrees (default: 130 for Innomaker wide-angle)")
    p.add_argument("--no-undistort",  action="store_true",
                   help="Disable wide-angle lens undistortion")
    p.add_argument("--calibration",   type=str,   default=None,
                   help="Path to calibration .npz file (uses approximation if omitted)")
    p.add_argument("--cam-index",     type=int,   default=-1,
                   help="Force Innomaker USB camera device index (default: auto-detect)")
    args = p.parse_args()

    AutonomousVehicle(
        max_speed        = args.speed,
        dnn_model        = args.dnn_model,
        dnn_skip         = args.dnn_skip,
        enable_web       = not args.no_web,
        web_port         = args.port,
        show_display     = not args.no_display,
        fov_deg          = args.fov,
        undistort        = not args.no_undistort,
        calibration_file = args.calibration,
        cam_index        = args.cam_index,
    ).run()


if __name__ == "__main__":
    main()
