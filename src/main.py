"""
main.py - VectorVance Autonomous Car
Live camera dashboard on Raspberry Pi using picamera2.
Lane detection + PID steering + adaptive speed + STOP sign detection.
Hardware: Dual L298N motor drivers + HC-SR04 ultrasonic sensor.
"""

import cv2
import time
import lgpio
import urllib.request
import json
from gpiozero import Motor
from picamera2 import Picamera2
from perception import LaneDetector
from controller import PIDController
from speed_controller import AdaptiveSpeedController, draw_speed_indicator
from safety import ObstacleDetector
from sign_detector import TrafficSignDetector
from color_sign_detector import ColorSignDetector

# ── GPIO pin assignments (confirmed working from smart_car.py) ────────────────
# Motor Driver 1 — Front Left (IN1/IN2) and Front Right (IN3/IN4)
FL_FWD, FL_BWD = 25, 27
FR_FWD, FR_BWD = 5, 15
# Motor Driver 2 — Rear Left (IN5/IN6) and Rear Right (IN7/IN8)
RL_FWD, RL_BWD = 26, 20
RR_FWD, RR_BWD = 16, 6
# Ultrasonic HC-SR04
TRIG_PIN, ECHO_PIN = 4, 17
STOP_DISTANCE = 20   # cm — emergency stop threshold
SLOW_DISTANCE = 50   # cm — slow-down threshold

# ── Web dashboard ─────────────────────────────────────────────────────────────
DASHBOARD_URL  = "http://localhost:3000"   # change to Pi's IP if dashboard runs elsewhere
FORK_HOLD_FRAMES = 15   # how many frames fork must be visible before pausing


class AutonomousVehicle:
    def __init__(self, max_speed=0.8):
        self.perception = LaneDetector(width=640, height=480)
        self.steering = PIDController(Kp=0.003, Ki=0.0001, Kd=0.001)
        self.speed_control = AdaptiveSpeedController(min_speed=0.2, max_speed=max_speed)
        self.safety = ObstacleDetector(
            emergency_distance=STOP_DISTANCE,
            warning_distance=SLOW_DISTANCE
        )
        self.sign_detector  = TrafficSignDetector()
        self.color_detector = ColorSignDetector()

        # ── fork / intersection state ─────────────────────────────────────────
        self._fork_frame_count  = 0     # consecutive frames fork has been visible
        self._at_fork           = False # True while waiting for dashboard command
        self._fork_command      = None  # color to execute, set by dashboard poll

        # ── motors (gpiozero Motor uses PWM on direction pins) ────────────────
        self.front_left  = Motor(forward=FL_FWD, backward=FL_BWD)
        self.rear_left   = Motor(forward=RL_FWD, backward=RL_BWD)
        self.front_right = Motor(forward=FR_FWD, backward=FR_BWD)
        self.rear_right  = Motor(forward=RR_FWD, backward=RR_BWD)
        print("[Motors] All 4 motors initialised OK")

        # ── ultrasonic sensor (lgpio) ─────────────────────────────────────────
        self._gpio = lgpio.gpiochip_open(0)
        lgpio.gpio_claim_output(self._gpio, TRIG_PIN)
        lgpio.gpio_claim_input(self._gpio, ECHO_PIN)
        lgpio.gpio_write(self._gpio, TRIG_PIN, 0)
        time.sleep(0.1)
        print("[Ultrasonic] HC-SR04 initialised OK")

        self.autonomous_enabled = True
        self.current_speed_limit = max_speed
        self.stop_sign_timer = 0
        self.stop_sign_cooldown = 0
        self.frame_count = 0
        self.total_error = 0
        self.stop_signs_detected = 0
        self._last_distance = 999.0

    # ── dashboard helpers ─────────────────────────────────────────────────────

    def _post_json(self, path, data):
        try:
            body = json.dumps(data).encode()
            req  = urllib.request.Request(
                DASHBOARD_URL + path, data=body,
                headers={"Content-Type": "application/json"}, method="POST"
            )
            urllib.request.urlopen(req, timeout=1)
        except Exception:
            pass  # dashboard may not be running — don't crash the car

    def _get_json(self, path):
        try:
            with urllib.request.urlopen(DASHBOARD_URL + path, timeout=1) as r:
                return json.loads(r.read())
        except Exception:
            return {}

    def _notify_fork(self, colors):
        self._post_json("/api/fork_detected", {"colors": colors})

    def _poll_command(self):
        """Returns the chosen color string, or None if dashboard hasn't decided yet."""
        data = self._get_json("/api/poll_command")
        return data.get("command")

    def _notify_complete(self):
        self._post_json("/api/complete", {})

    def _execute_fork_turn(self, target_color):
        """
        Spin in place until the target color tape is centred in the camera,
        then drive straight down that path.
        The car pauses between frames, so we capture live frames here.
        """
        TURN_SPEED   = 0.45   # spin speed
        CENTER_TOL   = 60     # px — how close to centre counts as aligned
        MAX_SPIN_SEC = 4.0    # safety timeout

        print(f"[Fork] Aligning with {target_color.upper()} tape...")
        picam2 = getattr(self, '_picam2', None)   # set during run()
        if picam2 is None:
            return  # camera not available in this context

        start = time.time()
        while time.time() - start < MAX_SPIN_SEC:
            frame = picam2.capture_array()
            self.color_detector.detect(frame)
            signs = self.color_detector.get_both_signs()
            det = signs.get(target_color.lower())

            if det is not None:
                offset = det["center_x"] - 320   # 320 = centre of 640px frame
                if abs(offset) <= CENTER_TOL:
                    print(f"[Fork] Aligned! offset={offset}px — driving forward")
                    self._stop_motors()
                    time.sleep(0.1)
                    # Drive forward for 1 s to clear the fork zone
                    self._drive(0.5, 0.5)
                    time.sleep(1.0)
                    self._stop_motors()
                    return
                # Spin towards the sign
                if offset < 0:   # sign is left — spin left
                    self._drive_spin(TURN_SPEED, direction="left")
                else:            # sign is right — spin right
                    self._drive_spin(TURN_SPEED, direction="right")
            else:
                # Can't see it yet — keep spinning right slowly
                self._drive_spin(TURN_SPEED, direction="right")

            time.sleep(0.05)

        print("[Fork] Alignment timeout — continuing straight")
        self._stop_motors()

    def _drive_spin(self, speed, direction="right"):
        """Spin in place."""
        if direction == "right":
            # left side forward, right side backward
            self.front_left.backward(speed)
            self.rear_left.backward(speed)
            self.front_right.forward(speed)
            self.rear_right.forward(speed)
        else:
            self.front_left.forward(speed)
            self.rear_left.forward(speed)
            self.front_right.backward(speed)
            self.rear_right.backward(speed)

    # ── hardware helpers ──────────────────────────────────────────────────────

    def _get_distance(self):
        """Measure distance in cm with HC-SR04. Returns 999 on timeout."""
        lgpio.gpio_write(self._gpio, TRIG_PIN, 1)
        time.sleep(0.00001)
        lgpio.gpio_write(self._gpio, TRIG_PIN, 0)

        timeout = time.time() + 0.04
        start = time.time()
        while lgpio.gpio_read(self._gpio, ECHO_PIN) == 0:
            start = time.time()
            if time.time() > timeout:
                return 999.0

        stop = time.time()
        timeout = time.time() + 0.04
        while lgpio.gpio_read(self._gpio, ECHO_PIN) == 1:
            stop = time.time()
            if time.time() > timeout:
                return 999.0

        return round((stop - start) * 34300 / 2, 1)

    def _drive(self, left_speed, right_speed):
        """Send PWM speeds (0.0–1.0) to all four motors."""
        left_speed  = max(0.0, min(1.0, left_speed))
        right_speed = max(0.0, min(1.0, right_speed))

        if left_speed < 0.05:
            self.front_left.stop()
            self.rear_left.stop()
        else:
            self.front_left.backward(left_speed)
            self.rear_left.backward(left_speed)

        if right_speed < 0.05:
            self.front_right.stop()
            self.rear_right.stop()
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

    # ── perception + decision ─────────────────────────────────────────────────

    def process_frame(self, frame):
        self.frame_count += 1
        steering_error, vision_frame = self.perception.process_frame(frame)
        self.total_error += abs(steering_error)

        # Read real ultrasonic sensor every 3 frames to avoid slowing the loop
        if self.frame_count % 3 == 0:
            self._last_distance = self._get_distance()

        # Feed real distance into the safety system (replaces simulated sensors)
        self.safety.sensors['front']['distance'] = self._last_distance
        self.safety._check_obstacles()
        obstacle_modifier = self.safety.get_speed_modifier()

        # ── color sign / fork detection ───────────────────────────────────────
        self.color_detector.detect(frame)
        if self.color_detector.is_fork_visible():
            self._fork_frame_count += 1
            if self._fork_frame_count == FORK_HOLD_FRAMES and not self._at_fork:
                # Just arrived at the fork — pause and ping the dashboard
                self._at_fork = True
                colors = self.color_detector.get_confirmed_colors()
                print(f"[Fork] Detected: {colors} — notifying dashboard")
                self._notify_fork(colors)
        else:
            self._fork_frame_count = 0

        # While at fork: poll dashboard for the execute command
        if self._at_fork and self._fork_command is None:
            self._fork_command = self._poll_command()
            if self._fork_command:
                print(f"[Fork] Dashboard says: GO {self._fork_command.upper()}")

        # ── traffic sign detection ────────────────────────────────────────────
        detected_signs = self.sign_detector.detect_signs(frame)
        sign_action, _ = self.sign_detector.get_action()

        if self.stop_sign_cooldown > 0:
            self.stop_sign_cooldown -= 1

        if sign_action == "STOP" and self.stop_sign_cooldown == 0:
            if self.stop_sign_timer == 0:
                self.stop_sign_timer = 60
                self.stop_sign_cooldown = 120
                self.stop_signs_detected += 1
                print("STOP SIGN - stopping for 2 seconds")

        # ── speed / status decision ───────────────────────────────────────────
        if self._at_fork and self._fork_command is None:
            # Waiting at fork for dashboard input
            base_speed = 0.0
            status = "AT FORK — AWAITING CMD"
        elif self.stop_sign_timer > 0:
            base_speed = 0.0
            self.stop_sign_timer -= 1
            status = f"STOPPED (Sign: {self.stop_sign_timer})"
        else:
            base_speed = self.speed_control.calculate_speed(steering_error, obstacle_modifier)
            base_speed = min(base_speed, self.current_speed_limit)
            speed_category = self.speed_control.get_speed_category(abs(steering_error))
            status = speed_category.replace("_", " ")

        if self.autonomous_enabled and base_speed > 0:
            pid_output = self.steering.compute(steering_error)
            left_speed = max(0.0, min(1.0, base_speed + pid_output))
            right_speed = max(0.0, min(1.0, base_speed - pid_output))
        else:
            pid_output = 0.0
            left_speed = 0.0
            right_speed = 0.0

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
        frame = self._draw_motor_bars(frame, left_speed, right_speed, pid_output)
        speed_category = self.speed_control.get_speed_category(abs(error))
        target_speed = self.speed_control.target_speed
        frame = draw_speed_indicator(frame, base_speed, target_speed, speed_category)
        status_color = (0, 0, 255) if "STOPPED" in status else (255, 255, 255)
        cv2.putText(frame, f"Status: {status}",
                    (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        if sign_action == "STOP":
            cv2.putText(frame, "STOP DETECTED!",
                        (10, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
        else:
            cv2.putText(frame, "Scanning...",
                        (10, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
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
        left_fill = int(bar_height * left_speed)
        right_fill = int(bar_height * right_speed)
        bar_color = (0, 255, 0) if abs(pid_output) < 0.1 else (0, 165, 255)
        cv2.rectangle(frame, (bar_x_left, bar_y + bar_height - left_fill),
                      (bar_x_left + bar_width, bar_y + bar_height), bar_color, -1)
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

    def run(self):
        picam2 = Picamera2()
        config = picam2.create_preview_configuration(
            main={"format": "RGB888", "size": (640, 480)}
        )
        picam2.configure(config)
        picam2.start()
        self._picam2 = picam2   # make available to _execute_fork_turn
        time.sleep(1)
        print("Camera started! Resolution: 640x480")
        print("Controls: [Q] Quit  [SPACE] Toggle auto  [R] Reset  [S] Snapshot  [D] Debug signs")
        start_time = time.time()

        while True:
            frame = picam2.capture_array()
            debug_frame, (left, right, status) = self.process_frame(frame)

            # ── drive the real motors ─────────────────────────────────────────
            if self._at_fork and self._fork_command:
                # Execute the chosen route: spin until that color is centered
                self._execute_fork_turn(self._fork_command)
                # Reset fork state after turn completes
                self._at_fork       = False
                self._fork_command  = None
                self._fork_frame_count = 0
                self.color_detector.reset()
                self._notify_complete()
            elif self.autonomous_enabled:
                self._drive(left, right)
            else:
                self._stop_motors()

            window_title = "VectorVance - " + \
                           ("AUTONOMOUS" if self.autonomous_enabled else "MANUAL")
            cv2.imshow(window_title, debug_frame)

            if self.frame_count % 30 == 0:
                fps = self.frame_count / (time.time() - start_time)
                avg_error = self.total_error / self.frame_count
                print(f"Frame {self.frame_count:04d} | "
                      f"L:{left:.2f} R:{right:.2f} | "
                      f"{status:25s} | "
                      f"Dist:{self._last_distance:.0f}cm | "
                      f"FPS:{fps:.1f} | "
                      f"AvgErr:{avg_error:.1f}px")

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                self.autonomous_enabled = not self.autonomous_enabled
                print(f"Autonomous: {'ENABLED' if self.autonomous_enabled else 'DISABLED'}")
            elif key == ord('r'):
                self.steering.reset()
                self.speed_control.reset()
                self.safety.reset()
                self.perception.reset_smoothing()
                self.sign_detector.reset()
                self.current_speed_limit = 0.8
                self.stop_sign_timer = 0
                print("All systems reset")
            elif key == ord('s'):
                filename = f"/home/pi/test/snapshot_{self.frame_count:04d}.jpg"
                cv2.imwrite(filename, debug_frame)
                print(f"Snapshot saved: {filename}")
            elif key == ord('d'):
                print(f"Sign detector: {len(self.sign_detector.detected_signs)} raw, "
                      f"{len(self.sign_detector.confirmed_signs)} confirmed")
                for s in self.sign_detector.confirmed_signs:
                    print(f"  {s[0].value} at {s[1]} conf={s[2]:.2f}")

        picam2.stop()
        cv2.destroyAllWindows()
        self._stop_motors()
        self._cleanup_hardware()
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
        print("=" * 70)


def main():
    vehicle = AutonomousVehicle(max_speed=0.8)
    vehicle.run()


if __name__ == "__main__":
    main()