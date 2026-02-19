"""
main_robot.py - Hardware entry point for Raspberry Pi deployment.

Runs the full stack: lane detection, PID steering, adaptive speed,
obstacle avoidance (HC-SR04), traffic sign recognition, and GPIO motor control.
Robot starts in manual mode for safety - press SPACE to enable autonomous driving.
"""

import cv2
import time
from picamera2 import Picamera2
from perception import LaneDetector
from controller import PIDController
from speed_controller import AdaptiveSpeedController, draw_speed_indicator
from sign_detector import TrafficSignDetector
from hardware import MotorController, HardwareObstacleDetector


class AutonomousRobot:
    def __init__(self, max_speed=0.8, cruise_speed=0.6):
        self.perception = LaneDetector()
        self.steering = PIDController(Kp=0.003, Ki=0.0001, Kd=0.001)
        self.speed_control = AdaptiveSpeedController(min_speed=0.2, max_speed=max_speed)
        self.motors = MotorController(max_speed=80)  # 80% PWM cap for safety
        self.safety = HardwareObstacleDetector(emergency_distance=20, warning_distance=50)
        self.sign_detector = TrafficSignDetector()

        # start in manual mode so the robot doesn't move until the operator is ready
        self.autonomous_enabled = False
        self.cruise_control_enabled = False
        self.cruise_speed = cruise_speed
        self.current_speed_limit = max_speed
        self.stop_sign_timer = 0
        self.last_stop_sign_frame = -999  # track frame of last stop to enforce cooldown

        self.frame_count = 0
        self.total_error = 0
        self.emergency_stops = 0

        print("Robot ready. Press SPACE to enable autonomous driving.")

    def process_frame(self, frame):
        self.frame_count += 1

        steering_error, vision_frame = self.perception.process_frame(frame)
        self.total_error += abs(steering_error)

        obstacle_modifier = self.safety.get_speed_modifier()
        distances = self.safety.get_distances()

        detected_signs = self.sign_detector.detect_signs(frame)
        sign_action, sign_value = self.sign_detector.get_action()

        if sign_action == "STOP":
            # 9-second cooldown (180 frames) to avoid stopping at the same sign twice
            if self.frame_count - self.last_stop_sign_frame > 180:
                self.stop_sign_timer = 60
                self.last_stop_sign_frame = self.frame_count
                print("STOP SIGN - stopping for 3 seconds")

        elif sign_action == "LIMIT" and sign_value:
            if self.current_speed_limit != sign_value:
                self.current_speed_limit = sign_value
                print(f"Speed limit: {sign_value*100:.0f}%")

        elif sign_action == "SLOW":
            # yield sign - cap speed modifier
            obstacle_modifier = min(obstacle_modifier, 0.4)

        if self.stop_sign_timer > 0:
            base_speed = 0.0
            self.stop_sign_timer -= 1
            status = "STOPPED (Stop Sign)"

        elif self.safety.should_emergency_stop():
            base_speed = 0.0
            status = "EMERGENCY STOP (Obstacle)"
            self.emergency_stops += 1

        elif self.cruise_control_enabled:
            base_speed = min(self.cruise_speed * obstacle_modifier, self.current_speed_limit)
            status = f"CRUISE {self.cruise_speed*100:.0f}%"

        else:
            base_speed = self.speed_control.calculate_speed(steering_error, obstacle_modifier)
            base_speed = min(base_speed, self.current_speed_limit)

            if self.safety.should_slow_down():
                status = "SLOWING (Obstacle)"
            else:
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

        if self.autonomous_enabled:
            self.motors.set_speeds(left_speed, right_speed)
        else:
            self.motors.emergency_stop()

        debug_frame = self._create_debug_frame(
            vision_frame, steering_error, pid_output,
            base_speed, left_speed, right_speed, status, distances
        )

        return debug_frame, (left_speed, right_speed, status)

    def _create_debug_frame(self, vision_frame, error, pid_output,
                           base_speed, left_speed, right_speed, status, distances):
        import cv2
        frame = vision_frame.copy()
        h, w = frame.shape[:2]

        self._draw_obstacle_overlay(frame, distances)
        frame = self.sign_detector.draw_overlay(frame)
        frame = self._draw_motor_bars(frame, left_speed, right_speed, pid_output)

        speed_category = self.speed_control.get_speed_category(abs(error))
        target_speed = self.speed_control.target_speed if not self.cruise_control_enabled else self.cruise_speed
        frame = draw_speed_indicator(frame, base_speed, target_speed, speed_category)

        status_color = (0, 255, 0) if self.autonomous_enabled else (0, 0, 255)
        cv2.putText(frame, f"Status: {status}",
                   (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

        mode_text = "AUTO" if self.autonomous_enabled else "MANUAL"
        cv2.putText(frame, mode_text,
                   (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

        if self.cruise_control_enabled:
            cv2.putText(frame, "CRUISE",
                       (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        return frame

    def _draw_obstacle_overlay(self, frame, distances):
        h, w = frame.shape[:2]
        center_x = w // 2
        center_y = int(h * 0.85)

        left_color = self._get_distance_color(distances['left'])
        cv2.circle(frame, (center_x - 60, center_y), 12, left_color, -1)
        cv2.putText(frame, f"{int(distances['left'])}",
                   (center_x - 80, center_y - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, left_color, 1)

        right_color = self._get_distance_color(distances['right'])
        cv2.circle(frame, (center_x + 60, center_y), 12, right_color, -1)
        cv2.putText(frame, f"{int(distances['right'])}",
                   (center_x + 65, center_y - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, right_color, 1)

        if self.safety.should_emergency_stop():
            cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), 10)
            cv2.putText(frame, "!!! EMERGENCY STOP !!!",
                       (w//2 - 180, h//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    def _get_distance_color(self, distance):
        if distance < 20:
            return (0, 0, 255)    # red
        elif distance < 50:
            return (0, 165, 255)  # orange
        else:
            return (0, 255, 0)    # green

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
        print("Initializing Pi Camera...")
        picam = Picamera2()
        config = picam.create_preview_configuration(
            main={"size": (640, 480), "format": "RGB888"}
        )
        picam.configure(config)
        picam.start()
        time.sleep(2)  # let the camera warm up

        print("Controls: [SPACE] Toggle auto  [C] Cruise  [+/-] Cruise speed  [E] E-stop  [R] Reset  [Q] Quit")

        start_time = time.time()

        try:
            while True:
                frame = picam.capture_array()
                debug_frame, (left, right, status) = self.process_frame(frame)

                try:
                    cv2.imshow("Autonomous Robot", debug_frame)
                except:
                    pass  # headless - no display attached

                if self.frame_count % 30 == 0:
                    fps = self.frame_count / (time.time() - start_time)
                    avg_error = self.total_error / self.frame_count
                    print(f"Frame {self.frame_count:04d} | "
                          f"{'AUTO' if self.autonomous_enabled else 'MANUAL'} | "
                          f"L:{left:.2f} R:{right:.2f} | "
                          f"{status:25s} | "
                          f"FPS:{fps:.1f} | "
                          f"AvgErr:{avg_error:.1f}px")

                key = cv2.waitKey(1) & 0xFF

                if key == ord(' '):
                    self.autonomous_enabled = not self.autonomous_enabled
                    print(f"Autonomous: {'ENABLED' if self.autonomous_enabled else 'DISABLED'}")
                    if not self.autonomous_enabled:
                        self.motors.emergency_stop()

                elif key == ord('c'):
                    self.cruise_control_enabled = not self.cruise_control_enabled
                    print(f"Cruise control: {'ON' if self.cruise_control_enabled else 'OFF'} "
                          f"({self.cruise_speed*100:.0f}%)")

                elif key == ord('=') or key == ord('+'):
                    self.cruise_speed = min(0.8, self.cruise_speed + 0.1)
                    print(f"Cruise speed: {self.cruise_speed*100:.0f}%")

                elif key == ord('-'):
                    self.cruise_speed = max(0.2, self.cruise_speed - 0.1)
                    print(f"Cruise speed: {self.cruise_speed*100:.0f}%")

                elif key == ord('e'):
                    self.autonomous_enabled = False
                    self.motors.emergency_stop()
                    print("EMERGENCY STOP")

                elif key == ord('r'):
                    self.steering.reset()
                    self.speed_control.reset()
                    self.perception.reset_smoothing()
                    print("All systems reset")

                elif key == ord('q'):
                    print("Shutting down...")
                    break

        finally:
            self.motors.emergency_stop()
            self.safety.stop_monitoring()
            self.motors.cleanup()
            picam.stop()
            cv2.destroyAllWindows()
            self._print_statistics(start_time)

    def _print_statistics(self, start_time):
        duration = time.time() - start_time
        avg_error = self.total_error / max(self.frame_count, 1)
        fps = self.frame_count / max(duration, 0.1)

        print("=" * 70)
        print(f"Duration:        {duration:.1f}s")
        print(f"Frames:          {self.frame_count}")
        print(f"Average FPS:     {fps:.1f}")
        print(f"Average error:   {avg_error:.1f}px")
        print(f"Emergency stops: {self.emergency_stops}")
        print("=" * 70)


def main():
    robot = AutonomousRobot(max_speed=0.8, cruise_speed=0.6)
    robot.run()


if __name__ == "__main__":
    main()
