"""
main.py - Simulation entry point (no hardware).

Lane detection + PID steering + adaptive speed + STOP sign detection.
Obstacle detection is disabled here; enable it if needed for testing.
"""

import cv2
import time
from perception import LaneDetector
from controller import PIDController
from speed_controller import AdaptiveSpeedController, draw_speed_indicator
from safety import ObstacleDetector
from sign_detector import TrafficSignDetector


class AutonomousVehicle:
    def __init__(self, max_speed=0.8):
        self.perception = LaneDetector()
        self.steering = PIDController(Kp=0.003, Ki=0.0001, Kd=0.001)
        self.speed_control = AdaptiveSpeedController(min_speed=0.2, max_speed=max_speed)

        # kept alive so draw_overlay still works, but simulate_sensors is never called
        self.safety = ObstacleDetector(emergency_distance=20, warning_distance=50)

        self.sign_detector = TrafficSignDetector()

        self.autonomous_enabled = True
        self.current_speed_limit = max_speed
        self.stop_sign_timer = 0
        self.stop_sign_cooldown = 0

        self.frame_count = 0
        self.total_error = 0
        self.stop_signs_detected = 0

    def process_frame(self, frame):
        self.frame_count += 1

        steering_error, vision_frame = self.perception.process_frame(frame)
        self.total_error += abs(steering_error)

        # obstacle detection disabled in sim - always full speed modifier
        obstacle_modifier = 1.0

        detected_signs = self.sign_detector.detect_signs(frame)
        sign_action, _ = self.sign_detector.get_action()

        if self.stop_sign_cooldown > 0:
            self.stop_sign_cooldown -= 1

        if sign_action == "STOP" and self.stop_sign_cooldown == 0:
            if self.stop_sign_timer == 0:
                self.stop_sign_timer = 60      # ~2 seconds at 30fps
                self.stop_sign_cooldown = 120  # don't retrigger for 4 seconds
                self.stop_signs_detected += 1
                print("STOP SIGN - stopping for 2 seconds")

        if self.stop_sign_timer > 0:
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
        h, w = frame.shape[:2]

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

    def run(self, video_source):
        cap = cv2.VideoCapture(video_source)

        if not cap.isOpened():
            print(f"Cannot open video: {video_source}")
            return

        print("Controls: [Q] Quit  [SPACE] Toggle auto  [R] Reset  [S] Snapshot  [D] Debug signs")

        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of video")
                break

            debug_frame, (left, right, status) = self.process_frame(frame)

            window_title = "AUTONOMOUS VEHICLE - " + \
                          ("ENABLED" if self.autonomous_enabled else "DISABLED")
            cv2.imshow(window_title, debug_frame)

            if self.frame_count % 30 == 0:
                fps = self.frame_count / (time.time() - start_time)
                avg_error = self.total_error / self.frame_count
                print(f"Frame {self.frame_count:04d} | "
                      f"L:{left:.2f} R:{right:.2f} | "
                      f"{status:25s} | "
                      f"FPS:{fps:.1f} | "
                      f"AvgErr:{avg_error:.1f}px")

            key = cv2.waitKey(30) & 0xFF

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
                filename = f"../outputs/autonomous_{self.frame_count:04d}.jpg"
                cv2.imwrite(filename, debug_frame)
                print(f"Saved: {filename}")
            elif key == ord('d'):
                print(f"Sign detector: {len(self.sign_detector.detected_signs)} raw, "
                      f"{len(self.sign_detector.confirmed_signs)} confirmed")
                for s in self.sign_detector.confirmed_signs:
                    print(f"  {s[0].value} at {s[1]} conf={s[2]:.2f}")

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
        print("=" * 70)


def main():
    vehicle = AutonomousVehicle(max_speed=0.8)
    vehicle.run("../test_videos/challenge.mp4")


if __name__ == "__main__":
    main()
