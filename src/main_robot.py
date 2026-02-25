"""
main_robot.py - Hardware entry point for Raspberry Pi deployment.

Runs the full stack: lane detection, PID steering, adaptive speed,
obstacle avoidance (HC-SR04), STOP sign recognition, and GPIO motor control.
Robot starts in manual mode for safety - press SPACE to enable autonomous driving.

VectorVance Autonomous Vehicle Project
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
        print("\n" + "="*70)
        print("   🚗 VECTORVANCE AUTONOMOUS ROBOT")
        print("   Raspberry Pi Hardware Implementation")
        print("="*70)
        
        # ══════════════════════════════════════════════════════════════════
        # Initialize all subsystems
        # ══════════════════════════════════════════════════════════════════
        print("\n📦 Initializing subsystems...")
        
        self.perception = LaneDetector()
        self.steering = PIDController(Kp=0.003, Ki=0.0001, Kd=0.001)
        self.speed_control = AdaptiveSpeedController(min_speed=0.2, max_speed=max_speed)
        self.motors = MotorController(max_speed=80)  # 80% PWM cap for safety
        self.safety = HardwareObstacleDetector(emergency_distance=20, warning_distance=50)
        self.sign_detector = TrafficSignDetector()  # STOP signs only

        # ══════════════════════════════════════════════════════════════════
        # Control state
        # ══════════════════════════════════════════════════════════════════
        self.autonomous_enabled = False  # Start in manual mode for safety
        self.cruise_control_enabled = False
        self.cruise_speed = cruise_speed
        self.max_speed = max_speed
        
        # STOP sign handling
        self.stop_sign_timer = 0
        self.stop_sign_cooldown = 0
        
        # Statistics
        self.frame_count = 0
        self.total_error = 0
        self.emergency_stops = 0
        self.stop_signs_detected = 0

        print("\n" + "="*70)
        print("   ✅ INITIALIZATION COMPLETE")
        print("   • Motors:     READY (PWM cap: 80%)")
        print("   • Camera:     Pi Camera")
        print("   • Ultrasonic: 2x HC-SR04 (Left + Right)")
        print("   • Detection:  STOP signs only")
        print("="*70)
        print("\n⚠️  Robot in MANUAL mode - Press SPACE to enable autonomous driving")

    def process_frame(self, frame):
        """Process a single frame and control motors."""
        self.frame_count += 1

        # ══════════════════════════════════════════════════════════════════
        # STEP 1: Lane Detection
        # ══════════════════════════════════════════════════════════════════
        steering_error, vision_frame = self.perception.process_frame(frame)
        self.total_error += abs(steering_error)

        # ══════════════════════════════════════════════════════════════════
        # STEP 2: Obstacle Detection (Ultrasonic Sensors)
        # ══════════════════════════════════════════════════════════════════
        obstacle_modifier = self.safety.get_speed_modifier()
        distances = self.safety.get_distances()

        # ══════════════════════════════════════════════════════════════════
        # STEP 3: STOP Sign Detection
        # ══════════════════════════════════════════════════════════════════
        self.sign_detector.detect_signs(frame)
        sign_action, _ = self.sign_detector.get_action()
        
        # Handle cooldown
        if self.stop_sign_cooldown > 0:
            self.stop_sign_cooldown -= 1

        # STOP sign detected
        if sign_action == "STOP" and self.stop_sign_cooldown == 0:
            if self.stop_sign_timer == 0:
                self.stop_sign_timer = 90  # Stop for 3 seconds at 30fps
                self.stop_sign_cooldown = 180  # 6 second cooldown
                self.stop_signs_detected += 1
                print("🛑 STOP SIGN DETECTED - Stopping for 3 seconds")

        # ══════════════════════════════════════════════════════════════════
        # STEP 4: Determine Speed & Status
        # ══════════════════════════════════════════════════════════════════
        if self.stop_sign_timer > 0:
            base_speed = 0.0
            self.stop_sign_timer -= 1
            status = f"STOPPED (Sign: {self.stop_sign_timer//30 + 1}s)"

        elif self.safety.should_emergency_stop():
            base_speed = 0.0
            status = "🚨 EMERGENCY STOP"
            self.emergency_stops += 1

        elif self.cruise_control_enabled:
            base_speed = min(self.cruise_speed * obstacle_modifier, self.max_speed)
            status = f"CRUISE {self.cruise_speed*100:.0f}%"

        else:
            base_speed = self.speed_control.calculate_speed(steering_error, obstacle_modifier)
            base_speed = min(base_speed, self.max_speed)

            if self.safety.should_slow_down():
                status = "⚠️ SLOWING (Obstacle)"
            else:
                speed_category = self.speed_control.get_speed_category(abs(steering_error))
                status = speed_category.replace("_", " ")

        # ══════════════════════════════════════════════════════════════════
        # STEP 5: Calculate Motor Speeds (Differential Drive)
        # ══════════════════════════════════════════════════════════════════
        if self.autonomous_enabled and base_speed > 0:
            pid_output = self.steering.compute(steering_error)
            left_speed = max(0.0, min(1.0, base_speed + pid_output))
            right_speed = max(0.0, min(1.0, base_speed - pid_output))
        else:
            pid_output = 0.0
            left_speed = 0.0
            right_speed = 0.0

        # ══════════════════════════════════════════════════════════════════
        # STEP 6: Apply Motor Commands
        # ══════════════════════════════════════════════════════════════════
        if self.autonomous_enabled:
            self.motors.set_speeds(left_speed, right_speed)
        else:
            self.motors.emergency_stop()

        # ══════════════════════════════════════════════════════════════════
        # STEP 7: Create Debug Visualization
        # ══════════════════════════════════════════════════════════════════
        debug_frame = self._create_debug_frame(
            vision_frame, steering_error, pid_output,
            base_speed, left_speed, right_speed, status, distances
        )

        return debug_frame, (left_speed, right_speed, status)

    def _create_debug_frame(self, vision_frame, error, pid_output,
                           base_speed, left_speed, right_speed, status, distances):
        """Create visualization overlay."""
        frame = vision_frame.copy()
        h, w = frame.shape[:2]

        # Obstacle overlay (ultrasonic sensor distances)
        self._draw_obstacle_overlay(frame, distances)
        
        # STOP sign detection overlay
        frame = self.sign_detector.draw_overlay(frame)
        
        # Motor speed bars
        frame = self._draw_motor_bars(frame, left_speed, right_speed, pid_output)

        # Speed indicator
        speed_category = self.speed_control.get_speed_category(abs(error))
        target_speed = self.speed_control.target_speed if not self.cruise_control_enabled else self.cruise_speed
        frame = draw_speed_indicator(frame, base_speed, target_speed, speed_category)

        # Status text
        status_color = (0, 255, 0) if self.autonomous_enabled else (0, 0, 255)
        cv2.putText(frame, f"Status: {status}",
                   (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

        # Mode indicator
        mode_text = "🟢 AUTO" if self.autonomous_enabled else "🔴 MANUAL"
        cv2.putText(frame, mode_text,
                   (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        # Cruise control indicator
        if self.cruise_control_enabled:
            cv2.putText(frame, f"CRUISE {self.cruise_speed*100:.0f}%",
                       (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        # STOP sign count
        if self.stop_signs_detected > 0:
            cv2.putText(frame, f"STOP signs: {self.stop_signs_detected}",
                       (w - 150, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        return frame

    def _draw_obstacle_overlay(self, frame, distances):
        """Draw ultrasonic sensor distance indicators."""
        h, w = frame.shape[:2]
        center_x = w // 2
        center_y = int(h * 0.85)

        # Left sensor
        left_color = self._get_distance_color(distances['left'])
        cv2.circle(frame, (center_x - 60, center_y), 12, left_color, -1)
        cv2.putText(frame, f"{int(distances['left'])}cm",
                   (center_x - 85, center_y - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, left_color, 1)

        # Right sensor
        right_color = self._get_distance_color(distances['right'])
        cv2.circle(frame, (center_x + 60, center_y), 12, right_color, -1)
        cv2.putText(frame, f"{int(distances['right'])}cm",
                   (center_x + 45, center_y - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, right_color, 1)

        # Emergency stop overlay
        if self.safety.should_emergency_stop():
            cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), 10)
            cv2.putText(frame, "!!! EMERGENCY STOP !!!",
                       (w//2 - 180, h//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    def _get_distance_color(self, distance):
        """Color code based on distance."""
        if distance < 20:
            return (0, 0, 255)    # Red - danger
        elif distance < 50:
            return (0, 165, 255)  # Orange - warning
        else:
            return (0, 255, 0)    # Green - safe

    def _draw_motor_bars(self, frame, left_speed, right_speed, pid_output):
        """Draw motor speed visualization bars."""
        h, w = frame.shape[:2]

        bar_width = 40
        bar_height = 200
        bar_x_left = w - 120
        bar_x_right = w - 60
        bar_y = h - bar_height - 50

        # Background
        cv2.rectangle(frame, (bar_x_left, bar_y),
                     (bar_x_left + bar_width, bar_y + bar_height), (50, 50, 50), -1)
        cv2.rectangle(frame, (bar_x_right, bar_y),
                     (bar_x_right + bar_width, bar_y + bar_height), (50, 50, 50), -1)

        # Fill based on speed
        left_fill = int(bar_height * left_speed)
        right_fill = int(bar_height * right_speed)

        bar_color = (0, 255, 0) if abs(pid_output) < 0.1 else (0, 165, 255)

        cv2.rectangle(frame, (bar_x_left, bar_y + bar_height - left_fill),
                     (bar_x_left + bar_width, bar_y + bar_height), bar_color, -1)
        cv2.rectangle(frame, (bar_x_right, bar_y + bar_height - right_fill),
                     (bar_x_right + bar_width, bar_y + bar_height), bar_color, -1)

        # Labels
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
        """Main loop - run the autonomous robot."""
        print("\n📷 Initializing Pi Camera...")
        picam = Picamera2()
        config = picam.create_preview_configuration(
            main={"size": (640, 480), "format": "RGB888"}
        )
        picam.configure(config)
        picam.start()
        time.sleep(2)  # Camera warmup

        print("\n" + "="*70)
        print("   🎬 ROBOT RUNNING")
        print("="*70)
        print("CONTROLS:")
        print("   [SPACE] Toggle autonomous mode")
        print("   [C]     Toggle cruise control")
        print("   [+/-]   Adjust cruise speed")
        print("   [E]     Emergency stop")
        print("   [R]     Reset all systems")
        print("   [D]     Debug: print sign detector state")
        print("   [Q]     Quit")
        print("="*70 + "\n")

        start_time = time.time()

        try:
            while True:
                # Capture frame
                frame = picam.capture_array()
                
                # Process and get motor commands
                debug_frame, (left, right, status) = self.process_frame(frame)

                # Display (may fail if no monitor attached)
                try:
                    cv2.imshow("VectorVance Robot", debug_frame)
                except:
                    pass  # Headless mode

                # Periodic status print
                if self.frame_count % 30 == 0:
                    elapsed = time.time() - start_time
                    fps = self.frame_count / elapsed
                    avg_error = self.total_error / self.frame_count
                    mode = "AUTO" if self.autonomous_enabled else "MANUAL"
                    
                    print(f"Frame {self.frame_count:04d} | {mode:6s} | "
                          f"L:{left:.2f} R:{right:.2f} | "
                          f"{status:25s} | "
                          f"FPS:{fps:.1f} | Err:{avg_error:.1f}px")

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF

                if key == ord(' '):
                    self.autonomous_enabled = not self.autonomous_enabled
                    mode = "ENABLED" if self.autonomous_enabled else "DISABLED"
                    print(f"\n{'🟢' if self.autonomous_enabled else '🔴'} Autonomous: {mode}")
                    if not self.autonomous_enabled:
                        self.motors.emergency_stop()

                elif key == ord('c'):
                    self.cruise_control_enabled = not self.cruise_control_enabled
                    print(f"\n🚗 Cruise control: {'ON' if self.cruise_control_enabled else 'OFF'} "
                          f"({self.cruise_speed*100:.0f}%)")

                elif key == ord('=') or key == ord('+'):
                    self.cruise_speed = min(0.8, self.cruise_speed + 0.1)
                    print(f"\n⬆️ Cruise speed: {self.cruise_speed*100:.0f}%")

                elif key == ord('-'):
                    self.cruise_speed = max(0.2, self.cruise_speed - 0.1)
                    print(f"\n⬇️ Cruise speed: {self.cruise_speed*100:.0f}%")

                elif key == ord('e'):
                    self.autonomous_enabled = False
                    self.motors.emergency_stop()
                    print("\n🛑 EMERGENCY STOP ACTIVATED")

                elif key == ord('r'):
                    self.steering.reset()
                    self.speed_control.reset()
                    self.perception.reset_smoothing()
                    self.sign_detector.reset()
                    print("\n🔄 All systems reset")

                elif key == ord('d'):
                    print(f"\n🔍 SIGN DETECTOR DEBUG:")
                    print(f"   Raw detections: {len(self.sign_detector.detected_signs)}")
                    print(f"   Confirmed: {len(self.sign_detector.confirmed_signs)}")
                    for s in self.sign_detector.confirmed_signs:
                        print(f"      → {s[0]} at {s[1]} conf={s[2]:.2f}")

                elif key == ord('q'):
                    print("\n👋 Shutting down...")
                    break

        except KeyboardInterrupt:
            print("\n\n⚠️ Interrupted by user")

        finally:
            # Always cleanup
            self.motors.emergency_stop()
            self.safety.stop_monitoring()
            self.motors.cleanup()
            picam.stop()
            cv2.destroyAllWindows()
            self._print_statistics(start_time)

    def _print_statistics(self, start_time):
        """Print session summary."""
        duration = time.time() - start_time
        avg_error = self.total_error / max(self.frame_count, 1)
        fps = self.frame_count / max(duration, 0.1)

        print("\n" + "="*70)
        print("   📊 SESSION SUMMARY")
        print("="*70)
        print(f"   Duration:        {duration:.1f} seconds")
        print(f"   Frames:          {self.frame_count}")
        print(f"   Average FPS:     {fps:.1f}")
        print(f"   Average Error:   {avg_error:.1f} px")
        print(f"   Emergency Stops: {self.emergency_stops}")
        print(f"   STOP Signs:      {self.stop_signs_detected}")
        print("="*70 + "\n")


def main():
    robot = AutonomousRobot(max_speed=0.8, cruise_speed=0.6)
    robot.run()


if __name__ == "__main__":
    main()