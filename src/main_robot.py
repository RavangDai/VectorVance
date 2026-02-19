"""
main_robot.py - Complete Autonomous Robot (HARDWARE VERSION)
------------------------------------------------------------
Full-featured autonomous vehicle for Raspberry Pi deployment.

FEATURES ENABLED:
✅ Lane detection (camera)
✅ PID steering control
✅ Adaptive speed control
✅ Obstacle detection (HC-SR04 sensors)
✅ Traffic sign recognition (hand-drawn signs)
✅ Cruise control
✅ Emergency stop
✅ Physical motor control (GPIO)

Author: Autonomous Vehicle Team
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
        """
        Initialize complete autonomous robot.
        
        Args:
            max_speed: Maximum allowed speed (0-1)
            cruise_speed: Default cruise control speed (0-1)
        """
        print("\n" + "=" * 70)
        print("🤖 AUTONOMOUS ROBOT - HARDWARE MODE")
        print("=" * 70)
        
        # Initialize vision
        self.perception = LaneDetector()
        
        # Initialize control
        self.steering = PIDController(Kp=0.003, Ki=0.0001, Kd=0.001)
        self.speed_control = AdaptiveSpeedController(min_speed=0.2, max_speed=max_speed)
        
        # Initialize hardware
        self.motors = MotorController(max_speed=80)  # 80% max PWM for safety
        self.safety = HardwareObstacleDetector(emergency_distance=20, warning_distance=50)
        
        # Initialize sign detector
        self.sign_detector = TrafficSignDetector()
        
        # Robot state
        self.autonomous_enabled = False  # Start in manual mode for safety
        self.cruise_control_enabled = False
        self.cruise_speed = cruise_speed
        self.current_speed_limit = max_speed
        self.stop_sign_timer = 0
        self.last_stop_sign_frame = -999  # Prevent re-triggering
        
        # Statistics
        self.frame_count = 0
        self.total_error = 0
        self.emergency_stops = 0
        
        print("\n✅ All systems initialized!")
        print("⚠️  Robot starts in MANUAL mode - Press SPACE to enable autonomous")
        print("=" * 70)
    
    def process_frame(self, frame):
        """Main control loop - processes one camera frame."""
        self.frame_count += 1
        
        # STEP 1: PERCEPTION - Where is the lane?
        steering_error, vision_frame = self.perception.process_frame(frame)
        self.total_error += abs(steering_error)
        
        # STEP 2: SAFETY - Are there obstacles?
        obstacle_modifier = self.safety.get_speed_modifier()
        distances = self.safety.get_distances()
        
        # STEP 3: TRAFFIC SIGNS - What are the rules?
        detected_signs = self.sign_detector.detect_signs(frame)
        sign_action, sign_value = self.sign_detector.get_action()
        
        # Handle stop signs (one-shot detection)
        if sign_action == "STOP":
            if self.frame_count - self.last_stop_sign_frame > 180:  # 9 seconds cooldown
                self.stop_sign_timer = 60  # Stop for 3 seconds
                self.last_stop_sign_frame = self.frame_count
                print("🛑 STOP SIGN DETECTED - Stopping for 3 seconds")
        
        # Handle speed limits
        elif sign_action == "LIMIT" and sign_value:
            if self.current_speed_limit != sign_value:
                self.current_speed_limit = sign_value
                print(f"🚦 Speed limit: {sign_value*100:.0f}%")
        
        # Handle yield signs
        elif sign_action == "SLOW":
            obstacle_modifier = min(obstacle_modifier, 0.4)
        
        # STEP 4: SPEED CONTROL - How fast should we go?
        if self.stop_sign_timer > 0:
            # Stopped at stop sign
            base_speed = 0.0
            self.stop_sign_timer -= 1
            status = "STOPPED (Stop Sign)"
        
        elif self.safety.should_emergency_stop():
            # EMERGENCY STOP
            base_speed = 0.0
            status = "🚨 EMERGENCY STOP (Obstacle)"
            self.emergency_stops += 1
        
        elif self.cruise_control_enabled:
            # CRUISE CONTROL - maintain set speed
            base_speed = self.cruise_speed
            base_speed *= obstacle_modifier  # Still respect obstacles
            base_speed = min(base_speed, self.current_speed_limit)
            status = f"CRUISE {self.cruise_speed*100:.0f}%"
        
        else:
            # ADAPTIVE SPEED - adjust based on curves
            base_speed = self.speed_control.calculate_speed(steering_error, obstacle_modifier)
            base_speed = min(base_speed, self.current_speed_limit)
            
            if self.safety.should_slow_down():
                status = "⚠️  SLOWING (Obstacle)"
            else:
                speed_category = self.speed_control.get_speed_category(abs(steering_error))
                status = speed_category.replace("_", " ")
        
        # STEP 5: STEERING - Calculate motor commands
        if self.autonomous_enabled and base_speed > 0:
            pid_output = self.steering.compute(steering_error)
            left_speed = max(0.0, min(1.0, base_speed + pid_output))
            right_speed = max(0.0, min(1.0, base_speed - pid_output))
        else:
            pid_output = 0.0
            left_speed = 0.0
            right_speed = 0.0
        
        # STEP 6: ACTUATE MOTORS
        if self.autonomous_enabled:
            self.motors.set_speeds(left_speed, right_speed)
        else:
            self.motors.emergency_stop()
        
        # STEP 7: VISUALIZATION
        debug_frame = self._create_debug_frame(
            vision_frame, steering_error, pid_output,
            base_speed, left_speed, right_speed, status, distances
        )
        
        return debug_frame, (left_speed, right_speed, status)
    
    def _create_debug_frame(self, vision_frame, error, pid_output,
                           base_speed, left_speed, right_speed, status, distances):
        """Create comprehensive debug visualization."""
        import cv2
        frame = vision_frame.copy()
        h, w = frame.shape[:2]
        
        # Draw obstacle distances
        self._draw_obstacle_overlay(frame, distances)
        
        # Draw detected signs
        frame = self.sign_detector.draw_overlay(frame)
        
        # Draw motor bars
        frame = self._draw_motor_bars(frame, left_speed, right_speed, pid_output)
        
        # Draw speed indicator
        speed_category = self.speed_control.get_speed_category(abs(error))
        target_speed = self.speed_control.target_speed if not self.cruise_control_enabled else self.cruise_speed
        frame = draw_speed_indicator(frame, base_speed, target_speed, speed_category)
        
        # Status text
        status_color = (0, 255, 0) if self.autonomous_enabled else (0, 0, 255)
        cv2.putText(frame, f"Status: {status}", 
                   (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
        # Mode indicators
        mode_text = "AUTO" if self.autonomous_enabled else "MANUAL"
        mode_color = (0, 255, 0) if self.autonomous_enabled else (0, 0, 255)
        cv2.putText(frame, mode_text, 
                   (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, mode_color, 2)
        
        if self.cruise_control_enabled:
            cv2.putText(frame, "CRUISE", 
                       (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        return frame
    
    def _draw_obstacle_overlay(self, frame, distances):
        """Draw ultrasonic sensor distances."""
        h, w = frame.shape[:2]
        center_x = w // 2
        center_y = int(h * 0.85)
        
        # Left sensor
        left_color = self._get_distance_color(distances['left'])
        cv2.circle(frame, (center_x - 60, center_y), 12, left_color, -1)
        cv2.putText(frame, f"{int(distances['left'])}", 
                   (center_x - 80, center_y - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, left_color, 1)
        
        # Right sensor
        right_color = self._get_distance_color(distances['right'])
        cv2.circle(frame, (center_x + 60, center_y), 12, right_color, -1)
        cv2.putText(frame, f"{int(distances['right'])}", 
                   (center_x + 65, center_y - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, right_color, 1)
        
        # Emergency warning
        if self.safety.should_emergency_stop():
            cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), 10)
            cv2.putText(frame, "!!! EMERGENCY STOP !!!", 
                       (w//2 - 180, h//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    
    def _get_distance_color(self, distance):
        """Color code by distance."""
        if distance < 20:
            return (0, 0, 255)  # Red
        elif distance < 50:
            return (0, 165, 255)  # Orange
        else:
            return (0, 255, 0)  # Green
    
    def _draw_motor_bars(self, frame, left_speed, right_speed, pid_output):
        """Draw motor speed bars."""
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
        
        left_color = (0, 255, 0) if abs(pid_output) < 0.1 else (0, 165, 255)
        right_color = (0, 255, 0) if abs(pid_output) < 0.1 else (0, 165, 255)
        
        cv2.rectangle(frame, (bar_x_left, bar_y + bar_height - left_fill), 
                     (bar_x_left + bar_width, bar_y + bar_height), left_color, -1)
        cv2.rectangle(frame, (bar_x_right, bar_y + bar_height - right_fill), 
                     (bar_x_right + bar_width, bar_y + bar_height), right_color, -1)
        
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
        """Main robot control loop."""
        # Initialize Pi Camera
        print("\n📷 Initializing Pi Camera...")
        picam = Picamera2()
        config = picam.create_preview_configuration(
            main={"size": (640, 480), "format": "RGB888"}
        )
        picam.configure(config)
        picam.start()
        time.sleep(2)  # Camera warmup
        
        print("\n🎬 ROBOT ACTIVE")
        print("=" * 70)
        print("CONTROLS:")
        print("  [SPACE] Toggle Autonomous Mode (CRITICAL!)")
        print("  [C] Toggle Cruise Control")
        print("  [+/-] Adjust Cruise Speed")
        print("  [E] Emergency Stop")
        print("  [R] Reset Systems")
        print("  [Q] Quit")
        print("=" * 70)
        print("\n⚠️  SAFETY: Robot starts in MANUAL mode")
        print("Press SPACE when ready to enable autonomous driving\n")
        
        start_time = time.time()
        
        try:
            while True:
                # Capture frame
                frame = picam.capture_array()
                
                # Process frame
                debug_frame, (left, right, status) = self.process_frame(frame)
                
                # Display (if monitor connected)
                try:
                    cv2.imshow("Autonomous Robot", debug_frame)
                except:
                    pass  # Headless mode
                
                # Print telemetry
                if self.frame_count % 30 == 0:
                    fps = self.frame_count / (time.time() - start_time)
                    avg_error = self.total_error / self.frame_count
                    
                    print(f"Frame {self.frame_count:04d} | "
                          f"{'AUTO' if self.autonomous_enabled else 'MANUAL'} | "
                          f"L:{left:.2f} R:{right:.2f} | "
                          f"{status:25s} | "
                          f"FPS:{fps:.1f} | "
                          f"AvgErr:{avg_error:.1f}px")
                
                # Handle keyboard
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord(' '):
                    # TOGGLE AUTONOMOUS MODE
                    self.autonomous_enabled = not self.autonomous_enabled
                    mode = "ENABLED ✅" if self.autonomous_enabled else "DISABLED ❌"
                    print(f"\n🚨 AUTONOMOUS MODE: {mode}\n")
                    if not self.autonomous_enabled:
                        self.motors.emergency_stop()
                
                elif key == ord('c'):
                    # TOGGLE CRUISE CONTROL
                    self.cruise_control_enabled = not self.cruise_control_enabled
                    status = "ON" if self.cruise_control_enabled else "OFF"
                    print(f"\n🚗 Cruise Control: {status} ({self.cruise_speed*100:.0f}%)\n")
                
                elif key == ord('=') or key == ord('+'):
                    # INCREASE CRUISE SPEED
                    self.cruise_speed = min(0.8, self.cruise_speed + 0.1)
                    print(f"🔼 Cruise speed: {self.cruise_speed*100:.0f}%")
                
                elif key == ord('-'):
                    # DECREASE CRUISE SPEED
                    self.cruise_speed = max(0.2, self.cruise_speed - 0.1)
                    print(f"🔽 Cruise speed: {self.cruise_speed*100:.0f}%")
                
                elif key == ord('e'):
                    # EMERGENCY STOP
                    self.autonomous_enabled = False
                    self.motors.emergency_stop()
                    print("\n🚨 EMERGENCY STOP ACTIVATED\n")
                
                elif key == ord('r'):
                    # RESET SYSTEMS
                    self.steering.reset()
                    self.speed_control.reset()
                    self.perception.reset_smoothing()
                    print("\n🔄 All systems reset\n")
                
                elif key == ord('q'):
                    print("\n👋 Shutting down robot...")
                    break
        
        finally:
            # Clean shutdown
            self.motors.emergency_stop()
            self.safety.stop_monitoring()
            self.motors.cleanup()
            picam.stop()
            cv2.destroyAllWindows()
            
            self._print_statistics(start_time)
    
    def _print_statistics(self, start_time):
        """Print session statistics."""
        duration = time.time() - start_time
        avg_error = self.total_error / max(self.frame_count, 1)
        fps = self.frame_count / max(duration, 0.1)
        
        print("\n" + "=" * 70)
        print("📊 SESSION SUMMARY")
        print("=" * 70)
        print(f"Duration:          {duration:.1f}s")
        print(f"Frames:            {self.frame_count}")
        print(f"Average FPS:       {fps:.1f}")
        print(f"Average error:     {avg_error:.1f}px")
        print(f"Emergency stops:   {self.emergency_stops}")
        print("=" * 70)
        print("\n✅ Robot shutdown complete\n")


def main():
    """Main entry point."""
    robot = AutonomousRobot(max_speed=0.8, cruise_speed=0.6)
    robot.run()


if __name__ == "__main__":
    main()