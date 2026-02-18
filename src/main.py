"""
main.py - Complete Autonomous Vehicle System
---------------------------------------------
FINAL REFINED VERSION with all bugs fixed.

Video source: test2.mp4 (hardcoded)
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
        print("\n" + "=" * 70)
        print("🚗 AUTONOMOUS VEHICLE SYSTEM")
        print("=" * 70)
        
        self.perception = LaneDetector()
        self.steering = PIDController(Kp=0.003, Ki=0.0001, Kd=0.001)
        self.speed_control = AdaptiveSpeedController(min_speed=0.2, max_speed=max_speed)
        self.safety = ObstacleDetector(emergency_distance=20, warning_distance=50)
        self.sign_detector = TrafficSignDetector()
        
        self.autonomous_enabled = True
        self.current_speed_limit = max_speed
        self.stop_sign_timer = 0
        
        self.frame_count = 0
        self.total_error = 0
        
        print("\n✅ All systems initialized!")
        print("=" * 70)
    
    def process_frame(self, frame):
        self.frame_count += 1
        
        # STEP 1: Perception
        steering_error, vision_frame = self.perception.process_frame(frame)
        self.total_error += abs(steering_error)
        
        # STEP 2: Safety
        self.safety.simulate_sensors(frame, steering_error)
        obstacle_modifier = self.safety.get_speed_modifier()
        
        # STEP 3: Traffic Signs
        detected_signs = self.sign_detector.detect_signs(frame)
        sign_action, sign_value = self.sign_detector.get_action()
        
        if sign_action == "STOP":
            if self.stop_sign_timer == 0:
                self.stop_sign_timer = 60
        elif sign_action == "LIMIT" and sign_value:
            if self.current_speed_limit != sign_value:
                self.current_speed_limit = sign_value
        elif sign_action == "SLOW":
            obstacle_modifier = min(obstacle_modifier, 0.4)
        
        # STEP 4: Speed Control
        if self.stop_sign_timer > 0:
            base_speed = 0.0
            self.stop_sign_timer -= 1
            status = "STOPPED (Stop Sign)"
        elif self.safety.should_emergency_stop():
            base_speed = 0.0
            status = "EMERGENCY STOP (Obstacle)"
        else:
            base_speed = self.speed_control.calculate_speed(steering_error, obstacle_modifier)
            base_speed = min(base_speed, self.current_speed_limit)
            
            if self.safety.should_slow_down():
                status = "SLOWING (Obstacle Warning)"
            else:
                speed_category = self.speed_control.get_speed_category(abs(steering_error))
                status = speed_category.replace("_", " ")
        
        # STEP 5: Steering
        if self.autonomous_enabled and base_speed > 0:
            pid_output = self.steering.compute(steering_error)
            left_speed = max(0.0, min(1.0, base_speed + pid_output))
            right_speed = max(0.0, min(1.0, base_speed - pid_output))
        else:
            pid_output = 0.0
            left_speed = 0.0
            right_speed = 0.0
        
        # STEP 6: Visualization
        debug_frame = self._create_debug_frame(
            vision_frame, steering_error, pid_output,
            base_speed, left_speed, right_speed, status
        )
        
        motor_commands = (left_speed, right_speed, status)
        return debug_frame, motor_commands
    
    def _create_debug_frame(self, vision_frame, error, pid_output,
                           base_speed, left_speed, right_speed, status):
        frame = vision_frame.copy()
        h, w = frame.shape[:2]
        
        frame = self.safety.draw_overlay(frame)
        frame = self.sign_detector.draw_overlay(frame)
        frame = self._draw_motor_bars(frame, left_speed, right_speed, pid_output)
        
        speed_category = self.speed_control.get_speed_category(abs(error))
        target_speed = self.speed_control.target_speed
        frame = draw_speed_indicator(frame, base_speed, target_speed, speed_category)
        
        cv2.putText(frame, f"Status: {status}", 
                   (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.putText(frame, f"Limit: {self.current_speed_limit*100:.0f}%", 
                   (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
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
    
    def run(self, video_source):
        cap = cv2.VideoCapture(video_source)
        
        if not cap.isOpened():
            print(f"❌ Cannot open video: {video_source}")
            print("📌 Make sure test2.mp4 is in test_videos folder!")
            return
        
        print("\n🎬 AUTONOMOUS SYSTEM ACTIVE")
        print("=" * 70)
        print("CONTROLS:")
        print("  [Q] Quit")
        print("  [SPACE] Toggle Autonomous Mode")
        print("  [R] Reset all systems")
        print("  [S] Save snapshot")
        print("=" * 70)
        
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("\n📹 End of video")
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
                print("\n👋 Shutting down...")
                break
            elif key == ord(' '):
                self.autonomous_enabled = not self.autonomous_enabled
                mode = "ENABLED" if self.autonomous_enabled else "DISABLED"
                print(f"\n🔄 Autonomous mode: {mode}")
            elif key == ord('r'):
                self.steering.reset()
                self.speed_control.reset()
                self.safety.reset()
                self.perception.reset_smoothing()
                print("\n🔄 All systems reset")
            elif key == ord('s'):
                filename = f"../outputs/autonomous_{self.frame_count:04d}.jpg"
                cv2.imwrite(filename, debug_frame)
                print(f"\n💾 Saved: {filename}")
        
        cap.release()
        cv2.destroyAllWindows()
        self._print_statistics(start_time)
    
    def _print_statistics(self, start_time):
        duration = time.time() - start_time
        avg_error = self.total_error / max(self.frame_count, 1)
        fps = self.frame_count / max(duration, 0.1)
        
        print("\n" + "=" * 70)
        print("📊 SESSION SUMMARY")
        print("=" * 70)
        print(f"Duration:        {duration:.1f}s")
        print(f"Frames:          {self.frame_count}")
        print(f"Average FPS:     {fps:.1f}")
        print(f"Average error:   {avg_error:.1f}px")
        print("=" * 70)
        print("\n✅ System shutdown complete\n")


def main():
    vehicle = AutonomousVehicle(max_speed=0.8)
    
    # HARDCODED VIDEO SOURCE - test2.mp4
    video_source = "../test_videos/test2.mp4"
    
    vehicle.run(video_source)


if __name__ == "__main__":
    main()