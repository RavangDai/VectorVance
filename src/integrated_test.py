"""
integrated_test.py - Full System Integration
Combines perception + PID control for autonomous navigation.
"""

import cv2
import time
from perception import LaneDetector
from controller import PIDController


class AutonomousSystem:
    """Complete autonomous navigation: vision + PID + motor control."""
    
    def __init__(self, base_speed=0.6):
        self.detector = LaneDetector()
        
        self.pid = PIDController(
            Kp=0.003,
            Ki=0.0001,
            Kd=0.001
        )
        
        self.base_speed = base_speed
        
        self.frame_count = 0
        self.error_sum = 0
        self.max_error = 0
        
        self.log_data = []
        
        print("🚗 Autonomous System Initialized")
        print(f"   Base speed: {base_speed*100:.0f}%")
        print(f"   PID: Kp={0.003} Ki={0.0001} Kd={0.001}")
    
    def calculate_motor_speeds(self, pid_output):
        """Convert PID output to differential drive motor speeds."""
        left_speed = self.base_speed + pid_output
        right_speed = self.base_speed - pid_output
        
        left_speed = max(0.0, min(1.0, left_speed))
        right_speed = max(0.0, min(1.0, right_speed))
        
        return left_speed, right_speed
    
    def draw_motor_visualization(self, frame, left_speed, right_speed, pid_output):
        """Draw motor speed bars and PID info on frame."""
        h, w = frame.shape[:2]
        
        bar_width = 40
        bar_height = 200
        bar_x_left = w - 120
        bar_x_right = w - 60
        bar_y = h - bar_height - 20
        
        # Background bars
        cv2.rectangle(frame, (bar_x_left, bar_y), 
                     (bar_x_left + bar_width, bar_y + bar_height), 
                     (50, 50, 50), -1)
        cv2.rectangle(frame, (bar_x_right, bar_y), 
                     (bar_x_right + bar_width, bar_y + bar_height), 
                     (50, 50, 50), -1)
        
        left_fill_height = int(bar_height * left_speed)
        right_fill_height = int(bar_height * right_speed)
        
        # Green when balanced, orange when turning
        left_color = (0, 255, 0) if abs(pid_output) < 0.1 else (0, 165, 255)
        right_color = (0, 255, 0) if abs(pid_output) < 0.1 else (0, 165, 255)
        
        cv2.rectangle(frame, 
                     (bar_x_left, bar_y + bar_height - left_fill_height), 
                     (bar_x_left + bar_width, bar_y + bar_height), 
                     left_color, -1)
        cv2.rectangle(frame, 
                     (bar_x_right, bar_y + bar_height - right_fill_height), 
                     (bar_x_right + bar_width, bar_y + bar_height), 
                     right_color, -1)
        
        cv2.putText(frame, "L", (bar_x_left + 12, bar_y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, "R", (bar_x_right + 12, bar_y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.putText(frame, f"{left_speed:.2f}", (bar_x_left, bar_y + bar_height + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"{right_speed:.2f}", (bar_x_right, bar_y + bar_height + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        pid_text = f"PID: {pid_output:+.3f}"
        pid_color = (0, 255, 0) if abs(pid_output) < 0.2 else (0, 165, 255) if abs(pid_output) < 0.5 else (0, 0, 255)
        cv2.putText(frame, pid_text, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, pid_color, 2)
        
        state = self.pid.get_state()
        cv2.putText(frame, f"P:{state['P_term']:+.3f} I:{state['I_term']:+.3f} D:{state['D_term']:+.3f}", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return frame
    
    def run(self, video_source="../test_videos/test2.mp4"):
        """Main autonomous navigation loop."""
        cap = cv2.VideoCapture(video_source)
        
        if not cap.isOpened():
            print("❌ Cannot open video source")
            return
        
        print("\n🎬 Autonomous System Active")
        print("=" * 70)
        print("CONTROLS:")
        print("  [Q] Quit")
        print("  [R] Reset PID")
        print("  [P] Pause/Resume")
        print("  [S] Save snapshot")
        print("  [L] Toggle data logging")
        print("  [↑/↓] Increase/Decrease Kp")
        print("=" * 70)
        
        paused = False
        logging_enabled = False
        start_time = time.time()
        
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("📹 End of video")
                    break
                
                error, debug_frame = self.detector.process_frame(frame)
                pid_output = self.pid.compute(error)
                left_speed, right_speed = self.calculate_motor_speeds(pid_output)
                
                debug_frame = self.draw_motor_visualization(
                    debug_frame, left_speed, right_speed, pid_output
                )
                
                self.frame_count += 1
                self.error_sum += abs(error)
                self.max_error = max(self.max_error, abs(error))
                
                if logging_enabled:
                    self.log_data.append({
                        'frame': self.frame_count,
                        'error': error,
                        'pid_output': pid_output,
                        'left_speed': left_speed,
                        'right_speed': right_speed
                    })
                
                if self.frame_count % 30 == 0:
                    avg_error = self.error_sum / self.frame_count
                    fps = self.frame_count / (time.time() - start_time)
                    
                    direction = "⬅️ LEFT " if error > 10 else "➡️ RIGHT" if error < -10 else "⬆️ STRAIGHT"
                    
                    print(f"Frame {self.frame_count:04d} | Error: {error:+4d}px | "
                          f"PID: {pid_output:+.3f} | L:{left_speed:.2f} R:{right_speed:.2f} | "
                          f"{direction} | FPS: {fps:.1f}")
            
            cv2.imshow("Autonomous System - Perception + PID Control", debug_frame)
            
            key = cv2.waitKey(30 if not paused else 0) & 0xFF
            
            if key == ord('q'):
                print("\n👋 Shutting down...")
                break
            
            elif key == ord('r'):
                self.pid.reset()
                self.detector.reset_smoothing()
                print("🔄 System reset")
            
            elif key == ord('p'):
                paused = not paused
                status = "⏸️  PAUSED" if paused else "▶️  RESUMED"
                print(status)
            
            elif key == ord('s'):
                filename = f"../outputs/autonomous_frame_{self.frame_count:04d}.jpg"
                cv2.imwrite(filename, debug_frame)
                print(f"💾 Saved: {filename}")
            
            elif key == ord('l'):
                logging_enabled = not logging_enabled
                status = "ON" if logging_enabled else "OFF"
                print(f"📊 Data logging: {status}")
            
            elif key == 82:  # Up arrow
                new_kp = self.pid.Kp + 0.001
                self.pid.tune(Kp=new_kp)
            
            elif key == 84:  # Down arrow
                new_kp = max(0.001, self.pid.Kp - 0.001)
                self.pid.tune(Kp=new_kp)
        
        cap.release()
        cv2.destroyAllWindows()
        
        self._print_statistics(start_time)
        
        if logging_enabled and len(self.log_data) > 0:
            self._save_log()
    
    def _print_statistics(self, start_time):
        """Print session statistics."""
        duration = time.time() - start_time
        avg_error = self.error_sum / max(self.frame_count, 1)
        fps = self.frame_count / max(duration, 0.1)
        
        print("\n" + "=" * 70)
        print("📊 SESSION STATISTICS")
        print("=" * 70)
        print(f"Total frames:     {self.frame_count}")
        print(f"Duration:         {duration:.1f}s")
        print(f"Average FPS:      {fps:.1f}")
        print(f"Average error:    {avg_error:.1f}px")
        print(f"Max error:        {self.max_error:.0f}px")
        print("=" * 70)
    
    def _save_log(self):
        """Save logged data to CSV."""
        import csv
        
        filename = f"../outputs/autonomous_log_{int(time.time())}.csv"
        
        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['frame', 'error', 'pid_output', 
                                                     'left_speed', 'right_speed'])
            writer.writeheader()
            writer.writerows(self.log_data)
        
        print(f"📁 Log saved: {filename}")


def main():
    """Run autonomous system."""
    system = AutonomousSystem(base_speed=0.6)
    
    video_source = "../test_videos/test2.mp4"
    
    system.run(video_source)


if __name__ == "__main__":
    main()