"""
integrated_test.py - Basic Perception + PID Test
-------------------------------------------------
Standalone test for lane detection + PID control.
Use this for simple testing without all advanced features.
"""

import cv2
import time
from perception import LaneDetector
from controller import PIDController


def run_basic_test():
    detector = LaneDetector()
    pid = PIDController(Kp=0.003, Ki=0.0001, Kd=0.001)
    
    # Video source - CHANGE THIS to your video file
    cap = cv2.VideoCapture("../test_videos/test2.mp4")
    
    if not cap.isOpened():
        print("❌ Cannot open video")
        return
    
    print("🚗 Basic Autonomous Test Running")
    print("Press 'Q' to quit")
    
    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        error, debug_frame = detector.process_frame(frame)
        pid_output = pid.compute(error)
        
        base_speed = 0.6
        left_speed = max(0.0, min(1.0, base_speed + pid_output))
        right_speed = max(0.0, min(1.0, base_speed - pid_output))
        
        cv2.putText(debug_frame, f"PID: {pid_output:+.3f}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(debug_frame, f"L:{left_speed:.2f} R:{right_speed:.2f}", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        frame_count += 1
        
        if frame_count % 30 == 0:
            fps = frame_count / (time.time() - start_time)
            print(f"Frame {frame_count:04d} | Error: {error:+4d}px | "
                  f"PID: {pid_output:+.3f} | L:{left_speed:.2f} R:{right_speed:.2f} | FPS:{fps:.1f}")
        
        cv2.imshow("Basic Test - Press Q to Quit", debug_frame)
        
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"✅ Processed {frame_count} frames")


if __name__ == "__main__":
    run_basic_test()