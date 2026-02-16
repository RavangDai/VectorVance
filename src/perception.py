"""
perception.py - Lane Detection Module
Detects lane lines from camera/video and calculates steering error.
"""

import cv2
import numpy as np
from collections import deque


class LaneDetector:
    """Detects lane boundaries using computer vision."""
    
    def __init__(self, width=640, height=480, smoothing_window=3):
        self.width = width
        self.height = height
        
        # Smoothing to reduce jitter
        self.smoothing_window = smoothing_window
        self.error_history = deque(maxlen=smoothing_window)
        
        # ROI trapezoid (narrower at top to match road perspective)
        roi_bottom = height
        roi_top = int(height * 0.55)
        roi_top_width_fraction = 0.3
        
        self.roi_vertices = np.array([[
            (0, roi_bottom),
            (int(width * (0.5 - roi_top_width_fraction/2)), roi_top),
            (int(width * (0.5 + roi_top_width_fraction/2)), roi_top),
            (width, roi_bottom)
        ]], dtype=np.int32)
        
        # Canny thresholds
        self.canny_low = 50
        self.canny_high = 150
        
        # Hough parameters
        self.hough_threshold = 30
        self.hough_min_line_length = 30
        self.hough_max_line_gap = 150
        
    def process_frame(self, frame):
        """Main pipeline: frame → steering error + debug visualization."""
        frame = cv2.resize(frame, (self.width, self.height))
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Adaptive histogram equalization for varying lighting
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, self.canny_low, self.canny_high)
        masked_edges = self._apply_roi(edges)
        
        lines = cv2.HoughLinesP(
            masked_edges,
            rho=2,
            theta=np.pi/180,
            threshold=self.hough_threshold,
            minLineLength=self.hough_min_line_length,
            maxLineGap=self.hough_max_line_gap
        )
        
        lane_center = self._get_lane_center(lines)
        frame_center = self.width // 2
        
        if lane_center is not None:
            raw_error = frame_center - lane_center
        else:
            raw_error = 0
        
        self.error_history.append(raw_error)
        steering_error = int(np.mean(self.error_history))
        
        debug_frame = self._draw_overlay(frame, lines, lane_center, steering_error, masked_edges)
        
        return steering_error, debug_frame
    
    def _apply_roi(self, edges):
        """Mask everything except road trapezoid."""
        mask = np.zeros_like(edges)
        cv2.fillPoly(mask, self.roi_vertices, 255)
        masked = cv2.bitwise_and(edges, mask)
        return masked
    
    def _get_lane_center(self, lines):
        """Calculate lane center, filtering out horizontal lines."""
        if lines is None or len(lines) == 0:
            return None
        
        x_positions = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            dx = abs(x2 - x1)
            dy = abs(y2 - y1)
            
            if dy == 0:
                continue
            
            angle = np.degrees(np.arctan(dx / dy))
            
            # Keep only lines < 45° from vertical
            if angle < 45:
                midpoint_x = (x1 + x2) // 2
                x_positions.append(midpoint_x)
        
        if len(x_positions) == 0:
            return None
        
        # Median is more robust to outliers
        lane_center = int(np.median(x_positions))
        return lane_center
    
    def _draw_overlay(self, frame, lines, lane_center, error, edges):
        """Draw debug visualization on frame."""
        debug = frame.copy()
        
        # Detected lines (green)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                dx = abs(x2 - x1)
                dy = abs(y2 - y1)
                if dy > 0:
                    angle = np.degrees(np.arctan(dx / dy))
                    if angle < 45:
                        cv2.line(debug, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        # Lane center (red)
        if lane_center is not None:
            cv2.line(debug, (lane_center, 0), 
                    (lane_center, self.height), (0, 0, 255), 3)
            cv2.putText(debug, f"Lane: {lane_center}px", 
                       (lane_center + 10, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Frame center (blue)
        frame_center = self.width // 2
        cv2.line(debug, (frame_center, 0), 
                (frame_center, self.height), (255, 0, 0), 3)
        cv2.putText(debug, "Target", 
                   (frame_center + 10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # ROI boundary (yellow)
        cv2.polylines(debug, self.roi_vertices, True, (0, 255, 255), 2)
        
        # Direction indicator
        if error > 10:
            direction = "LEFT"
            color = (0, 165, 255)
        elif error < -10:
            direction = "RIGHT"
            color = (255, 0, 255)
        else:
            direction = "STRAIGHT"
            color = (0, 255, 0)
        
        cv2.putText(debug, f"Error: {error:+4d}px -> {direction}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Edge preview (top-left corner)
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        edges_small = cv2.resize(edges_colored, (160, 120))
        debug[10:130, 10:170] = edges_small
        cv2.putText(debug, "Edge View", 
                   (15, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return debug
    
    def reset_smoothing(self):
        """Clear error history."""
        self.error_history.clear()


def test_lane_detection():
    """Test lane detector with video/webcam."""
    detector = LaneDetector()
    
    video_path = "../test_videos/road_sample.mp4"
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("⚠️  Video not found, trying webcam...")
        cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ No video source!")
        return
    
    print("🎥 Lane Detection Active")
    print("=" * 60)
    print("Controls: [Q]uit | [S]ave | [P]ause | [R]eset smoothing")
    print("=" * 60)
    
    frame_count = 0
    paused = False
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            
            error, debug_frame = detector.process_frame(frame)
            frame_count += 1
            
            if frame_count % 30 == 0:
                dir_symbol = "⬅️ " if error > 10 else "➡️ " if error < -10 else "⬆️ "
                print(f"Frame {frame_count:04d} | Error: {error:+4d}px | {dir_symbol}")
        
        cv2.imshow("Lane Detection - Press Q to Quit", debug_frame)
        
        key = cv2.waitKey(30 if not paused else 0) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f"../outputs/snapshot_{frame_count:04d}.jpg"
            cv2.imwrite(filename, debug_frame)
            print(f"💾 Saved: {filename}")
        elif key == ord('p'):
            paused = not paused
            print("⏸️  PAUSED" if paused else "▶️  RESUMED")
        elif key == ord('r'):
            detector.reset_smoothing()
            print("🔄 Smoothing reset")
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"✅ Processed {frame_count} frames")


if __name__ == "__main__":
    test_lane_detection()