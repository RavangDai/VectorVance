"""
perception.py - Lane Detection Module
--------------------------------------
Detects lane lines from camera/video and calculates steering error.

KEY CONCEPTS:
- Grayscale: Converts RGB to single channel (faster processing)
- Canny: Detects edges using gradient magnitude
- ROI: Region of Interest - ignores sky/sides, focuses on road
- Hough Transform: Detects straight lines from edge pixels
"""

import cv2
import numpy as np


class LaneDetector: 
    """
    Processes video frames to detect lane boundaries.
    
    Architecture:
    Input: BGR image (640x480)
    Output: steering_error (pixels from center)
    """
    
    def __init__(self, width=640, height=480):
        """
        Initialize detector with frame dimensions.
        
        Args:
            width: Frame width in pixels
            height: Frame height in pixels
        """
        self.width = width
        self.height = height
        
        # Define Region of Interest (trapezoid shape)
        # WHY TRAPEZOID? Camera perspective makes road appear wider at bottom
        self.roi_vertices = np.array([[
            (0, height),                    # Bottom-left corner
            (0, int(height * 0.6)),        # Top-left (60% up from bottom)
            (width, int(height * 0.6)),    # Top-right
            (width, height)                 # Bottom-right corner
        ]], dtype=np.int32)
        
    def process_frame(self, frame):
        """
        Main pipeline: Raw frame → Steering error
        
        PIPELINE STAGES:
        1. Resize (standardize input)
        2. Grayscale (reduce channels 3→1)
        3. Blur (reduce noise)
        4. Canny (detect edges)
        5. ROI mask (focus on road)
        6. Hough (find lines)
        7. Calculate center
        
        Args:
            frame: Input image (BGR format)
            
        Returns:
            steering_error: Pixels from center (-320 to +320)
            debug_frame: Annotated visualization
        """
        # Stage 1: Standardize size
        frame = cv2.resize(frame, (self.width, self.height))
        
        # Stage 2: Convert to grayscale
        # MATH: gray = 0.299*R + 0.587*G + 0.114*B
        # WHY: Human eye is more sensitive to green, less to blue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Stage 3: Gaussian blur
        # KERNEL (5,5) means we average each pixel with 5x5 neighbors
        # WHY: Reduces high-frequency noise before edge detection
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Stage 4: Canny edge detection
        # THEORY: Calculates gradient magnitude at each pixel
        # If gradient > high_threshold (150) → definite edge
        # If low < gradient < high (50-150) → edge if connected to strong edge
        edges = cv2.Canny(blur, 30, 100)
        
        # Stage 5: Apply Region of Interest mask
        masked_edges = self._apply_roi(edges)
        
        # Stage 6: Hough Line Transform
        # PARAMETERS:
        # rho=2: Lines detected with 2-pixel precision
        # theta=π/180: Angle precision of 1 degree
        # threshold=50: Need 50 collinear pixels to call it a line
        # minLineLength=40: Ignore short line segments
        # maxLineGap=100: Connect nearby line fragments
        lines = cv2.HoughLinesP(
            masked_edges,
            rho=2,
            theta=np.pi/180,
            threshold=50,
            minLineLength=40,
            maxLineGap=100
        )
        
        # Stage 7: Calculate steering error
        lane_center = self._get_lane_center(lines)
        frame_center = self.width // 2  # Integer division: 640//2 = 320
        
        if lane_center is not None:
            # Positive error → lane is LEFT of center → turn LEFT
            # Negative error → lane is RIGHT of center → turn RIGHT
            steering_error = frame_center - lane_center
        else:
            steering_error = 0  # No lanes detected → go straight
        
        # Create visualization for debugging
        debug_frame = self._draw_overlay(frame, lines, lane_center, steering_error)
        
        return steering_error, debug_frame
    
    def _apply_roi(self, edges):
        """
        Mask everything except road trapezoid.
        
        BITWISE OPERATION:
        mask: 255 inside trapezoid, 0 outside
        edges: edge pixels are 255, non-edges are 0
        result: edges AND mask → only edges inside trapezoid
        """
        mask = np.zeros_like(edges)
        cv2.fillPoly(mask, self.roi_vertices, 255)
        masked = cv2.bitwise_and(edges, mask)
        return masked
    
    def _get_lane_center(self, lines):
        """
        Calculate average X-coordinate of all detected lines.
        
        DERIVATION:
        For line segment from (x1,y1) to (x2,y2):
        - Midpoint X = (x1 + x2) / 2
        
        If we detect left lane and right lane:
        - Left midpoint: 100px
        - Right midpoint: 540px
        - Average: (100 + 540) / 2 = 320px (lane center!)
        
        Returns:
            Integer X-coordinate, or None if no lines detected
        """
        if lines is None or len(lines) == 0:
            return None
        
        x_positions = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            midpoint_x = (x1 + x2) // 2
            x_positions.append(midpoint_x)
        
        # Calculate mean (average)
        lane_center = int(np.mean(x_positions))
        return lane_center
    
    def _draw_overlay(self, frame, lines, lane_center, error):
        """
        Create annotated debug visualization.
        
        OVERLAY ELEMENTS:
        - Green lines: Detected lane boundaries
        - Red line: Calculated lane center
        - Blue line: Frame center (target)
        - Yellow polygon: ROI boundary
        - Text: Steering error value
        """
        debug = frame.copy()
        
        # Draw detected lines (green, thickness=2)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(debug, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw lane center (red vertical line)
        if lane_center is not None:
            cv2.line(debug, (lane_center, 0), 
                    (lane_center, self.height), (0, 0, 255), 2)
            
            # Add text label
            cv2.putText(debug, f"Lane: {lane_center}px", 
                       (lane_center + 10, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Draw frame center (blue vertical line)
        frame_center = self.width // 2
        cv2.line(debug, (frame_center, 0), 
                (frame_center, self.height), (255, 0, 0), 2)
        
        # Draw ROI boundary (yellow)
        cv2.polylines(debug, self.roi_vertices, True, (0, 255, 255), 2)
        
        # Add steering error text (top-left)
        direction = "STRAIGHT"
        if error > 5:
            direction = "LEFT"
        elif error < -5:
            direction = "RIGHT"
        
        cv2.putText(debug, f"Error: {error}px -> Turn {direction}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return debug


# ============================================
# TEST FUNCTION - Run this module standalone
# ============================================

def test_lane_detection():
    """
    Test the lane detector with video file or webcam.
    
    CONTROLS:
    - Press 'q' to quit
    - Press 's' to save screenshot
    - Press 'p' to pause/resume
    """
    detector = LaneDetector()
    
    # Try video file first, fallback to webcam
    video_path = "../test_videos/road_sample.mp4"
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("⚠️ Video file not found, trying webcam...")
        cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ No video source available!")
        print("📌 Please add a video to: test_videos/road_sample.mp4")
        return
    
    print("🎥 Lane Detection Started!")
    print("Controls: [Q]uit | [S]ave frame | [P]ause")
    print("-" * 50)
    
    frame_count = 0
    paused = False
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("📹 End of video")
                break
            
            # Process frame
            error, debug_frame = detector.process_frame(frame)
            frame_count += 1
            
            # Print stats every 30 frames (reduce spam)
            if frame_count % 30 == 0:
                print(f"Frame {frame_count:04d} | Error: {error:+4d}px")
        
        # Display result
        cv2.imshow("Lane Detection - Press Q to quit", debug_frame)
        
        # Handle keyboard input
        key = cv2.waitKey(30 if not paused else 0) & 0xFF
        
        if key == ord('q'):
            print("👋 Exiting...")
            break
        elif key == ord('s'):
            filename = f"../outputs/snapshot_{frame_count:04d}.jpg"
            cv2.imwrite(filename, debug_frame)
            print(f"💾 Saved: {filename}")
        elif key == ord('p'):
            paused = not paused
            print("⏸️ PAUSED" if paused else "▶️ RESUMED")
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"✅ Processed {frame_count} total frames")


# Entry point
if __name__ == "__main__":
    test_lane_detection()