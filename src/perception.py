"""
perception.py - Lane Detection Module
--------------------------------------
Final refined version with all optimizations.
"""

import cv2
import numpy as np
from collections import deque


class LaneDetector:
    def __init__(self, width=640, height=480, smoothing_window=3):
        self.width = width
        self.height = height
        self.smoothing_window = smoothing_window
        self.error_history = deque(maxlen=smoothing_window)
        
        # ROI trapezoid
        roi_bottom = height
        roi_top = int(height * 0.55)
        roi_top_width_fraction = 0.3
        
        self.roi_vertices = np.array([[
            (0, roi_bottom),
            (int(width * (0.5 - roi_top_width_fraction/2)), roi_top),
            (int(width * (0.5 + roi_top_width_fraction/2)), roi_top),
            (width, roi_bottom)
        ]], dtype=np.int32)
        
        self.canny_low = 50
        self.canny_high = 150
        self.hough_threshold = 30
        self.hough_min_line_length = 30
        self.hough_max_line_gap = 150
    
    def process_frame(self, frame):
        frame = cv2.resize(frame, (self.width, self.height))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, self.canny_low, self.canny_high)
        masked_edges = self._apply_roi(edges)
        
        lines = cv2.HoughLinesP(
            masked_edges, rho=2, theta=np.pi/180,
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
        mask = np.zeros_like(edges)
        cv2.fillPoly(mask, self.roi_vertices, 255)
        return cv2.bitwise_and(edges, mask)
    
    def _get_lane_center(self, lines):
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
            if angle < 45:
                midpoint_x = (x1 + x2) // 2
                x_positions.append(midpoint_x)
        
        if len(x_positions) == 0:
            return None
        
        return int(np.median(x_positions))
    
    def _draw_overlay(self, frame, lines, lane_center, error, edges):
        debug = frame.copy()
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                dx = abs(x2 - x1)
                dy = abs(y2 - y1)
                if dy > 0:
                    angle = np.degrees(np.arctan(dx / dy))
                    if angle < 45:
                        cv2.line(debug, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        if lane_center is not None:
            cv2.line(debug, (lane_center, 0), (lane_center, self.height), (0, 0, 255), 3)
            cv2.putText(debug, f"Lane: {lane_center}px", 
                       (lane_center + 10, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        frame_center = self.width // 2
        cv2.line(debug, (frame_center, 0), (frame_center, self.height), (255, 0, 0), 3)
        cv2.putText(debug, "Target", (frame_center + 10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        cv2.polylines(debug, self.roi_vertices, True, (0, 255, 255), 2)
        
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
        
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        edges_small = cv2.resize(edges_colored, (160, 120))
        debug[10:130, 10:170] = edges_small
        cv2.putText(debug, "Edge View", (15, 145), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return debug
    
    def reset_smoothing(self):
        self.error_history.clear()