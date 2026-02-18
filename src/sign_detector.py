"""
sign_detector.py - Traffic Sign Recognition
--------------------------------------------
Final refined version.
"""

import cv2
import numpy as np
from enum import Enum


class SignType(Enum):
    STOP = "STOP"
    SPEED_LIMIT_30 = "SPEED_30"
    SPEED_LIMIT_50 = "SPEED_50"
    YIELD = "YIELD"
    UNKNOWN = "UNKNOWN"


class TrafficSignDetector:
    def __init__(self):
        self.red_lower1 = np.array([0, 120, 70])
        self.red_upper1 = np.array([10, 255, 255])
        self.red_lower2 = np.array([170, 120, 70])
        self.red_upper2 = np.array([180, 255, 255])
        
        self.blue_lower = np.array([100, 150, 50])
        self.blue_upper = np.array([130, 255, 255])
        
        self.min_sign_area = 500
        self.max_sign_area = 50000
        
        self.detected_signs = []
        self.sign_confidence = {}
        
        print("🚦 Traffic Sign Detector Initialized")
        print("   Supported: STOP, YIELD, SPEED_LIMIT")
    
    def detect_signs(self, frame):
        signs_found = []
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        red_signs = self._detect_red_signs(frame, hsv)
        signs_found.extend(red_signs)
        
        blue_signs = self._detect_blue_signs(frame, hsv)
        signs_found.extend(blue_signs)
        
        self.detected_signs = signs_found
        return signs_found
    
    def _detect_red_signs(self, frame, hsv):
        mask1 = cv2.inRange(hsv, self.red_lower1, self.red_upper1)
        mask2 = cv2.inRange(hsv, self.red_lower2, self.red_upper2)
        red_mask = cv2.bitwise_or(mask1, mask2)
        
        kernel = np.ones((5,5), np.uint8)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detected = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if self.min_sign_area < area < self.max_sign_area:
                x, y, w, h = cv2.boundingRect(cnt)
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
                num_sides = len(approx)
                
                if num_sides == 8:
                    sign_type = SignType.STOP
                    confidence = self._calculate_confidence(cnt, num_sides, 8)
                elif num_sides == 3:
                    sign_type = SignType.YIELD
                    confidence = self._calculate_confidence(cnt, num_sides, 3)
                else:
                    sign_type = SignType.UNKNOWN
                    confidence = 0.3
                
                if confidence > 0.5:
                    detected.append((sign_type, (x, y, w, h), confidence))
        
        return detected
    
    def _detect_blue_signs(self, frame, hsv):
        blue_mask = cv2.inRange(hsv, self.blue_lower, self.blue_upper)
        
        kernel = np.ones((5,5), np.uint8)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detected = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if self.min_sign_area < area < self.max_sign_area:
                x, y, w, h = cv2.boundingRect(cnt)
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
                
                if len(approx) > 6:
                    aspect_ratio = float(w) / h
                    if 0.8 < aspect_ratio < 1.2:
                        sign_type = SignType.SPEED_LIMIT_30 if area < 2000 \
                                   else SignType.SPEED_LIMIT_50
                        confidence = self._calculate_confidence(cnt, len(approx), 10)
                        if confidence > 0.5:
                            detected.append((sign_type, (x, y, w, h), confidence))
        
        return detected
    
    def _calculate_confidence(self, contour, detected_sides, expected_sides):
        shape_match = 1.0 - abs(detected_sides - expected_sides) / expected_sides
        area = cv2.contourArea(contour)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        
        if hull_area > 0:
            solidity = area / hull_area
        else:
            solidity = 0
        
        confidence = (shape_match * 0.7 + solidity * 0.3)
        return max(0, min(1, confidence))
    
    def get_action(self):
        if not self.detected_signs:
            return None, None
        
        sorted_signs = sorted(self.detected_signs, key=lambda x: x[2], reverse=True)
        sign_type, bbox, confidence = sorted_signs[0]
        
        if sign_type == SignType.STOP:
            return "STOP", None
        elif sign_type == SignType.YIELD:
            return "SLOW", 0.3
        elif sign_type == SignType.SPEED_LIMIT_30:
            return "LIMIT", 0.3
        elif sign_type == SignType.SPEED_LIMIT_50:
            return "LIMIT", 0.5
        
        return None, None
    
    def draw_overlay(self, frame):
        for sign_type, (x, y, w, h), confidence in self.detected_signs:
            if sign_type in [SignType.STOP, SignType.YIELD]:
                color = (0, 0, 255)
            elif sign_type in [SignType.SPEED_LIMIT_30, SignType.SPEED_LIMIT_50]:
                color = (255, 0, 0)
            else:
                color = (128, 128, 128)
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
            
            label = f"{sign_type.value} ({confidence*100:.0f}%)"
            label_y = y - 10 if y > 30 else y + h + 25
            
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x, label_y - text_h - 5), 
                         (x + text_w + 5, label_y + 5), color, -1)
            
            cv2.putText(frame, label, (x, label_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame