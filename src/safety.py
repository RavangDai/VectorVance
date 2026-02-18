"""
safety.py - Obstacle Detection
-------------------------------
Final refined version.
"""

import cv2
import numpy as np
import random


class ObstacleDetector:
    def __init__(self, emergency_distance=20, warning_distance=50, max_range=200):
        self.emergency_distance = emergency_distance
        self.warning_distance = warning_distance
        self.max_range = max_range
        
        self.sensors = {
            'front': {'distance': max_range, 'angle': 0},
            'front_left': {'distance': max_range, 'angle': -30},
            'front_right': {'distance': max_range, 'angle': 30}
        }
        
        self.emergency_stop_triggered = False
        self.warning_triggered = False
        
        print("🛡️  Obstacle Detection Initialized")
        print(f"   Emergency stop: <{emergency_distance}cm")
        print(f"   Warning zone:   <{warning_distance}cm")
    
    def simulate_sensors(self, frame, steering_error):
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        front_roi = gray[int(h*0.4):int(h*0.7), int(w*0.4):int(w*0.6)]
        left_roi = gray[int(h*0.4):int(h*0.7), int(w*0.2):int(w*0.4)]
        right_roi = gray[int(h*0.4):int(h*0.7), int(w*0.6):int(w*0.8)]
        
        front_brightness = np.mean(front_roi) if front_roi.size > 0 else 255
        left_brightness = np.mean(left_roi) if left_roi.size > 0 else 255
        right_brightness = np.mean(right_roi) if right_roi.size > 0 else 255
        
        noise = lambda: random.uniform(-5, 5)
        
        front_dist = min(self.max_range, (front_brightness / 255) * self.max_range + noise())
        left_dist = min(self.max_range, (left_brightness / 255) * self.max_range + noise())
        right_dist = min(self.max_range, (right_brightness / 255) * self.max_range + noise())
        
        self.sensors['front']['distance'] = max(5, front_dist)
        self.sensors['front_left']['distance'] = max(5, left_dist)
        self.sensors['front_right']['distance'] = max(5, right_dist)
        
        self._check_obstacles()
        return self.sensors
    
    def _check_obstacles(self):
        min_distance = min(s['distance'] for s in self.sensors.values())
        
        if min_distance < self.emergency_distance:
            self.emergency_stop_triggered = True
            self.warning_triggered = True
        elif min_distance < self.warning_distance:
            self.emergency_stop_triggered = False
            self.warning_triggered = True
        else:
            self.emergency_stop_triggered = False
            self.warning_triggered = False
    
    def should_emergency_stop(self):
        return self.emergency_stop_triggered
    
    def should_slow_down(self):
        return self.warning_triggered
    
    def get_speed_modifier(self):
        if self.emergency_stop_triggered:
            return 0.0
        
        min_distance = min(s['distance'] for s in self.sensors.values())
        
        if min_distance < self.warning_distance:
            ratio = (min_distance - self.emergency_distance) / \
                   (self.warning_distance - self.emergency_distance)
            return max(0.3, min(1.0, ratio))
        
        return 1.0
    
    def draw_overlay(self, frame):
        h, w = frame.shape[:2]
        center_x = w // 2
        center_y = int(h * 0.85)
        
        front_color = self._get_sensor_color(self.sensors['front']['distance'])
        cv2.circle(frame, (center_x, center_y), 15, front_color, -1)
        cv2.putText(frame, f"{int(self.sensors['front']['distance'])}cm", 
                   (center_x - 20, center_y - 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, front_color, 2)
        
        left_color = self._get_sensor_color(self.sensors['front_left']['distance'])
        cv2.circle(frame, (center_x - 60, center_y), 12, left_color, -1)
        cv2.putText(frame, f"{int(self.sensors['front_left']['distance'])}", 
                   (center_x - 80, center_y - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, left_color, 1)
        
        right_color = self._get_sensor_color(self.sensors['front_right']['distance'])
        cv2.circle(frame, (center_x + 60, center_y), 12, right_color, -1)
        cv2.putText(frame, f"{int(self.sensors['front_right']['distance'])}", 
                   (center_x + 65, center_y - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, right_color, 1)
        
        if self.emergency_stop_triggered:
            cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), 10)
            cv2.putText(frame, "!!! EMERGENCY STOP !!!", 
                       (w//2 - 180, h//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        elif self.warning_triggered:
            cv2.putText(frame, "OBSTACLE DETECTED - SLOWING", 
                       (10, h - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        
        return frame
    
    def _get_sensor_color(self, distance):
        if distance < self.emergency_distance:
            return (0, 0, 255)
        elif distance < self.warning_distance:
            return (0, 165, 255)
        else:
            return (0, 255, 0)
    
    def reset(self):
        for sensor in self.sensors.values():
            sensor['distance'] = self.max_range
        self.emergency_stop_triggered = False
        self.warning_triggered = False