"""
speed_controller.py - Adaptive speed based on steering error.

Slows down proportionally to how sharp the upcoming turn is,
so the car doesn't overshoot curves at full throttle.
"""

import cv2


class AdaptiveSpeedController:
    STRAIGHT_THRESHOLD  = 30
    GENTLE_THRESHOLD    = 80
    MODERATE_THRESHOLD  = 150

    SPEED_MAP = {
        "STRAIGHT":       1.0,
        "GENTLE_CURVE":   0.75,
        "MODERATE_CURVE": 0.5,
        "SHARP_CURVE":    0.3,
    }

    def __init__(self, min_speed=0.2, max_speed=0.8):
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.target_speed = max_speed
        self._current_category = "STRAIGHT"

    def get_speed_category(self, abs_error):
        if abs_error < self.STRAIGHT_THRESHOLD:
            return "STRAIGHT"
        elif abs_error < self.GENTLE_THRESHOLD:
            return "GENTLE_CURVE"
        elif abs_error < self.MODERATE_THRESHOLD:
            return "MODERATE_CURVE"
        else:
            return "SHARP_CURVE"

    def calculate_speed(self, steering_error, obstacle_modifier=1.0):
        abs_error = abs(steering_error)
        self._current_category = self.get_speed_category(abs_error)

        multiplier = self.SPEED_MAP[self._current_category]
        speed = self.max_speed * multiplier * obstacle_modifier
        self.target_speed = max(self.min_speed, min(self.max_speed, speed))
        return self.target_speed

    def reset(self):
        self.target_speed = self.max_speed
        self._current_category = "STRAIGHT"


def draw_speed_indicator(frame, current_speed, target_speed, category):
    """Draw a compact speed bar in the bottom-left of the frame."""
    h, w = frame.shape[:2]
    x, y = 10, h - 80

    color_map = {
        "STRAIGHT":       (0, 255, 0),
        "GENTLE_CURVE":   (0, 255, 255),
        "MODERATE_CURVE": (0, 165, 255),
        "SHARP_CURVE":    (0, 0, 255),
    }
    color = color_map.get(category, (200, 200, 200))

    bar_width  = 200
    bar_height = 20
    cv2.rectangle(frame, (x, y), (x + bar_width, y + bar_height), (50, 50, 50), -1)

    fill = int(bar_width * current_speed)
    cv2.rectangle(frame, (x, y), (x + fill, y + bar_height), color, -1)

    # marker showing the target speed
    target_x = x + int(bar_width * target_speed)
    cv2.line(frame, (target_x, y - 3), (target_x, y + bar_height + 3), (255, 255, 255), 2)

    cv2.putText(frame, f"Speed: {current_speed*100:.0f}%",
                (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, category.replace("_", " "),
                (x + bar_width + 10, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return frame
