"""
dev_server.py — Standalone dashboard preview (Windows / no Pi hardware)
Run:  python dev_server.py
Then open:  http://localhost:5000/
"""

import time
import math
import threading
import pi_server

def _push_fake_telemetry():
    t = 0.0
    while True:
        t += 0.05
        steering = math.sin(t) * 60
        left  = max(0.0, 0.6 + math.sin(t) * 0.3)
        right = max(0.0, 0.6 - math.sin(t) * 0.3)
        pi_server.push_telemetry({
            "mode":               "AUTONOMOUS",
            "drive_mode":         "LANE",
            "status":             "STRAIGHT" if abs(steering) < 20 else "TURNING",
            "car_state":          "LANE_FOLLOW",
            "speed_left":         round(left, 3),
            "speed_right":        round(right, 3),
            "base_speed":         0.6,
            "steering_error":     round(steering, 1),
            "front_distance_cm":  85.0,
            "fps":                28.5,
            "frame_count":        int(t * 20),
            "stop_signs_detected": 0,
            "stop_sign_active":   False,
            "dnn_enabled":        True,
            "dnn_detections":     [],
            "speed_limit_kmh":    None,
            "obstacle_modifier":  1.0,
            "track_available":    True,
            "track_click_available": True,
            "track_mode":         "CLASS",
            "track_target":       "",
            "track_locked":       False,
            "track_conf":         0.0,
            "track_lost_frames":  999,
            "track_classes":      ["person","bicycle","car","dog","cat","sports ball","bottle","cup","chair","laptop","cell phone","book","teddy bear"],
            "sentry_armed":       False,
            "sentry_motion":      False,
            "sentry_person":      False,
        })
        time.sleep(0.5)

if __name__ == "__main__":
    threading.Thread(target=_push_fake_telemetry, daemon=True).start()
    print("Dashboard → http://localhost:5000/")
    pi_server.start_server(port=5000)
    # Keep main thread alive
    while True:
        time.sleep(1)
