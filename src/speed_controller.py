"""
controller.py - PID Controller
-------------------------------
Final refined version.
"""

import time
from collections import deque


class PIDController:
    def __init__(self, Kp, Ki, Kd, output_limits=(-1.0, 1.0), derivative_filter_size=5):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.output_min, self.output_max = output_limits
        
        self._integral = 0.0
        self._previous_error = 0.0
        self._previous_time = None
        
        self.derivative_filter_size = derivative_filter_size
        self.derivative_history = deque(maxlen=derivative_filter_size)
        
        self.last_P = 0.0
        self.last_I = 0.0
        self.last_D = 0.0
        
        print(f"🎛️  PID Controller Initialized")
        print(f"   Gains: Kp={Kp:.4f} | Ki={Ki:.4f} | Kd={Kd:.4f}")
        print(f"   Output: [{self.output_min:+.2f}, {self.output_max:+.2f}]")
    
    def compute(self, error, current_time=None):
        if current_time is None:
            current_time = time.time()
        
        if self._previous_time is None:
            dt = 0.02
        else:
            dt = current_time - self._previous_time
        
        if dt <= 0.0 or dt > 1.0:
            dt = 0.02
        
        P = self.Kp * error
        
        was_saturated = (self.last_P + self.last_I + self.last_D) >= self.output_max or \
                        (self.last_P + self.last_I + self.last_D) <= self.output_min
        
        if not was_saturated or (error * self._integral < 0):
            self._integral += error * dt
        
        integral_limit = 200.0
        self._integral = max(-integral_limit, min(integral_limit, self._integral))
        I = self.Ki * self._integral
        
        if dt > 0:
            raw_derivative = (error - self._previous_error) / dt
        else:
            raw_derivative = 0.0
        
        self.derivative_history.append(raw_derivative)
        filtered_derivative = sum(self.derivative_history) / len(self.derivative_history)
        D = self.Kd * filtered_derivative
        
        output = P + I + D
        output = max(self.output_min, min(self.output_max, output))
        
        self._previous_error = error
        self._previous_time = current_time
        self.last_P = P
        self.last_I = I
        self.last_D = D
        
        return output
    
    def reset(self):
        self._integral = 0.0
        self._previous_error = 0.0
        self._previous_time = None
        self.derivative_history.clear()
        self.last_P = 0.0
        self.last_I = 0.0
        self.last_D = 0.0
    
    def get_state(self):
        return {
            'Kp': self.Kp,
            'Ki': self.Ki,
            'Kd': self.Kd,
            'P_term': self.last_P,
            'I_term': self.last_I,
            'D_term': self.last_D,
            'integral': self._integral,
            'previous_error': self._previous_error
        }