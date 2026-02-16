"""
controller.py - PID Controller
Converts steering error into motor speed adjustments.
"""

import time


class PIDController:
    """
    Discrete PID controller for differential drive steering.
    output = Kp*error + Ki*∫error*dt + Kd*d(error)/dt
    """
    
    def __init__(self, Kp, Ki, Kd, output_limits=(-1.0, 1.0), derivative_filter_size=5):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        
        self.output_min, self.output_max = output_limits
        
        self._integral = 0.0
        self._previous_error = 0.0
        self._previous_time = None
        
        # Moving average filter for derivative noise
        from collections import deque
        self.derivative_filter_size = derivative_filter_size
        self.derivative_history = deque(maxlen=derivative_filter_size)
        
        self.last_P = 0.0
        self.last_I = 0.0
        self.last_D = 0.0
        
        print(f"🎛️  PID Controller Initialized")
        print(f"   Gains: Kp={Kp:.4f} | Ki={Ki:.4f} | Kd={Kd:.4f}")
        print(f"   Output: [{self.output_min:+.2f}, {self.output_max:+.2f}]")
    
    def compute(self, error, current_time=None):
        """
        Calculate PID output for current error.
        +ve error → lane LEFT → turn LEFT → +ve output → slow LEFT wheels
        """
        if current_time is None:
            current_time = time.time()
        
        if self._previous_time is None:
            dt = 0.02
        else:
            dt = current_time - self._previous_time
        
        if dt <= 0.0 or dt > 1.0:
            dt = 0.02
        
        # Proportional
        P = self.Kp * error
        
        # Integral (with anti-windup)
        was_saturated = (self.last_P + self.last_I + self.last_D) >= self.output_max or \
                        (self.last_P + self.last_I + self.last_D) <= self.output_min
        
        if not was_saturated or (error * self._integral < 0):
            self._integral += error * dt
        
        integral_limit = 200.0
        self._integral = max(-integral_limit, min(integral_limit, self._integral))
        
        I = self.Ki * self._integral
        
        # Derivative (with filtering)
        if dt > 0:
            raw_derivative = (error - self._previous_error) / dt
        else:
            raw_derivative = 0.0
        
        self.derivative_history.append(raw_derivative)
        filtered_derivative = sum(self.derivative_history) / len(self.derivative_history)
        
        D = self.Kd * filtered_derivative
        
        # Combine and clamp
        output = P + I + D
        output = max(self.output_min, min(self.output_max, output))
        
        self._previous_error = error
        self._previous_time = current_time
        
        self.last_P = P
        self.last_I = I
        self.last_D = D
        
        return output
    
    def reset(self):
        """Reset controller state."""
        self._integral = 0.0
        self._previous_error = 0.0
        self._previous_time = None
        self.derivative_history.clear()
        
        self.last_P = 0.0
        self.last_I = 0.0
        self.last_D = 0.0
        
        print("🔄 PID reset")
    
    def get_state(self):
        """Get current PID state for debugging."""
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
    
    def tune(self, Kp=None, Ki=None, Kd=None):
        """Adjust gains on-the-fly."""
        if Kp is not None:
            self.Kp = Kp
            print(f"📊 Kp → {Kp:.4f}")
        
        if Ki is not None:
            self.Ki = Ki
            self._integral = 0.0
            print(f"📊 Ki → {Ki:.4f} (integral reset)")
        
        if Kd is not None:
            self.Kd = Kd
            print(f"📊 Kd → {Kd:.4f}")


def test_pid_simulation():
    """Simulate PID response with synthetic errors."""
    import matplotlib.pyplot as plt
    
    pid = PIDController(Kp=0.003, Ki=0.0002, Kd=0.001)
    
    initial_error = 80
    duration = 5.0
    dt = 0.05
    
    times = []
    errors = []
    outputs = []
    p_terms = []
    i_terms = []
    d_terms = []
    
    error = initial_error
    t = 0.0
    
    print("\n🎮 PID Simulation Running...")
    print(f"Initial error: {initial_error}px")
    print("-" * 60)
    
    while t < duration:
        output = pid.compute(error, current_time=t)
        state = pid.get_state()
        
        correction = output * 25
        error -= correction
        
        import random
        error += random.uniform(-1.5, 1.5)
        
        times.append(t)
        errors.append(error)
        outputs.append(output)
        p_terms.append(state['P_term'])
        i_terms.append(state['I_term'])
        d_terms.append(state['D_term'])
        
        if int(t / 0.5) != int((t - dt) / 0.5):
            print(f"t={t:.1f}s | Error={error:+6.1f}px | Output={output:+.3f} | "
                  f"P={state['P_term']:+.3f} I={state['I_term']:+.3f} D={state['D_term']:+.3f}")
        
        t += dt
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    axes[0].plot(times, errors, 'b-', linewidth=2, label='Steering Error')
    axes[0].axhline(y=0, color='g', linestyle='--', label='Target (centered)')
    axes[0].fill_between(times, -5, 5, color='g', alpha=0.1, label='Acceptable range')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Error (pixels)')
    axes[0].set_title('PID Response: Error Convergence')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(times, outputs, 'r-', linewidth=2, label='Total Output')
    axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Control Output')
    axes[1].set_title('PID Controller Output')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(times, p_terms, 'b-', linewidth=1.5, label='P term', alpha=0.7)
    axes[2].plot(times, i_terms, 'g-', linewidth=1.5, label='I term', alpha=0.7)
    axes[2].plot(times, d_terms, 'r-', linewidth=1.5, label='D term', alpha=0.7)
    axes[2].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Component Value')
    axes[2].set_title('PID Components (P + I + D)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = "../outputs/pid_simulation.png"
    plt.savefig(output_path, dpi=150)
    print(f"\n✅ Simulation complete!")
    print(f"📊 Plot saved: {output_path}")
    
    settling_time = next((t for t, e in zip(times, errors) if abs(e) < 5), None)
    final_error = errors[-1]
    
    print(f"\n📈 Performance Metrics:")
    print(f"   Settling time (±5px): {settling_time:.2f}s" if settling_time else "   Did not settle")
    print(f"   Final error: {final_error:+.1f}px")
    print(f"   Max overshoot: {min(errors):+.1f}px")
    
    plt.show()


if __name__ == "__main__":
    test_pid_simulation()