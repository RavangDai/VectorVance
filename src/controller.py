"""
controller.py - PID Controller for Autonomous Steering
-------------------------------------------------------
Converts steering error (pixels) into motor speed adjustments.

THEORY:
PID = Proportional + Integral + Derivative control
- P: React to current error
- I: Correct accumulated drift
- D: Smooth out changes

OUTPUT:
Motor speed adjustment: -1.0 to +1.0
- Positive → Turn LEFT (slow left wheels)
- Negative → Turn RIGHT (slow right wheels)
"""

import time


class PIDController:
    """
    Discrete PID controller for differential drive steering.
    
    The controller calculates motor speed adjustments to minimize
    steering error over time.
    """
    
    def __init__(self, Kp, Ki, Kd, output_limits=(-1.0, 1.0)):
        """
        Initialize PID controller with tuning parameters.
        
        Args:
            Kp: Proportional gain (how aggressively to react)
            Ki: Integral gain (correct accumulated error)
            Kd: Derivative gain (dampen oscillations)
            output_limits: Tuple of (min, max) output values
        
        TUNING STARTING POINTS:
        - Kp: 0.01 (conservative) to 2.0 (aggressive)
        - Ki: 0.001 (slow correction) to 0.1 (fast correction)
        - Kd: 0.1 (light damping) to 1.0 (heavy damping)
        """
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        
        self.output_min, self.output_max = output_limits
        
        # Internal state variables
        self._integral = 0.0        # Accumulated error over time
        self._previous_error = 0.0  # Error from last iteration
        self._previous_time = None  # Timestamp of last update
        
        print(f"🎛️  PID Controller Initialized")
        print(f"   Kp={Kp} | Ki={Ki} | Kd={Kd}")
        print(f"   Output range: [{self.output_min}, {self.output_max}]")
    
    def compute(self, error, current_time=None):
        """
        Calculate PID output based on current error.
        
        ALGORITHM STEPS:
        1. Calculate time delta (dt)
        2. Proportional term: P = Kp × error
        3. Integral term: I = I_prev + Ki × error × dt
        4. Derivative term: D = Kd × (error - error_prev) / dt
        5. Output = P + I + D (clamped to limits)
        
        Args:
            error: Current steering error in pixels
                   Positive → lane is LEFT of center
                   Negative → lane is RIGHT of center
            current_time: Current timestamp (auto-generated if None)
        
        Returns:
            output: Motor adjustment value (-1.0 to +1.0)
                   Positive → slow LEFT wheels (turn left)
                   Negative → slow RIGHT wheels (turn right)
        """
        # Get current time if not provided
        if current_time is None:
            current_time = time.time()
        
        # Calculate time step (dt)
        if self._previous_time is None:
            dt = 0.0  # First iteration, no time delta
        else:
            dt = current_time - self._previous_time
        
        # Prevent division by zero
        if dt <= 0.0:
            dt = 0.01  # Assume 100Hz update rate
        
        # ============================================
        # P: PROPORTIONAL TERM
        # ============================================
        # Direct reaction to current error
        # Large error → Large correction
        P = self.Kp * error
        
        # ============================================
        # I: INTEGRAL TERM
        # ============================================
        # Accumulate error over time
        # Eliminates steady-state error (constant drift)
        self._integral += error * dt
        
        # Anti-windup: Prevent integral from growing too large
        # WHY: If robot is stuck, integral keeps growing → huge overshoot when freed
        integral_limit = 100.0 / (self.Ki + 0.001)  # Scale based on Ki
        self._integral = max(-integral_limit, min(integral_limit, self._integral))
        
        I = self.Ki * self._integral
        
        # ============================================
        # D: DERIVATIVE TERM
        # ============================================
        # Rate of change of error
        # Dampens oscillations by predicting future error
        if dt > 0:
            derivative = (error - self._previous_error) / dt
        else:
            derivative = 0.0
        
        D = self.Kd * derivative
        
        # ============================================
        # COMBINE TERMS
        # ============================================
        output = P + I + D
        
        # Clamp output to limits
        output = max(self.output_min, min(self.output_max, output))
        
        # Store state for next iteration
        self._previous_error = error
        self._previous_time = current_time
        
        return output
    
    def reset(self):
        """
        Reset controller state.
        
        Call this when:
        - Starting a new run
        - Robot has been stopped for a while
        - Switching between manual and autonomous mode
        """
        self._integral = 0.0
        self._previous_error = 0.0
        self._previous_time = None
        print("🔄 PID Controller reset")
    
    def get_state(self):
        """
        Get current PID state for debugging/tuning.
        
        Returns:
            Dictionary with P, I, D components and settings
        """
        return {
            'Kp': self.Kp,
            'Ki': self.Ki,
            'Kd': self.Kd,
            'integral': self._integral,
            'previous_error': self._previous_error
        }
    
    def tune(self, Kp=None, Ki=None, Kd=None):
        """
        Adjust PID gains on-the-fly (useful for live tuning).
        
        Args:
            Kp, Ki, Kd: New gain values (None = keep current)
        """
        if Kp is not None:
            self.Kp = Kp
            print(f"📊 Kp adjusted to {Kp}")
        
        if Ki is not None:
            self.Ki = Ki
            print(f"📊 Ki adjusted to {Ki}")
        
        if Kd is not None:
            self.Kd = Kd
            print(f"📊 Kd adjusted to {Kd}")


# ============================================
# TEST FUNCTION - Simulate PID with synthetic data
# ============================================

def test_pid_response():
    """
    Test PID controller with simulated steering errors.
    
    SIMULATION:
    - Robot starts 100px off-center
    - PID gradually corrects it
    - We plot the response curve
    """
    import matplotlib.pyplot as plt
    
    # Create PID controller
    # Starting with conservative gains
    pid = PIDController(Kp=0.5, Ki=0.05, Kd=0.3)
    
    # Simulation parameters
    initial_error = 100  # pixels off-center
    simulation_time = 5.0  # seconds
    dt = 0.05  # 20 Hz update rate
    
    # Data storage for plotting
    times = []
    errors = []
    outputs = []
    
    # Simulate robot behavior
    error = initial_error
    t = 0.0
    
    print("\n🎮 Running PID Simulation...")
    print("Initial error: 100px (robot is LEFT of center)")
    print("-" * 50)
    
    while t < simulation_time:
        # Calculate PID output
        output = pid.compute(error, current_time=t)
        
        # Simulate robot response to control output
        # In reality: output → motor speeds → robot moves → error changes
        # Here: We fake it with a simple model
        error -= output * 30  # Assume 30px correction per unit output
        
        # Add some "noise" to simulate real world
        import random
        error += random.uniform(-2, 2)
        
        # Store data
        times.append(t)
        errors.append(error)
        outputs.append(output)
        
        # Print every 0.5 seconds
        if int(t * 10) % 5 == 0:
            print(f"t={t:.1f}s | Error={error:+6.1f}px | Output={output:+5.2f}")
        
        t += dt
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot error over time
    ax1.plot(times, errors, 'b-', linewidth=2, label='Steering Error')
    ax1.axhline(y=0, color='g', linestyle='--', label='Target (centered)')
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Error (pixels)')
    ax1.set_title('PID Response: Error Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot PID output over time
    ax2.plot(times, outputs, 'r-', linewidth=2, label='PID Output')
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Control Output')
    ax2.set_title('PID Controller Output')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_path = "../outputs/pid_simulation.png"
    plt.savefig(output_path, dpi=150)
    print(f"\n✅ Simulation complete!")
    print(f"📊 Plot saved to: {output_path}")
    
    plt.show()


# Entry point
if __name__ == "__main__":
    test_pid_response()