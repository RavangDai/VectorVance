"""
hardware.py - Raspberry Pi GPIO Motor & Sensor Control
-------------------------------------------------------
Hardware abstraction layer for physical robot deployment.

HARDWARE CONNECTIONS:
- L298N Motor Driver (4WD differential drive)
- HC-SR04 Ultrasonic Sensors (x2)
- Pi Camera Module v2

Author: Autonomous Vehicle Team
"""

import RPi.GPIO as GPIO
import time
import threading


class MotorController:
    """
    Controls 4WD motors via L298N motor driver.
    
    WIRING (Raspberry Pi → L298N):
    --------------------------------
    GPIO 17 → IN1 (Left motor forward)
    GPIO 18 → IN2 (Left motor backward)
    GPIO 22 → IN3 (Right motor forward)
    GPIO 23 → IN4 (Right motor backward)
    GPIO 12 → ENA (Left motor PWM speed)
    GPIO 13 → ENB (Right motor PWM speed)
    GND     → GND
    
    L298N → Motors:
    ---------------
    OUT1, OUT2 → Left motors (parallel)
    OUT3, OUT4 → Right motors (parallel)
    
    Power:
    ------
    7.4V Battery → L298N (12V input terminal)
    """
    
    def __init__(self, max_speed=100):
        """
        Initialize motor controller.
        
        Args:
            max_speed: Maximum PWM duty cycle (0-100)
        """
        # GPIO pin assignments
        self.LEFT_FORWARD = 17
        self.LEFT_BACKWARD = 18
        self.RIGHT_FORWARD = 22
        self.RIGHT_BACKWARD = 23
        self.LEFT_PWM_PIN = 12
        self.RIGHT_PWM_PIN = 13
        
        self.max_speed = max_speed
        
        # Setup GPIO
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        
        # Setup direction pins
        GPIO.setup(self.LEFT_FORWARD, GPIO.OUT)
        GPIO.setup(self.LEFT_BACKWARD, GPIO.OUT)
        GPIO.setup(self.RIGHT_FORWARD, GPIO.OUT)
        GPIO.setup(self.RIGHT_BACKWARD, GPIO.OUT)
        
        # Setup PWM pins
        GPIO.setup(self.LEFT_PWM_PIN, GPIO.OUT)
        GPIO.setup(self.RIGHT_PWM_PIN, GPIO.OUT)
        
        # Create PWM objects (1000 Hz frequency)
        self.left_pwm = GPIO.PWM(self.LEFT_PWM_PIN, 1000)
        self.right_pwm = GPIO.PWM(self.RIGHT_PWM_PIN, 1000)
        
        # Start PWM at 0% duty cycle
        self.left_pwm.start(0)
        self.right_pwm.start(0)
        
        print("🚗 Motor Controller Initialized")
        print(f"   Max speed: {max_speed}%")
    
    def set_speeds(self, left_speed, right_speed):
        """
        Set motor speeds for differential drive.
        
        Args:
            left_speed: -1.0 to +1.0 (negative = backward)
            right_speed: -1.0 to +1.0 (negative = backward)
        """
        # Clamp to valid range
        left_speed = max(-1.0, min(1.0, left_speed))
        right_speed = max(-1.0, min(1.0, right_speed))
        
        # Set left motor
        if left_speed > 0:
            # Forward
            GPIO.output(self.LEFT_FORWARD, GPIO.HIGH)
            GPIO.output(self.LEFT_BACKWARD, GPIO.LOW)
            duty_cycle = abs(left_speed) * self.max_speed
        elif left_speed < 0:
            # Backward
            GPIO.output(self.LEFT_FORWARD, GPIO.LOW)
            GPIO.output(self.LEFT_BACKWARD, GPIO.HIGH)
            duty_cycle = abs(left_speed) * self.max_speed
        else:
            # Stop
            GPIO.output(self.LEFT_FORWARD, GPIO.LOW)
            GPIO.output(self.LEFT_BACKWARD, GPIO.LOW)
            duty_cycle = 0
        
        self.left_pwm.ChangeDutyCycle(duty_cycle)
        
        # Set right motor
        if right_speed > 0:
            # Forward
            GPIO.output(self.RIGHT_FORWARD, GPIO.HIGH)
            GPIO.output(self.RIGHT_BACKWARD, GPIO.LOW)
            duty_cycle = abs(right_speed) * self.max_speed
        elif right_speed < 0:
            # Backward
            GPIO.output(self.RIGHT_FORWARD, GPIO.LOW)
            GPIO.output(self.RIGHT_BACKWARD, GPIO.HIGH)
            duty_cycle = abs(right_speed) * self.max_speed
        else:
            # Stop
            GPIO.output(self.RIGHT_FORWARD, GPIO.LOW)
            GPIO.output(self.RIGHT_BACKWARD, GPIO.LOW)
            duty_cycle = 0
        
        self.right_pwm.ChangeDutyCycle(duty_cycle)
    
    def emergency_stop(self):
        """Immediately stop all motors."""
        self.set_speeds(0, 0)
        print("🛑 EMERGENCY STOP ACTIVATED")
    
    def cleanup(self):
        """Clean shutdown of GPIO."""
        self.emergency_stop()
        self.left_pwm.stop()
        self.right_pwm.stop()
        GPIO.cleanup()
        print("✅ Motor controller shutdown complete")


class UltrasonicSensor:
    """
    HC-SR04 Ultrasonic Distance Sensor.
    
    WIRING (per sensor):
    --------------------
    VCC  → 5V
    TRIG → GPIO pin (specified in __init__)
    ECHO → GPIO pin (specified in __init__)
    GND  → GND
    
    MEASUREMENT:
    -----------
    - Sends 10us pulse on TRIG
    - Measures pulse width on ECHO
    - Distance = (pulse_duration * 34300) / 2 (in cm)
    """
    
    def __init__(self, trig_pin, echo_pin, name="Sensor"):
        """
        Initialize ultrasonic sensor.
        
        Args:
            trig_pin: GPIO pin for trigger
            echo_pin: GPIO pin for echo
            name: Sensor identifier
        """
        self.trig_pin = trig_pin
        self.echo_pin = echo_pin
        self.name = name
        
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.trig_pin, GPIO.OUT)
        GPIO.setup(self.echo_pin, GPIO.IN)
        
        # Ensure trigger is low
        GPIO.output(self.trig_pin, GPIO.LOW)
        time.sleep(0.1)
        
        print(f"📡 {name} initialized (TRIG:{trig_pin}, ECHO:{echo_pin})")
    
    def get_distance(self):
        """
        Measure distance to nearest object.
        
        Returns:
            float: Distance in cm (or 999.9 on timeout/error)
        """
        # Send 10us pulse
        GPIO.output(self.trig_pin, GPIO.HIGH)
        time.sleep(0.00001)  # 10 microseconds
        GPIO.output(self.trig_pin, GPIO.LOW)
        
        # Timeout values
        timeout = 0.05  # 50ms timeout
        start_time = time.time()
        
        # Wait for echo start
        pulse_start = time.time()
        while GPIO.input(self.echo_pin) == 0:
            pulse_start = time.time()
            if pulse_start - start_time > timeout:
                return 999.9  # Timeout (no object detected)
        
        # Wait for echo end
        pulse_end = time.time()
        while GPIO.input(self.echo_pin) == 1:
            pulse_end = time.time()
            if pulse_end - pulse_start > timeout:
                return 999.9  # Timeout
        
        # Calculate distance
        pulse_duration = pulse_end - pulse_start
        distance = (pulse_duration * 34300) / 2  # Speed of sound = 343 m/s
        
        # Clamp to sensor range (2cm - 400cm)
        if distance < 2:
            return 2.0
        elif distance > 400:
            return 400.0
        else:
            return round(distance, 1)


class HardwareObstacleDetector:
    """
    Real obstacle detection using HC-SR04 sensors.
    
    Replaces simulated obstacle detection with actual hardware.
    """
    
    def __init__(self, emergency_distance=20, warning_distance=50):
        """
        Initialize with two ultrasonic sensors (left and right).
        
        WIRING:
        -------
        Front-Left Sensor:  TRIG=GPIO 24, ECHO=GPIO 25
        Front-Right Sensor: TRIG=GPIO 5,  ECHO=GPIO 6
        
        Args:
            emergency_distance: Stop if object < this (cm)
            warning_distance: Slow down if object < this (cm)
        """
        self.emergency_distance = emergency_distance
        self.warning_distance = warning_distance
        
        # Initialize sensors
        self.sensor_left = UltrasonicSensor(24, 25, "Front-Left")
        self.sensor_right = UltrasonicSensor(5, 6, "Front-Right")
        
        # State
        self.left_distance = 400
        self.right_distance = 400
        self.emergency_stop_triggered = False
        self.warning_triggered = False
        
        # Background thread for continuous monitoring
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        print("🛡️  Hardware Obstacle Detection Active")
        print(f"   Emergency: <{emergency_distance}cm")
        print(f"   Warning: <{warning_distance}cm")
    
    def _monitor_loop(self):
        """Background thread - continuously read sensors."""
        while self.monitoring:
            # Read both sensors
            self.left_distance = self.sensor_left.get_distance()
            self.right_distance = self.sensor_right.get_distance()
            
            # Check thresholds
            min_distance = min(self.left_distance, self.right_distance)
            
            if min_distance < self.emergency_distance:
                self.emergency_stop_triggered = True
                self.warning_triggered = True
            elif min_distance < self.warning_distance:
                self.emergency_stop_triggered = False
                self.warning_triggered = True
            else:
                self.emergency_stop_triggered = False
                self.warning_triggered = False
            
            time.sleep(0.05)  # 20 Hz update rate
    
    def should_emergency_stop(self):
        """Check if emergency stop needed."""
        return self.emergency_stop_triggered
    
    def should_slow_down(self):
        """Check if slowing needed."""
        return self.warning_triggered
    
    def get_speed_modifier(self):
        """
        Calculate speed reduction based on nearest obstacle.
        
        Returns:
            float: 0.0 (stop) to 1.0 (full speed)
        """
        if self.emergency_stop_triggered:
            return 0.0
        
        min_distance = min(self.left_distance, self.right_distance)
        
        if min_distance < self.warning_distance:
            # Linear scaling between emergency and warning distances
            ratio = (min_distance - self.emergency_distance) / \
                   (self.warning_distance - self.emergency_distance)
            return max(0.3, min(1.0, ratio))
        
        return 1.0
    
    def get_distances(self):
        """Get current sensor readings."""
        return {
            'left': self.left_distance,
            'right': self.right_distance,
            'min': min(self.left_distance, self.right_distance)
        }
    
    def stop_monitoring(self):
        """Stop background monitoring thread."""
        self.monitoring = False
        if self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=1.0)


# ============================================
# TEST FUNCTIONS
# ============================================

def test_motors():
    """Test motor controller."""
    print("\n🧪 Testing Motors...")
    motors = MotorController(max_speed=50)  # 50% max for safety
    
    try:
        print("Forward...")
        motors.set_speeds(0.5, 0.5)
        time.sleep(2)
        
        print("Turn left...")
        motors.set_speeds(0.3, 0.7)
        time.sleep(2)
        
        print("Turn right...")
        motors.set_speeds(0.7, 0.3)
        time.sleep(2)
        
        print("Stop...")
        motors.emergency_stop()
        
    finally:
        motors.cleanup()
    
    print("✅ Motor test complete")


def test_ultrasonic():
    """Test ultrasonic sensors."""
    print("\n🧪 Testing Ultrasonic Sensors...")
    detector = HardwareObstacleDetector()
    
    try:
        for i in range(20):
            distances = detector.get_distances()
            print(f"Left: {distances['left']:6.1f}cm | "
                  f"Right: {distances['right']:6.1f}cm | "
                  f"Min: {distances['min']:6.1f}cm")
            
            if detector.should_emergency_stop():
                print("🚨 EMERGENCY STOP!")
            elif detector.should_slow_down():
                print("⚠️  SLOWING DOWN")
            
            time.sleep(0.5)
    
    finally:
        detector.stop_monitoring()
        GPIO.cleanup()
    
    print("✅ Sensor test complete")


if __name__ == "__main__":
    print("Hardware Module Test")
    print("=" * 50)
    print("1. Test Motors")
    print("2. Test Ultrasonic Sensors")
    print("3. Exit")
    
    choice = input("\nSelect test (1-3): ")
    
    if choice == "1":
        test_motors()
    elif choice == "2":
        test_ultrasonic()
    else:
        print("Exiting...")