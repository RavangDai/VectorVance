"""
hardware.py - GPIO motor and sensor control for Raspberry Pi.
--------------------------------------------------------------
VectorVance Autonomous Vehicle Project

This module provides hardware abstraction for:
  - L298N motor driver control via GPIO
  - HC-SR04 ultrasonic distance sensors
  - PWM speed control for differential drive

WIRING DIAGRAM:
===============

┌─────────────────────────────────────────────────────────────────────────────┐
│                        RASPBERRY PI 3/4 GPIO PINOUT                         │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                         RASPBERRY PI                                │   │
│   │                                                                     │   │
│   │   3.3V  (1) (2)  5V ─────────────────────────┐                     │   │
│   │   GPIO2 (3) (4)  5V                          │                     │   │
│   │   GPIO3 (5) (6)  GND ──────────────────────┐ │                     │   │
│   │   GPIO4 (7) (8)  GPIO14                    │ │                     │   │
│   │   GND   (9) (10) GPIO15                    │ │                     │   │
│   │  GPIO17 (11)(12) GPIO18                    │ │                     │   │
│   │  GPIO27 (13)(14) GND                       │ │                     │   │
│   │  GPIO22 (15)(16) GPIO23                    │ │                     │   │
│   │   3.3V  (17)(18) GPIO24                    │ │                     │   │
│   │  GPIO10 (19)(20) GND                       │ │                     │   │
│   │   GPIO9 (21)(22) GPIO25                    │ │                     │   │
│   │  GPIO11 (23)(24) GPIO8                     │ │                     │   │
│   │   GND   (25)(26) GPIO7                     │ │                     │   │
│   │   GPIO0 (27)(28) GPIO1                     │ │                     │   │
│   │   GPIO5 (29)(30) GND                       │ │                     │   │
│   │   GPIO6 (31)(32) GPIO12                    │ │                     │   │
│   │  GPIO13 (33)(34) GND                       │ │                     │   │
│   │  GPIO19 (35)(36) GPIO16                    │ │                     │   │
│   │  GPIO26 (37)(38) GPIO20                    │ │                     │   │
│   │   GND   (39)(40) GPIO21                    │ │                     │   │
│   │                                                                     │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

MOTOR DRIVER CONNECTIONS (L298N):
=================================

    Raspberry Pi              L298N Motor Driver              Motors
    ────────────              ──────────────────              ──────
    
    GPIO 17 (Pin 11) ────────► IN1  ┐                    ┌──── Motor A
    GPIO 18 (Pin 12) ────────► IN2  ├── Left Motor ──────┤    (LEFT)
    GPIO 12 (Pin 32) ────────► ENA  ┘   Control          └────────────
                                        (PWM Speed)
    
    GPIO 22 (Pin 15) ────────► IN3  ┐                    ┌──── Motor B
    GPIO 23 (Pin 16) ────────► IN4  ├── Right Motor ─────┤    (RIGHT)
    GPIO 13 (Pin 33) ────────► ENB  ┘   Control          └────────────
                                        (PWM Speed)
    
    GND (Pin 6/9/14) ────────► GND
    5V  (Pin 2/4)    ────────► +5V (Logic)
    
    Battery 7-12V    ────────► +12V (Motor Power)
    Battery GND      ────────► GND


ULTRASONIC SENSOR CONNECTIONS (HC-SR04):
========================================

    Front-Left Sensor:
    ─────────────────
    GPIO 24 (Pin 18) ────────► TRIG
    GPIO 25 (Pin 22) ◄──────── ECHO (use voltage divider!)
    5V               ────────► VCC
    GND              ────────► GND

    Front-Right Sensor:
    ──────────────────
    GPIO 5  (Pin 29) ────────► TRIG
    GPIO 6  (Pin 31) ◄──────── ECHO (use voltage divider!)
    5V               ────────► VCC
    GND              ────────► GND

    ⚠️  IMPORTANT: HC-SR04 ECHO pin outputs 5V!
        Use a voltage divider (1kΩ + 2kΩ) to protect the Pi's 3.3V GPIO!
        
                 ECHO ────┬──── 1kΩ ────┬──── GPIO (3.3V safe)
                          │             │
                         2kΩ           GND
                          │
                         GND


CAMERA CONNECTION:
==================

    Pi Camera Module:
    ────────────────
    Connect ribbon cable to CSI port (between HDMI and audio jack)
    
    USB Webcam:
    ──────────
    Connect to any USB port


Authors: VectorVance Team
Date: 2024
"""

import time
import threading

# ══════════════════════════════════════════════════════════════════════════════
# GPIO SETUP - Detect if running on Raspberry Pi or Windows/other
# ══════════════════════════════════════════════════════════════════════════════

try:
    import RPi.GPIO as GPIO
    PI_AVAILABLE = True
    print("✅ Raspberry Pi GPIO detected - Hardware mode")
except ImportError:
    PI_AVAILABLE = False
    print("⚠️  RPi.GPIO not found - Running in SIMULATION mode (Windows/Mac)")
    print("   Motor commands will be printed but not executed.")
    
    # Mock GPIO class for testing on non-Pi systems
    class MockGPIO:
        BCM = "BCM"
        OUT = "OUT"
        IN = "IN"
        LOW = 0
        HIGH = 1
        
        @staticmethod
        def setmode(mode): 
            pass
        
        @staticmethod
        def setwarnings(flag): 
            pass
        
        @staticmethod
        def setup(pin, mode): 
            pass
        
        @staticmethod
        def output(pin, state): 
            pass
        
        @staticmethod
        def input(pin): 
            return 0
        
        @staticmethod
        def cleanup(): 
            pass
        
        class PWM:
            def __init__(self, pin, freq):
                self.pin = pin
                self.freq = freq
                self.dc = 0
            
            def start(self, dc): 
                self.dc = dc
            
            def ChangeDutyCycle(self, dc): 
                self.dc = dc
            
            def stop(self): 
                pass
    
    GPIO = MockGPIO()


class MotorController:
    """
    L298N Motor Driver Controller
    
    Controls two DC motors using PWM for speed control and
    direction pins for forward/reverse.
    
    Pin Configuration:
        Left Motor:  IN1=GPIO17, IN2=GPIO18, ENA=GPIO12 (PWM)
        Right Motor: IN3=GPIO22, IN4=GPIO23, ENB=GPIO13 (PWM)
    """
    
    def __init__(self, max_speed=100):
        """
        Initialize motor controller.
        
        Args:
            max_speed: Maximum PWM duty cycle (0-100), default 100
                      Use lower values (e.g., 80) for safety during testing
        """
        # Pin definitions
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

        # Configure direction pins as outputs
        GPIO.setup(self.LEFT_FORWARD, GPIO.OUT)
        GPIO.setup(self.LEFT_BACKWARD, GPIO.OUT)
        GPIO.setup(self.RIGHT_FORWARD, GPIO.OUT)
        GPIO.setup(self.RIGHT_BACKWARD, GPIO.OUT)
        GPIO.setup(self.LEFT_PWM_PIN, GPIO.OUT)
        GPIO.setup(self.RIGHT_PWM_PIN, GPIO.OUT)

        # Initialize PWM at 1kHz
        self.left_pwm = GPIO.PWM(self.LEFT_PWM_PIN, 1000)
        self.right_pwm = GPIO.PWM(self.RIGHT_PWM_PIN, 1000)
        self.left_pwm.start(0)
        self.right_pwm.start(0)
        
        print(f"✅ Motor Controller initialized (max PWM: {max_speed}%)")
        print(f"   Left:  GPIO {self.LEFT_FORWARD}/{self.LEFT_BACKWARD}, PWM: GPIO {self.LEFT_PWM_PIN}")
        print(f"   Right: GPIO {self.RIGHT_FORWARD}/{self.RIGHT_BACKWARD}, PWM: GPIO {self.RIGHT_PWM_PIN}")
        if not PI_AVAILABLE:
            print("   ⚠️  SIMULATION MODE - Motors will not actually move")
        
        # For simulation logging
        self.last_left = 0
        self.last_right = 0

    def set_speeds(self, left_speed, right_speed):
        """
        Set motor speeds for differential drive.
        
        Args:
            left_speed: -1.0 (full reverse) to +1.0 (full forward)
            right_speed: -1.0 (full reverse) to +1.0 (full forward)
        """
        # Clamp values
        left_speed = max(-1.0, min(1.0, left_speed))
        right_speed = max(-1.0, min(1.0, right_speed))
        
        # Simulation logging (only when values change significantly)
        if not PI_AVAILABLE:
            if abs(left_speed - self.last_left) > 0.05 or abs(right_speed - self.last_right) > 0.05:
                if left_speed == 0 and right_speed == 0:
                    pass  # Don't spam "stopped" messages
                else:
                    print(f"   [SIM] Motors: L={left_speed:+.2f} R={right_speed:+.2f}")
                self.last_left = left_speed
                self.last_right = right_speed

        # Left motor
        if left_speed > 0:
            GPIO.output(self.LEFT_FORWARD, GPIO.HIGH)
            GPIO.output(self.LEFT_BACKWARD, GPIO.LOW)
            duty_cycle = abs(left_speed) * self.max_speed
        elif left_speed < 0:
            GPIO.output(self.LEFT_FORWARD, GPIO.LOW)
            GPIO.output(self.LEFT_BACKWARD, GPIO.HIGH)
            duty_cycle = abs(left_speed) * self.max_speed
        else:
            GPIO.output(self.LEFT_FORWARD, GPIO.LOW)
            GPIO.output(self.LEFT_BACKWARD, GPIO.LOW)
            duty_cycle = 0
        self.left_pwm.ChangeDutyCycle(duty_cycle)

        # Right motor
        if right_speed > 0:
            GPIO.output(self.RIGHT_FORWARD, GPIO.HIGH)
            GPIO.output(self.RIGHT_BACKWARD, GPIO.LOW)
            duty_cycle = abs(right_speed) * self.max_speed
        elif right_speed < 0:
            GPIO.output(self.RIGHT_FORWARD, GPIO.LOW)
            GPIO.output(self.RIGHT_BACKWARD, GPIO.HIGH)
            duty_cycle = abs(right_speed) * self.max_speed
        else:
            GPIO.output(self.RIGHT_FORWARD, GPIO.LOW)
            GPIO.output(self.RIGHT_BACKWARD, GPIO.LOW)
            duty_cycle = 0
        self.right_pwm.ChangeDutyCycle(duty_cycle)

    def emergency_stop(self):
        """Immediately stop both motors."""
        self.set_speeds(0, 0)

    def cleanup(self):
        """Release GPIO resources."""
        self.emergency_stop()
        self.left_pwm.stop()
        self.right_pwm.stop()
        GPIO.cleanup()
        print("✅ Motor Controller cleaned up")


class UltrasonicSensor:
    """
    HC-SR04 Ultrasonic Distance Sensor
    
    Measures distance using ultrasonic pulses.
    Range: 2cm - 400cm
    """

    def __init__(self, trig_pin, echo_pin, name="Sensor"):
        """
        Initialize ultrasonic sensor.
        
        Args:
            trig_pin: GPIO pin for TRIG (output)
            echo_pin: GPIO pin for ECHO (input)
            name: Sensor name for debugging
        """
        self.trig_pin = trig_pin
        self.echo_pin = echo_pin
        self.name = name

        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.trig_pin, GPIO.OUT)
        GPIO.setup(self.echo_pin, GPIO.IN)

        GPIO.output(self.trig_pin, GPIO.LOW)
        time.sleep(0.1)
        
        print(f"✅ Ultrasonic Sensor '{name}' initialized (TRIG: GPIO{trig_pin}, ECHO: GPIO{echo_pin})")
        if not PI_AVAILABLE:
            print(f"   ⚠️  SIMULATION MODE - Will return fake distances")

    def get_distance(self):
        """
        Measure distance in centimeters.
        
        Returns:
            Distance in cm (2.0 - 400.0), or 999.9 on timeout
        """
        # Simulation mode - return safe distance
        if not PI_AVAILABLE:
            # Return random-ish safe distance for testing
            import random
            return round(random.uniform(80, 150), 1)
        
        # Real hardware mode
        # Send 10µs trigger pulse
        GPIO.output(self.trig_pin, GPIO.HIGH)
        time.sleep(0.00001)
        GPIO.output(self.trig_pin, GPIO.LOW)

        timeout = 0.05
        start_time = time.time()

        # Wait for echo to go HIGH
        pulse_start = time.time()
        while GPIO.input(self.echo_pin) == 0:
            pulse_start = time.time()
            if pulse_start - start_time > timeout:
                return 999.9

        # Wait for echo to go LOW
        pulse_end = time.time()
        while GPIO.input(self.echo_pin) == 1:
            pulse_end = time.time()
            if pulse_end - pulse_start > timeout:
                return 999.9

        # Calculate distance: time × speed of sound / 2
        pulse_duration = pulse_end - pulse_start
        distance = (pulse_duration * 34300) / 2

        return round(max(2.0, min(400.0, distance)), 1)


class HardwareObstacleDetector:
    """
    Dual Ultrasonic Sensor Obstacle Detector
    
    Monitors both front sensors in a background thread and provides
    speed modifiers and emergency stop signals.
    
    Sensor Configuration:
        Front-Left:  TRIG=GPIO24, ECHO=GPIO25
        Front-Right: TRIG=GPIO5,  ECHO=GPIO6
    """

    def __init__(self, emergency_distance=20, warning_distance=50):
        """
        Initialize obstacle detector.
        
        Args:
            emergency_distance: Distance (cm) that triggers emergency stop
            warning_distance: Distance (cm) that triggers slow-down
        """
        self.emergency_distance = emergency_distance
        self.warning_distance = warning_distance

        # Initialize sensors
        self.sensor_left = UltrasonicSensor(24, 25, "Front-Left")
        self.sensor_right = UltrasonicSensor(5, 6, "Front-Right")

        # Current readings
        self.left_distance = 400
        self.right_distance = 400
        self.emergency_stop_triggered = False
        self.warning_triggered = False

        # Start background monitoring thread
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        print(f"✅ Obstacle Detector initialized")
        print(f"   Emergency stop: <{emergency_distance}cm")
        print(f"   Warning zone:   <{warning_distance}cm")

    def _monitor_loop(self):
        """Background thread that continuously reads sensors."""
        while self.monitoring:
            self.left_distance = self.sensor_left.get_distance()
            self.right_distance = self.sensor_right.get_distance()

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
        """Returns True if obstacle is dangerously close."""
        return self.emergency_stop_triggered

    def should_slow_down(self):
        """Returns True if obstacle is in warning zone."""
        return self.warning_triggered

    def get_speed_modifier(self):
        """
        Get speed multiplier based on obstacle proximity.
        
        Returns:
            0.0 (stop) to 1.0 (full speed)
        """
        if self.emergency_stop_triggered:
            return 0.0

        min_distance = min(self.left_distance, self.right_distance)

        if min_distance < self.warning_distance:
            # Linear scale: emergency → 0.3, warning → 1.0
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
        """Stop the background monitoring thread."""
        self.monitoring = False
        if self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=1.0)
        print("✅ Obstacle Detector stopped")


# ══════════════════════════════════════════════════════════════════════════════
# HARDWARE TEST FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def test_motors():
    """Test motor functionality."""
    print("\n" + "="*50)
    print("🧪 MOTOR TEST")
    print("="*50)
    
    motors = MotorController(max_speed=50)  # 50% for safety

    try:
        print("\n▶️ Forward (2s)...")
        motors.set_speeds(0.5, 0.5)
        time.sleep(2)

        print("◀️ Backward (2s)...")
        motors.set_speeds(-0.5, -0.5)
        time.sleep(2)

        print("↩️ Turn left (2s)...")
        motors.set_speeds(0.3, 0.7)
        time.sleep(2)

        print("↪️ Turn right (2s)...")
        motors.set_speeds(0.7, 0.3)
        time.sleep(2)

        print("⏹️ Stop")
        motors.emergency_stop()
        
        print("\n✅ Motor test complete!")
        
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted")
    finally:
        motors.cleanup()


def test_ultrasonic():
    """Test ultrasonic sensors."""
    print("\n" + "="*50)
    print("🧪 ULTRASONIC SENSOR TEST")
    print("="*50)
    
    detector = HardwareObstacleDetector()

    try:
        print("\nReading sensors for 10 seconds...")
        print("Move objects in front of sensors to test.\n")
        
        for i in range(20):
            distances = detector.get_distances()
            
            status = ""
            if detector.should_emergency_stop():
                status = "🚨 EMERGENCY STOP!"
            elif detector.should_slow_down():
                status = "⚠️ SLOWING DOWN"
            else:
                status = "✅ CLEAR"
            
            print(f"Left: {distances['left']:6.1f}cm | "
                  f"Right: {distances['right']:6.1f}cm | "
                  f"Min: {distances['min']:6.1f}cm | "
                  f"{status}")

            time.sleep(0.5)
            
        print("\n✅ Ultrasonic test complete!")
        
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted")
    finally:
        detector.stop_monitoring()
        GPIO.cleanup()


def print_wiring_diagram():
    """Print the wiring diagram."""
    print(__doc__)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN - Hardware Test Menu
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "="*60)
    print("   🔧 VECTORVANCE HARDWARE TEST SUITE")
    print("="*60)
    
    if not PI_AVAILABLE:
        print("\n   ⚠️  SIMULATION MODE ACTIVE")
        print("   Running on Windows/Mac - hardware will be simulated")
        print("   Run on Raspberry Pi for real hardware control")
    
    print("\nOptions:")
    print("  1. Test Motors")
    print("  2. Test Ultrasonic Sensors")
    print("  3. Print Wiring Diagram")
    print("  4. Exit")
    print("="*60)

    try:
        choice = input("\nSelect test (1-4): ").strip()

        if choice == "1":
            test_motors()
        elif choice == "2":
            test_ultrasonic()
        elif choice == "3":
            print_wiring_diagram()
        else:
            print("Exiting...")
            
    except KeyboardInterrupt:
        print("\n\nExiting...")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        GPIO.cleanup()