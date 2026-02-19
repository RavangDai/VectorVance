"""
hardware.py - GPIO motor and sensor control for Raspberry Pi.

Wiring summary:
  L298N motor driver: GPIO 17/18 (left fwd/back), 22/23 (right fwd/back), 12/13 (PWM)
  HC-SR04 front-left: TRIG=24, ECHO=25
  HC-SR04 front-right: TRIG=5,  ECHO=6
"""

import RPi.GPIO as GPIO
import time
import threading


class MotorController:
    def __init__(self, max_speed=100):
        self.LEFT_FORWARD = 17
        self.LEFT_BACKWARD = 18
        self.RIGHT_FORWARD = 22
        self.RIGHT_BACKWARD = 23
        self.LEFT_PWM_PIN = 12
        self.RIGHT_PWM_PIN = 13

        self.max_speed = max_speed

        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)

        GPIO.setup(self.LEFT_FORWARD, GPIO.OUT)
        GPIO.setup(self.LEFT_BACKWARD, GPIO.OUT)
        GPIO.setup(self.RIGHT_FORWARD, GPIO.OUT)
        GPIO.setup(self.RIGHT_BACKWARD, GPIO.OUT)
        GPIO.setup(self.LEFT_PWM_PIN, GPIO.OUT)
        GPIO.setup(self.RIGHT_PWM_PIN, GPIO.OUT)

        self.left_pwm = GPIO.PWM(self.LEFT_PWM_PIN, 1000)
        self.right_pwm = GPIO.PWM(self.RIGHT_PWM_PIN, 1000)
        self.left_pwm.start(0)
        self.right_pwm.start(0)

    def set_speeds(self, left_speed, right_speed):
        """
        Set motor speeds. Both values in range -1.0 (full reverse) to +1.0 (full forward).
        """
        left_speed = max(-1.0, min(1.0, left_speed))
        right_speed = max(-1.0, min(1.0, right_speed))

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
        self.set_speeds(0, 0)

    def cleanup(self):
        self.emergency_stop()
        self.left_pwm.stop()
        self.right_pwm.stop()
        GPIO.cleanup()


class UltrasonicSensor:
    """HC-SR04 ultrasonic distance sensor."""

    def __init__(self, trig_pin, echo_pin, name="Sensor"):
        self.trig_pin = trig_pin
        self.echo_pin = echo_pin
        self.name = name

        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.trig_pin, GPIO.OUT)
        GPIO.setup(self.echo_pin, GPIO.IN)

        GPIO.output(self.trig_pin, GPIO.LOW)
        time.sleep(0.1)

    def get_distance(self):
        """Returns distance in cm, or 999.9 on timeout."""
        GPIO.output(self.trig_pin, GPIO.HIGH)
        time.sleep(0.00001)
        GPIO.output(self.trig_pin, GPIO.LOW)

        timeout = 0.05
        start_time = time.time()

        pulse_start = time.time()
        while GPIO.input(self.echo_pin) == 0:
            pulse_start = time.time()
            if pulse_start - start_time > timeout:
                return 999.9

        pulse_end = time.time()
        while GPIO.input(self.echo_pin) == 1:
            pulse_end = time.time()
            if pulse_end - pulse_start > timeout:
                return 999.9

        pulse_duration = pulse_end - pulse_start
        distance = (pulse_duration * 34300) / 2

        return round(max(2.0, min(400.0, distance)), 1)


class HardwareObstacleDetector:
    """
    Reads both front sensors in a background thread and tracks
    whether we need to slow down or emergency stop.
    """

    def __init__(self, emergency_distance=20, warning_distance=50):
        self.emergency_distance = emergency_distance
        self.warning_distance = warning_distance

        self.sensor_left = UltrasonicSensor(24, 25, "Front-Left")
        self.sensor_right = UltrasonicSensor(5, 6, "Front-Right")

        self.left_distance = 400
        self.right_distance = 400
        self.emergency_stop_triggered = False
        self.warning_triggered = False

        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

    def _monitor_loop(self):
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

            time.sleep(0.05)  # 20 Hz

    def should_emergency_stop(self):
        return self.emergency_stop_triggered

    def should_slow_down(self):
        return self.warning_triggered

    def get_speed_modifier(self):
        """Returns 0.0 (stop) to 1.0 (full speed) based on nearest obstacle."""
        if self.emergency_stop_triggered:
            return 0.0

        min_distance = min(self.left_distance, self.right_distance)

        if min_distance < self.warning_distance:
            # linear scale between emergency and warning thresholds, floor at 0.3
            ratio = (min_distance - self.emergency_distance) / \
                   (self.warning_distance - self.emergency_distance)
            return max(0.3, min(1.0, ratio))

        return 1.0

    def get_distances(self):
        return {
            'left': self.left_distance,
            'right': self.right_distance,
            'min': min(self.left_distance, self.right_distance)
        }

    def stop_monitoring(self):
        self.monitoring = False
        if self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=1.0)


# ── Quick hardware tests ──────────────────────────────────────────────────────

def test_motors():
    print("\nTesting motors...")
    motors = MotorController(max_speed=50)

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

        motors.emergency_stop()
    finally:
        motors.cleanup()


def test_ultrasonic():
    print("\nTesting ultrasonic sensors...")
    detector = HardwareObstacleDetector()

    try:
        for i in range(20):
            distances = detector.get_distances()
            print(f"Left: {distances['left']:6.1f}cm | "
                  f"Right: {distances['right']:6.1f}cm | "
                  f"Min: {distances['min']:6.1f}cm")

            if detector.should_emergency_stop():
                print("EMERGENCY STOP!")
            elif detector.should_slow_down():
                print("SLOWING DOWN")

            time.sleep(0.5)
    finally:
        detector.stop_monitoring()
        GPIO.cleanup()


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
