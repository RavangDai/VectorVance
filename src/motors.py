"""
VectorVance - Dual L298N Motor Driver Module
Raspberry Pi 3 | gpiozero library | 4WD individual motor control

Wiring - L298N #1 (LEFT side):
  ENA  -> GPIO18 (Pin 12)  PWM - Front Left speed
  IN1  -> GPIO23 (Pin 16)  Front Left direction
  IN2  -> GPIO24 (Pin 18)  Front Left direction
  ENB  -> GPIO13 (Pin 33)  PWM - Rear Left speed
  IN3  -> GPIO25 (Pin 22)  Rear Left direction
  IN4  -> GPIO8  (Pin 24)  Rear Left direction

Wiring - L298N #2 (RIGHT side):
  ENA  -> GPIO12 (Pin 32)  PWM - Front Right speed
  IN1  -> GPIO16 (Pin 36)  Front Right direction
  IN2  -> GPIO20 (Pin 38)  Front Right direction
  ENB  -> GPIO19 (Pin 35)  PWM - Rear Right speed
  IN3  -> GPIO21 (Pin 40)  Rear Right direction
  IN4  -> GPIO26 (Pin 37)  Rear Right direction

  GND  -> Pin 6  (shared between BOTH L298N boards and Pi)

Usage:
  car = MotorController()
  car.forward(speed=0.6)
  car.turn_left()
  car.stop()
  car.cleanup()
"""

from gpiozero import Motor, PWMOutputDevice
from time import sleep


class MotorController:
    def __init__(
        self,
        # L298N #1 — LEFT side
        fl_in1=23, fl_in2=24, fl_ena=18,   # Front Left
        rl_in3=25, rl_in4=8,  rl_enb=13,   # Rear Left

        # L298N #2 — RIGHT side
        fr_in1=16, fr_in2=20, fr_ena=12,   # Front Right
        rr_in3=21, rr_in4=26, rr_enb=19,   # Rear Right

        default_speed=0.5
    ):
        # Individual motors
        self.front_left  = Motor(forward=fl_in1, backward=fl_in2)
        self.rear_left   = Motor(forward=rl_in3, backward=rl_in4)
        self.front_right = Motor(forward=fr_in1, backward=fr_in2)
        self.rear_right  = Motor(forward=rr_in3, backward=rr_in4)

        # PWM speed pins (0.0 - 1.0)
        self.ena_fl = PWMOutputDevice(fl_ena, initial_value=1)
        self.enb_rl = PWMOutputDevice(rl_enb, initial_value=1)
        self.ena_fr = PWMOutputDevice(fr_ena, initial_value=1)
        self.enb_rr = PWMOutputDevice(rr_enb, initial_value=1)

        self.default_speed = default_speed
        print("[Motors] Dual L298N initialised OK — all 4 motors ready")

    # ── internal helpers ───────────────────────────────────────────────────────

    def _set_left_speed(self, speed):
        v = max(0.0, min(1.0, speed))
        self.ena_fl.value = v
        self.enb_rl.value = v

    def _set_right_speed(self, speed):
        v = max(0.0, min(1.0, speed))
        self.ena_fr.value = v
        self.enb_rr.value = v

    def _drive_motor(self, motor, pwm, value):
        pwm.value = abs(value)
        if value > 0.02:
            motor.forward()
        elif value < -0.02:
            motor.backward()
        else:
            motor.stop()

    @property
    def _all_motors(self):
        return [self.front_left, self.rear_left,
                self.front_right, self.rear_right]

    @property
    def _all_pwm(self):
        return [self.ena_fl, self.enb_rl, self.ena_fr, self.enb_rr]

    # ── basic movements ────────────────────────────────────────────────────────

    def forward(self, speed=None, duration=None):
        speed = speed or self.default_speed
        self._set_left_speed(speed)
        self._set_right_speed(speed)
        for m in self._all_motors:
            m.forward()
        if duration:
            sleep(duration)
            self.stop()

    def backward(self, speed=None, duration=None):
        speed = speed or self.default_speed
        self._set_left_speed(speed)
        self._set_right_speed(speed)
        for m in self._all_motors:
            m.backward()
        if duration:
            sleep(duration)
            self.stop()

    def stop(self):
        for m in self._all_motors:
            m.stop()

    def brake(self):
        self._set_left_speed(1.0)
        self._set_right_speed(1.0)
        for m in self._all_motors:
            m.forward()
        sleep(0.05)
        self.stop()

    # ── turning ────────────────────────────────────────────────────────────────

    def turn_left(self, speed=None, duration=None):
        speed = speed or self.default_speed
        self._set_left_speed(speed * 0.3)
        self._set_right_speed(speed)
        for m in self._all_motors:
            m.forward()
        if duration:
            sleep(duration)
            self.stop()

    def turn_right(self, speed=None, duration=None):
        speed = speed or self.default_speed
        self._set_left_speed(speed)
        self._set_right_speed(speed * 0.3)
        for m in self._all_motors:
            m.forward()
        if duration:
            sleep(duration)
            self.stop()

    def spin_left(self, speed=None, duration=None):
        speed = speed or self.default_speed
        self._set_left_speed(speed)
        self._set_right_speed(speed)
        self.front_left.backward()
        self.rear_left.backward()
        self.front_right.forward()
        self.rear_right.forward()
        if duration:
            sleep(duration)
            self.stop()

    def spin_right(self, speed=None, duration=None):
        speed = speed or self.default_speed
        self._set_left_speed(speed)
        self._set_right_speed(speed)
        self.front_left.forward()
        self.rear_left.forward()
        self.front_right.backward()
        self.rear_right.backward()
        if duration:
            sleep(duration)
            self.stop()

    # ── differential steering (used by autonomy logic) ─────────────────────────

    def steer(self, throttle=0.5, steering=0.0):
        """
        throttle  : -1.0 (full back) to +1.0 (full forward)
        steering  : -1.0 (full left) to +1.0 (full right)
        """
        left  = throttle - steering
        right = throttle + steering
        scale = max(abs(left), abs(right), 1.0)
        left  /= scale
        right /= scale

        self._drive_motor(self.front_left,  self.ena_fl, left)
        self._drive_motor(self.rear_left,   self.enb_rl, left)
        self._drive_motor(self.front_right, self.ena_fr, right)
        self._drive_motor(self.rear_right,  self.enb_rr, right)

    # ── individual motor test ─────────────────────────────────────────────────

    def test_individual(self, speed=0.4, duration=1.0):
        """Spin each motor one at a time — run this first to verify wiring."""
        motors = [
            ("Front Left",  self.front_left,  self.ena_fl),
            ("Rear Left",   self.rear_left,   self.enb_rl),
            ("Front Right", self.front_right, self.ena_fr),
            ("Rear Right",  self.rear_right,  self.enb_rr),
        ]
        for name, motor, pwm in motors:
            print(f"  Testing: {name}")
            pwm.value = speed
            motor.forward()
            sleep(duration)
            motor.stop()
            sleep(0.3)
        print("  Individual test complete.")

    # ── cleanup ────────────────────────────────────────────────────────────────

    def cleanup(self):
        self.stop()
        for m in self._all_motors:
            m.close()
        for p in self._all_pwm:
            p.close()
        print("[Motors] All GPIOs released")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.cleanup()


# ── quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("VectorVance — Dual L298N 4WD motor test")
    try:
        with MotorController(default_speed=0.5) as car:

            print("\n--- Step 1: Individual motor test (verify wiring) ---")
            car.test_individual(speed=0.4, duration=0.8)
            sleep(0.5)

            print("\n--- Step 2: Movement test ---")
            print("Forward 2s...")
            car.forward(duration=2)
            sleep(0.4)

            print("Turn left 1s...")
            car.turn_left(duration=1)
            sleep(0.4)

            print("Turn right 1s...")
            car.turn_right(duration=1)
            sleep(0.4)

            print("Spin left 1s...")
            car.spin_left(duration=1)
            sleep(0.4)

            print("Spin right 1s...")
            car.spin_right(duration=1)
            sleep(0.4)

            print("Backward 1s...")
            car.backward(duration=1)

            print("\nAll tests passed!")

    except KeyboardInterrupt:
        print("\nInterrupted — GPIOs cleaned up.")