import RPi.GPIO as GPIO
import time

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)

# Motor Driver 1 — front left, front right
IN1, IN2 = 25, 27
IN3, IN4 = 5, 13

# Motor Driver 2 — rear left, rear right
IN5, IN6 = 26, 20
IN7, IN8 = 16, 6

# Ultrasonic sensor
TRIG, ECHO = 4, 17

STOP_DISTANCE = 30     # cm — stop if object closer than this

ALL = [IN1, IN2, IN3, IN4, IN5, IN6, IN7, IN8]
for p in ALL:
    GPIO.setup(p, GPIO.OUT)
    GPIO.output(p, GPIO.LOW)

GPIO.setup(TRIG, GPIO.OUT)
GPIO.setup(ECHO, GPIO.IN)
GPIO.output(TRIG, False)
time.sleep(0.5)

# ── Movement functions ──────────────────────────────────

def forward():
    GPIO.output(IN1, GPIO.HIGH); GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.HIGH); GPIO.output(IN4, GPIO.LOW)
    GPIO.output(IN5, GPIO.HIGH); GPIO.output(IN6, GPIO.LOW)
    GPIO.output(IN7, GPIO.HIGH); GPIO.output(IN8, GPIO.LOW)

def stop_all():
    for p in ALL:
        GPIO.output(p, GPIO.LOW)

# ── Ultrasonic distance ─────────────────────────────────

def get_distance():
    GPIO.output(TRIG, True)
    time.sleep(0.00001)
    GPIO.output(TRIG, False)

    timeout = time.time() + 0.04
    start = time.time()
    while GPIO.input(ECHO) == 0:
        start = time.time()
        if time.time() > timeout:
            return 999

    timeout = time.time() + 0.04
    stop = time.time()
    while GPIO.input(ECHO) == 1:
        stop = time.time()
        if time.time() > timeout:
            return 999

    dist = (stop - start) * 34300 / 2
    return round(dist, 1)

# ── Main loop ───────────────────────────────────────────

print("Smart car started. Press Ctrl+C to stop.")
print(f"Stop distance: {STOP_DISTANCE} cm")

try:
    while True:
        dist = get_distance()
        print(f"Distance: {dist} cm")

        if dist < STOP_DISTANCE:
            stop_all()
            print("OBSTACLE DETECTED — waiting for path to clear...")
            while get_distance() < STOP_DISTANCE:
                print(f"  Still blocked: {get_distance()} cm — waiting...")
                time.sleep(0.3)
            print("Path clear — resuming!")
        else:
            forward()

        time.sleep(0.1)

except KeyboardInterrupt:
    print("\nStopped by user.")

finally:
    stop_all()
    GPIO.cleanup()
    print("Cleaned up. Bye!")