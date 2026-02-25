"""
Ultrasonic Sensor HC-SR04 - Fixed for Raspberry Pi OS Bookworm
Wiring:
  VCC (Red)    → Pin 2 (5V)
  GND (White)  → Pin 34 (GND) - with resistor for voltage divider
  TRIG (Green) → Pin 7 (GPIO 4)
  ECHO (Orange)→ Pin 11 (GPIO 17)
"""

import time
import lgpio

# Open GPIO chip
h = lgpio.gpiochip_open(0)

# Define GPIO pins (BCM numbering)
GPIO_TRIGGER = 4   # Pin 7
GPIO_ECHO = 17     # Pin 11

# Setup pins
lgpio.gpio_claim_output(h, GPIO_TRIGGER)
lgpio.gpio_claim_input(h, GPIO_ECHO)

# Set trigger to low initially
lgpio.gpio_write(h, GPIO_TRIGGER, 0)
time.sleep(0.1)

def measure_distance():
    """Measure distance in centimeters."""
    # Send 10us trigger pulse
    lgpio.gpio_write(h, GPIO_TRIGGER, 1)
    time.sleep(0.00001)  # 10 microseconds
    lgpio.gpio_write(h, GPIO_TRIGGER, 0)
    
    # Wait for echo to start (go HIGH)
    timeout = time.time() + 0.1
    start = time.time()
    
    while lgpio.gpio_read(h, GPIO_ECHO) == 0:
        start = time.time()
        if start > timeout:
            return -1  # Timeout
    
    # Wait for echo to end (go LOW)
    stop = time.time()
    while lgpio.gpio_read(h, GPIO_ECHO) == 1:
        stop = time.time()
        if stop > timeout:
            return -1  # Timeout
    
    # Calculate distance
    # Speed of sound = 34300 cm/s at ~20°C
    time_diff = stop - start
    distance = (time_diff * 34300) / 2
    
    return round(distance, 2)

# Main loop
print("=" * 50)
print("🔊 ULTRASONIC SENSOR TEST")
print("=" * 50)
print("Move your hand in front of the sensor...")
print("Press Ctrl+C to stop")
print("=" * 50)

try:
    while True:
        distance = measure_distance()
        
        if distance < 0:
            print("⚠️  Timeout - no echo received")
        elif distance < 20:
            print(f"🔴 Distance: {distance:6.2f} cm  - DANGER!")
        elif distance < 50:
            print(f"🟡 Distance: {distance:6.2f} cm  - Warning")
        else:
            print(f"🟢 Distance: {distance:6.2f} cm  - Clear")
        
        time.sleep(0.5)

except KeyboardInterrupt:
    print("\n\n✅ Test stopped by user")

finally:
    lgpio.gpiochip_close(h)
    print("✅ GPIO cleaned up")