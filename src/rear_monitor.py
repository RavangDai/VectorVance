"""
rear_monitor.py — Rear ultrasonic collision monitor + alert notifications
──────────────────────────────────────────────────────────────────────────
Monitors the HC-SR04 at the rear of the car in a background thread.
When something gets within COLLISION_DISTANCE cm of the rear, it fires an
alert via one or more of:
  • Telegram bot  (free, works on Pi, recommended)
  • SMS via Twilio (requires paid account)
  • Email via SMTP (e.g. Gmail app password)

Quick setup — Telegram (easiest):
  1. Message @BotFather on Telegram → /newbot → copy the BOT_TOKEN
  2. Send any message to your new bot, then open:
       https://api.telegram.org/bot<BOT_TOKEN>/getUpdates
     and copy your numeric chat_id from the response
  3. Fill in TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID below

GPIO: TRIG=4, ECHO=17  (same pins as before, now rear-facing)
"""

import time
import threading
import lgpio

# ── Notification credentials — fill in at least ONE ───────────────────────────

TELEGRAM_BOT_TOKEN = ""   # "123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11"
TELEGRAM_CHAT_ID   = ""   # "987654321"

TWILIO_ACCOUNT_SID = ""   # "ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
TWILIO_AUTH_TOKEN  = ""   # "your_auth_token"
TWILIO_FROM_NUMBER = ""   # "+15551234567"
TWILIO_TO_NUMBER   = ""   # "+15557654321"

SMTP_HOST     = "smtp.gmail.com"
SMTP_PORT     = 587
SMTP_USER     = ""        # "youraddress@gmail.com"
SMTP_PASSWORD = ""        # Gmail app password (not your account password)
SMTP_TO       = ""        # "recipient@example.com"


class RearMonitor:
    """
    Background thread that reads the rear HC-SR04 and fires an alert
    whenever a rear collision (or close approach) is detected.

    Usage:
        monitor = RearMonitor(trig=4, echo=17)
        monitor.start()
        ...
        status = monitor.get_status()   # call from main loop
        monitor.stop()                  # call on shutdown
    """

    COLLISION_DISTANCE = 20    # cm — trigger alert below this
    COOLDOWN_SECONDS   = 30    # minimum seconds between repeated alerts
    POLL_INTERVAL      = 0.10  # seconds between sensor reads
    MAX_RANGE          = 300   # cm — readings above this ignored (no echo)

    def __init__(self, trig: int = 4, echo: int = 17):
        self._trig = trig
        self._echo = echo

        self._gpio = lgpio.gpiochip_open(0)
        lgpio.gpio_claim_output(self._gpio, trig)
        lgpio.gpio_claim_input(self._gpio, echo)
        lgpio.gpio_write(self._gpio, trig, 0)
        time.sleep(0.05)

        self._distance    = float(self.MAX_RANGE)
        self._alert_count = 0
        self._last_alert  = 0.0

        self._lock   = threading.Lock()
        self._stop   = threading.Event()
        self._thread = None

    # ── Public API ─────────────────────────────────────────────────────────────

    def start(self):
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="RearMonitor")
        self._thread.start()
        print(f"[RearMonitor] Started  TRIG={self._trig}  ECHO={self._echo}  "
              f"alert < {self.COLLISION_DISTANCE} cm")

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        try:
            lgpio.gpiochip_close(self._gpio)
        except Exception:
            pass
        print("[RearMonitor] Stopped")

    @property
    def distance_cm(self) -> float:
        with self._lock:
            return self._distance

    def get_status(self) -> dict:
        with self._lock:
            return {
                "rear_distance_cm": round(self._distance, 1),
                "rear_alert_count": self._alert_count,
            }

    # ── Internal loop ──────────────────────────────────────────────────────────

    def _loop(self):
        while not self._stop.is_set():
            dist = self._read_sensor()
            with self._lock:
                self._distance = dist

            now = time.time()
            if dist < self.COLLISION_DISTANCE:
                if now - self._last_alert > self.COOLDOWN_SECONDS:
                    with self._lock:
                        self._last_alert   = now
                        self._alert_count += 1
                        count = self._alert_count
                    print(f"[RearMonitor] COLLISION DETECTED  {dist:.1f} cm  "
                          f"(alert #{count})")
                    self._send_alert(dist, count)

            time.sleep(self.POLL_INTERVAL)

    def _read_sensor(self) -> float:
        try:
            lgpio.gpio_write(self._gpio, self._trig, 1)
            time.sleep(0.00001)
            lgpio.gpio_write(self._gpio, self._trig, 0)

            deadline = time.time() + 0.04
            start = time.time()
            while lgpio.gpio_read(self._gpio, self._echo) == 0:
                start = time.time()
                if time.time() > deadline:
                    return float(self.MAX_RANGE)

            stop = time.time()
            deadline = time.time() + 0.04
            while lgpio.gpio_read(self._gpio, self._echo) == 1:
                stop = time.time()
                if time.time() > deadline:
                    return float(self.MAX_RANGE)

            return round((stop - start) * 34300 / 2, 1)
        except Exception:
            return float(self.MAX_RANGE)

    # ── Alert dispatch ─────────────────────────────────────────────────────────

    def _send_alert(self, dist_cm: float, count: int):
        msg = (
            f"VectorVance REAR COLLISION ALERT\n"
            f"Distance: {dist_cm:.1f} cm\n"
            f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Alert #{count}"
        )
        if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
            self._send_telegram(msg)
        if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN:
            self._send_twilio_sms(msg)
        if SMTP_USER and SMTP_PASSWORD and SMTP_TO:
            self._send_email(msg)

    def _send_telegram(self, text: str):
        try:
            import urllib.request, urllib.parse
            url     = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            payload = urllib.parse.urlencode({
                "chat_id": TELEGRAM_CHAT_ID,
                "text":    text,
            }).encode()
            with urllib.request.urlopen(url, data=payload, timeout=10):
                pass
            print("[RearMonitor] Telegram alert sent")
        except Exception as e:
            print(f"[RearMonitor] Telegram failed: {e}")

    def _send_twilio_sms(self, text: str):
        try:
            from twilio.rest import Client
            Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN).messages.create(
                body=text,
                from_=TWILIO_FROM_NUMBER,
                to=TWILIO_TO_NUMBER,
            )
            print("[RearMonitor] Twilio SMS sent")
        except Exception as e:
            print(f"[RearMonitor] Twilio failed: {e}")

    def _send_email(self, text: str):
        try:
            import smtplib
            from email.mime.text import MIMEText
            msg           = MIMEText(text)
            msg["Subject"] = "VectorVance Rear Collision Alert"
            msg["From"]    = SMTP_USER
            msg["To"]      = SMTP_TO
            with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as s:
                s.starttls()
                s.login(SMTP_USER, SMTP_PASSWORD)
                s.send_message(msg)
            print("[RearMonitor] Email alert sent")
        except Exception as e:
            print(f"[RearMonitor] Email failed: {e}")
