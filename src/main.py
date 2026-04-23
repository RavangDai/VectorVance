"""
main.py - VectorVance Autonomous Car (PRODUCTION)
──────────────────────────────────────────────────
Hardware : Innomaker 1080P USB2.0 UVC 130° wide-angle camera (front-mounted, upside-down) + dual L298N motors + HC-SR04 ultrasonic (front-facing)
Detection: MobileNet SSD v2 (always active)
Control  : PID lane-follow + adaptive speed + obstacle avoidance
Dashboard: Live web UI at http://<pi-ip>:5000/

USAGE:
  python main.py                    # full autonomous (MobileNet SSD always on)
  python main.py --no-web           # disable web dashboard
  python main.py --no-display       # headless (no cv2 window)

KEYBOARD (when display is on):
  Q        Quit
  SPACE    Toggle autonomous mode
  R        Reset all systems
  S        Save snapshot
  D        Print detector debug info
"""

import cv2
import time
import argparse
import threading
import lgpio
import os

# Load .env from the project root (one level above src/)
def _load_dotenv():
    env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
    env_path = os.path.normpath(env_path)
    if not os.path.isfile(env_path):
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            key, _, val = line.partition('=')
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            os.environ.setdefault(key, val)

_load_dotenv()

from gpiozero import Motor

from camera import FisheyeCamera
from perception import LaneDetector, SmoothValue
from controller import PIDController
from speed_controller import AdaptiveSpeedController, draw_speed_indicator
from safety import ObstacleDetector
from dnn_detector import StopSignConfirmer, load_ssd_net
from sign_detector import TrafficSignDetector, ArucoSpeedDetector, ARUCO_SPEED_MAP
from item_tracker import ItemTracker, TRACK_TARGETS
from sentry import SentryMonitor
import pi_server

# Speed limit → motor fraction (mirrors ArucoSpeedDetector.SPEED_FRACTIONS)
SPEED_LIMIT_MAP = {10: 0.30, 20: 0.50, 30: 0.75, 50: 1.00}

# ── GPIO pin assignments ──────────────────────────────────────────────────────
FL_FWD, FL_BWD = 25, 27
FR_FWD, FR_BWD = 5,  15
RL_FWD, RL_BWD = 26, 20
RR_FWD, RR_BWD = 16, 6
TRIG_PIN, ECHO_PIN = 4, 17
STOP_DISTANCE = 20
SLOW_DISTANCE = 50

# ── Car state machine ─────────────────────────────────────────────────────────
class State:
    LANE_FOLLOW = "LANE_FOLLOW"   # normal driving
    FREE_ROAM   = "FREE_ROAM"     # FSD: obstacle-avoiding free roam, no lane
    TRACKING    = "TRACKING"      # TRACK: chase the locked COCO target
    SENTRY      = "SENTRY"        # SENTRY: stationary surveillance, motors off


# ── Front HC-SR04 ultrasonic sensor (background thread) ──────────────────────
class FrontSensor:
    """
    Reads the front-facing HC-SR04 in a background thread.
    distance_cm property is always safe to read from the main loop.
    """
    MAX_RANGE       = 300   # cm — treat readings above this as no-echo
    POLL_INTERVAL   = 0.10  # seconds between reads
    CAMERA_OFFSET   = 5.0   # cm — camera body sits this far in front of the sensor face

    def __init__(self, trig: int = 4, echo: int = 17):
        self._trig = trig
        self._echo = echo
        self._gpio = lgpio.gpiochip_open(0)
        lgpio.gpio_claim_output(self._gpio, trig)
        lgpio.gpio_claim_input(self._gpio, echo)
        lgpio.gpio_write(self._gpio, trig, 0)
        time.sleep(0.05)
        self._distance = float(self.MAX_RANGE)
        self._lock   = threading.Lock()
        self._stop   = threading.Event()
        self._thread = None

    def start(self):
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="FrontSensor")
        self._thread.start()
        print(f"[FrontSensor] Started  TRIG={self._trig}  ECHO={self._echo}  "
              f"(front-facing obstacle detection)")

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        try:
            lgpio.gpiochip_close(self._gpio)
        except Exception:
            pass
        print("[FrontSensor] Stopped")

    @property
    def distance_cm(self) -> float:
        with self._lock:
            raw = self._distance
        if raw <= self.CAMERA_OFFSET:
            return self.MAX_RANGE
        return raw - self.CAMERA_OFFSET

    def _loop(self):
        while not self._stop.is_set():
            dist = self._read()
            with self._lock:
                self._distance = dist
            time.sleep(self.POLL_INTERVAL)

    def _read(self) -> float:
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


# ── Camera grabber (background thread, latest-frame policy) ──────────────────
class CameraGrabber:
    """
    Drains cv2.VideoCapture in a daemon thread and always exposes the most
    recent frame. Pi 3's USB camera read can cost 20-40 ms per call — doing
    it in-line on the perception loop starves PID + DNN of CPU time.
    latest() never blocks; it returns (ok, frame).
    """
    def __init__(self, cap):
        self._cap   = cap
        self._lock  = threading.Lock()
        self._frame = None
        self._stop  = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="CameraGrabber")

    def start(self):
        self._thread.start()
        print("[Camera] Grabber thread started")

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=1.5)

    def latest(self):
        with self._lock:
            if self._frame is None:
                return False, None
            # Hand out a view — perception is read-only; any module that
            # mutates must .copy() first (process_frame_* do this already).
            return True, self._frame

    def _loop(self):
        while not self._stop.is_set():
            ok, f = self._cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            with self._lock:
                self._frame = f


# ── FSD free-roam controller ──────────────────────────────────────────────────
class FreeRoamController:
    """
    Obstacle-avoiding controller for FSD mode.
    No lane detection — drives forward and steers away from obstacles.
    Uses DNN zone threats + ultrasonic for decisions.
    """
    BASE_SPEED    = 0.55
    TURN_SPEED    = 0.65
    COMMIT_FRAMES = 22   # hold a turn decision for ~0.8 s at 28 fps
    BACKUP_FRAMES = 12   # frames to reverse when ultrasonic triggers

    def __init__(self):
        self.turn_direction   = 0    # -1=left, 0=straight, +1=right
        self.committed_frames = 0
        self.backing_up       = 0

    def reset(self):
        self.turn_direction   = 0
        self.committed_frames = 0
        self.backing_up       = 0

    def compute(self, distance_cm: float):
        """
        Returns (left_speed, right_speed, status_str).
        Negative values = reverse — caller must use _drive_manual().
        Uses front ultrasonic distance only (no DNN obstacle zones).
        """
        # Backup countdown
        if self.backing_up > 0:
            self.backing_up -= 1
            return -0.35, -0.35, "FSD: BACKING UP"

        # Ultrasonic emergency → initiate backup
        if distance_cm < 20:
            self.backing_up       = self.BACKUP_FRAMES
            self.turn_direction   = 1 if self.turn_direction <= 0 else -1
            self.committed_frames = self.COMMIT_FRAMES
            return -0.35, -0.35, "FSD: BACKING UP"

        # Recalculate direction when commitment expires
        if self.committed_frames > 0:
            self.committed_frames -= 1
        else:
            if distance_cm < 50:
                new_dir = 1 if self.turn_direction <= 0 else -1
                self.turn_direction   = new_dir
                self.committed_frames = self.COMMIT_FRAMES
            else:
                self.turn_direction = 0

        # to turn left:  slow left motor, fast right → left=low, right=high
        # to turn right: fast left motor, slow right → left=high, right=low
        if self.turn_direction == -1:
            return self.BASE_SPEED * 0.10, self.TURN_SPEED, "FSD: TURN LEFT"
        elif self.turn_direction == 1:
            return self.TURN_SPEED, self.BASE_SPEED * 0.10, "FSD: TURN RIGHT"
        else:
            return self.BASE_SPEED, self.BASE_SPEED, "FSD: FORWARD"


# ─────────────────────────────────────────────────────────────────────────────
class AutonomousVehicle:

    def __init__(self,
                 max_speed        = 1.0,
                 dnn_model        = "ssd_mobilenet_v2_coco.pb",
                 dnn_skip         = 5,
                 enable_web       = True,
                 web_port         = 5000,
                 show_display     = True,
                 fov_deg          = 130.0,
                 undistort        = True,
                 calibration_file = None,
                 cam_index        = -1):

        # ── Camera (wide-angle undistortion for 130° Innomaker lens) ────
        self.undistort_enabled = undistort
        if undistort:
            if calibration_file:
                self.camera = FisheyeCamera.from_file(calibration_file)
            else:
                self.camera = FisheyeCamera(fov_deg=fov_deg)
        else:
            self.camera = None
            print(f"[Camera] Undistortion disabled (FOV={fov_deg}°)")

        # ── Perception & control ──────────────────────────────────────
        self.perception    = LaneDetector(width=640, height=480)
        self.steering      = PIDController(Kp=0.004, Ki=0.00008, Kd=0.002)
        self.speed_control = AdaptiveSpeedController(min_speed=0.25, max_speed=max_speed)
        self.safety        = ObstacleDetector(
            emergency_distance=STOP_DISTANCE, warning_distance=SLOW_DISTANCE
        )
        # ── Two-stage sign detection ──────────────────────────────────
        # Stage 1 — fast CV (every frame, <1 ms)
        self.sign_cv  = TrafficSignDetector()
        self.aruco    = ArucoSpeedDetector()
        # Stage 2 — DNN confirmation (gated by Stage 1 red-hint)
        #
        # Load the 67 MB SSD MobileNet v2 net ONCE and share it between the
        # stop-sign confirmer and the item tracker — halves RAM + startup.
        shared_net     = load_ssd_net(dnn_model)
        self.detector  = StopSignConfirmer(
            model_name=dnn_model, skip_frames=dnn_skip, net=shared_net)

        # ── Sentry (SENTRY mode) ─────────────────────────────────────
        self.sentry = SentryMonitor(net=shared_net)

        # ── Item tracker (TRACK mode) ─────────────────────────────────
        self.track_detector = ItemTracker(
            model_name=dnn_model, frame_width=640, frame_height=480,
            skip_frames=3, net=shared_net, prefer_tracker="KCF",
        )

        # ── State machine ─────────────────────────────────────────────
        self.car_state  = State.LANE_FOLLOW
        self.drive_mode = "LANE"   # "FSD" | "LANE" | "MANUAL"
        self.free_roam  = FreeRoamController()

        # ── Motors ───────────────────────────────────────────────────
        self.front_left  = Motor(forward=FL_FWD, backward=FL_BWD)
        self.rear_left   = Motor(forward=RL_FWD, backward=RL_BWD)
        self.front_right = Motor(forward=FR_FWD, backward=FR_BWD)
        self.rear_right  = Motor(forward=RR_FWD, backward=RR_BWD)
        print("[Motors] All 4 motors OK")

        # ── Front ultrasonic (HC-SR04 facing forward) ────────────────
        self.front_sensor = FrontSensor(trig=TRIG_PIN, echo=ECHO_PIN)
        self.front_sensor.start()

        # ── Track search state ────────────────────────────────────────
        self._track_spin_frames   = 0   # frames elapsed in 360° search spin
        self._SPIN_TOTAL          = 90  # ~3-4 s at 25 fps ≈ one full rotation

        # ── Web / display ─────────────────────────────────────────────
        self.web_enabled  = enable_web
        self.web_port     = web_port
        self.show_display = show_display

        # ── Smooth display values ─────────────────────────────────────
        self.smooth_left       = SmoothValue(0.0, alpha=0.28)
        self.smooth_right      = SmoothValue(0.0, alpha=0.28)
        self.smooth_base_speed = SmoothValue(0.0, alpha=0.22)
        self.smooth_pid        = SmoothValue(0.0, alpha=0.20)

        # ── Status hold ──────────────────────────────────────────────
        self.cam_index           = cam_index
        self._display_status     = "READY"
        self._status_hold_frames = 0
        self._STATUS_MIN_HOLD    = 6

        # ── Runtime counters ──────────────────────────────────────────
        self.autonomous_enabled   = True
        self.current_speed_limit  = max_speed
        self.active_speed_limit   = None   # km/h from sign (10/20/30/50), or None
        self.stop_sign_timer      = 0
        self.stop_sign_cooldown   = 0
        self.frame_count          = 0
        self.total_error          = 0
        self.stop_signs_detected  = 0
        self._last_steering_error = 0.0
        self._start_time          = 0.0

    # ── Status hold ───────────────────────────────────────────────────────────

    def _update_display_status(self, new_status: str) -> str:
        if new_status == self._display_status:
            self._status_hold_frames = self._STATUS_MIN_HOLD
            return self._display_status
        self._status_hold_frames -= 1
        if self._status_hold_frames <= 0:
            self._display_status     = new_status
            self._status_hold_frames = self._STATUS_MIN_HOLD
        return self._display_status

    # ── Drive-mode switcher ───────────────────────────────────────────────────

    def _set_drive_mode(self, mode: str):
        """Switch between FSD / LANE / MANUAL / TRACK / SENTRY modes safely."""
        if mode not in ("FSD", "LANE", "MANUAL", "TRACK", "SENTRY"):
            print(f"[Mode] Unknown mode '{mode}' — ignored")
            return
        prev = self.drive_mode

        # Disarm sentry when leaving SENTRY
        if prev == "SENTRY" and mode != "SENTRY":
            self.sentry.disarm()

        self.drive_mode = mode

        if mode == "MANUAL":
            self.autonomous_enabled = False
            self._stop_motors()
        elif mode == "LANE":
            self.autonomous_enabled = True
            if self.car_state in (State.FREE_ROAM, State.TRACKING, State.SENTRY):
                self.car_state = State.LANE_FOLLOW
        elif mode == "FSD":
            self.autonomous_enabled = True
            self.car_state = State.FREE_ROAM
            self.free_roam.reset()
        elif mode == "TRACK":
            self.autonomous_enabled = True
            self.car_state = State.TRACKING
            self.track_detector.last_bbox   = None
            self.track_detector.last_conf   = 0.0
            self.track_detector.lost_frames = 999
            if not self.track_detector.available:
                print("[Mode] TRACK unavailable — SSD weights missing")
        elif mode == "SENTRY":
            self.autonomous_enabled = False
            self.car_state = State.SENTRY
            self._stop_motors()
            self.sentry.arm()

        print(f"[Mode] {prev} → {mode}")

    # ── Hardware helpers ──────────────────────────────────────────────────────

    def _drive(self, left_speed: float, right_speed: float):
        left_speed  = max(0.0, min(1.0, left_speed))
        right_speed = max(0.0, min(1.0, right_speed))
        if left_speed < 0.05:
            self.front_left.stop();  self.rear_left.stop()
        else:
            self.front_left.backward(left_speed)
            self.rear_left.backward(left_speed)
        if right_speed < 0.05:
            self.front_right.stop(); self.rear_right.stop()
        else:
            self.front_right.backward(right_speed)
            self.rear_right.backward(right_speed)

    def _stop_motors(self):
        for m in (self.front_left, self.rear_left,
                  self.front_right, self.rear_right):
            m.stop()

    def _drive_manual(self, left: float, right: float):
        """Manual drive: positive = forward, negative = backward."""
        left  = max(-1.0, min(1.0, left))
        right = max(-1.0, min(1.0, right))

        if abs(left) < 0.05:
            self.front_left.stop();  self.rear_left.stop()
        elif left > 0:
            self.front_left.backward(left);  self.rear_left.backward(left)
        else:
            self.front_left.forward(-left);  self.rear_left.forward(-left)

        if abs(right) < 0.05:
            self.front_right.stop(); self.rear_right.stop()
        elif right > 0:
            self.front_right.backward(right); self.rear_right.backward(right)
        else:
            self.front_right.forward(-right); self.rear_right.forward(-right)

    def _apply_manual_drive(self, keys: dict):
        """Translate WASD key state into motor commands."""
        spd = 1.0
        w, a, s, d = keys.get("w"), keys.get("a"), keys.get("s"), keys.get("d")
        if w:
            if a:   l, r = 0.0,  spd       # forward-left  (pivot on left wheel)
            elif d: l, r = spd,  0.0        # forward-right (pivot on right wheel)
            else:   l, r = spd,  spd        # straight forward
        elif s:
            if a:   l, r = 0.0,  -spd      # reverse-left
            elif d: l, r = -spd, 0.0       # reverse-right
            else:   l, r = -spd, -spd      # straight reverse
        elif a:     l, r = -spd * 0.9,  spd * 0.9  # spin left
        elif d:     l, r =  spd * 0.9, -spd * 0.9  # spin right
        else:       l, r = 0.0, 0.0
        self._drive_manual(l, r)
        # Update smooth values so motor bars on the dashboard reflect actual commands
        self.smooth_left.update(abs(l))
        self.smooth_right.update(abs(r))

    def _cleanup_hardware(self):
        self._stop_motors()
        for m in (self.front_left, self.rear_left,
                  self.front_right, self.rear_right):
            m.close()
        self.front_sensor.stop()
        print("[Hardware] GPIO released")

    # ── FSD frame processing ─────────────────────────────────────────────────

    def process_frame_fsd(self, frame):
        """Lane-free frame loop for FSD mode. Uses front ultrasonic only."""
        self.frame_count += 1

        # Stage 1: fast CV
        self.sign_cv.detect_signs(frame)
        self.aruco.detect(frame)
        # Stage 2: DNN gated by red hint
        self.detector.detect(frame, red_hint=self.sign_cv.red_detected())

        # Read front distance and update safety overlay data
        front_dist = self.front_sensor.distance_cm
        self.safety.sensors['front']['distance'] = front_dist
        self.safety._check_obstacles()

        # FreeRoam decision — ultrasonic is now front-facing
        left, right, status = self.free_roam.compute(front_dist)

        self.smooth_left.update(abs(left))
        self.smooth_right.update(abs(right))
        self.smooth_base_speed.update(self.free_roam.BASE_SPEED)
        self.smooth_pid.update(0.0)
        self._last_steering_error = 0.0

        # Build debug frame
        debug = frame.copy()
        debug = self.safety.draw_overlay(debug)
        debug = self.sign_cv.draw_overlay(debug)
        debug = self.detector.draw_overlay(debug)
        debug = self.aruco.draw_overlay(debug)
        debug = self._draw_motor_bars(debug, abs(left), abs(right), 0.0)

        fsd_color = (0, 0, 255) if "BACKUP" in status else \
                    (0, 165, 255) if "TURN" in status else (0, 212, 255)
        cv2.putText(debug, f"[FSD] {status}",
                    (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, fsd_color, 2)
        cv2.putText(debug, f"Front: {front_dist:.0f} cm",
                    (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        if self.detector.stop_sign_detected():
            cv2.putText(debug, "STOP SIGN", (10, 175),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

        return debug, (left, right, status)

    # ── TRACK mode frame processing ──────────────────────────────────────────

    def process_frame_track(self, frame):
        """Lock onto selected COCO class and chase it. 360° spin search when lost."""
        self.frame_count += 1

        self.track_detector.detect(frame)

        front_dist = self.front_sensor.distance_cm
        self.safety.sensors['front']['distance'] = front_dist
        self.safety._check_obstacles()

        target = self.track_detector.target_class
        locked = self.track_detector.target_locked()

        if target is None:
            left, right, status = 0.0, 0.0, "TRACK: pick an item"
            self._track_spin_frames = 0

        elif locked:
            # Re-locked (also cancels any in-progress spin)
            self._track_spin_frames = 0
            offset   = self.track_detector.get_steering_offset() or 0.0
            distance = self.track_detector.get_distance_proxy()  or 0.0

            if distance > 0.60:
                left = right = 0.0
                status = f"TRACK: arrived at {target}"
            elif front_dist < STOP_DISTANCE:
                left = right = 0.0
                status = "TRACK: obstacle — stopped"
            else:
                base  = max(0.25, 0.70 * (1.0 - distance * 1.2))
                if front_dist < SLOW_DISTANCE:
                    base *= 0.6
                base  = min(base, self.current_speed_limit)
                steer = offset * 0.45
                left  = max(0.0, min(1.0, base + steer))
                right = max(0.0, min(1.0, base - steer))
                side  = "center" if abs(offset) < 0.12 else ("left" if offset < 0 else "right")
                status = f"TRACK: chasing {target} ({side})"

        elif self._track_spin_frames < self._SPIN_TOTAL:
            # 360° spin-in-place search — needs _drive_manual (supports negative)
            self._track_spin_frames += 1
            spd   = min(0.30, self.current_speed_limit)
            left  =  spd   # left side forward
            right = -spd   # right side backward → spins clockwise
            pct   = int(100 * self._track_spin_frames / self._SPIN_TOTAL)
            status = f"TRACK: searching {target}... {pct}%"

        else:
            # Full rotation done, still nothing → give up
            left, right, status = 0.0, 0.0, f"TRACK: lost {target}"

        self.smooth_left.update(left)
        self.smooth_right.update(right)
        self.smooth_base_speed.update((left + right) / 2)
        self.smooth_pid.update(0.0)
        self._last_steering_error = 0.0

        debug = frame.copy()
        debug = self.safety.draw_overlay(debug)
        debug = self.track_detector.draw_overlay(debug)
        debug = self._draw_motor_bars(debug, left, right, 0.0)

        hdr_color = (0, 220, 255) if locked else (120, 180, 200)
        cv2.putText(debug, f"[TRACK] {status}", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, hdr_color, 2)
        cv2.putText(debug, f"Front: {front_dist:.0f} cm",
                    (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        return debug, (left, right, status)

    # ── SENTRY mode frame processing ─────────────────────────────────────────

    def process_frame_sentry(self, frame):
        """Stationary surveillance — motors off, camera watches for motion/people."""
        self.frame_count += 1
        debug  = self.sentry.process(frame)
        status = (
            "SENTRY: person detected" if self.sentry.person_active else
            "SENTRY: motion detected" if self.sentry.motion_active else
            "SENTRY: watching"
        )
        self.smooth_left.update(0.0)
        self.smooth_right.update(0.0)
        self.smooth_base_speed.update(0.0)
        self.smooth_pid.update(0.0)
        self._last_steering_error = 0.0
        return debug, (0.0, 0.0, status)

    # ── Web command handler ───────────────────────────────────────────────────

    def _handle_web_commands(self):
        cmd = pi_server.get_pending_command()
        if cmd is None:
            return
        action = cmd.get("action")

        if action == "toggle_auto":
            self.autonomous_enabled = not self.autonomous_enabled
            print(f"[Web] Autonomous: {'ON' if self.autonomous_enabled else 'OFF'}")

        elif action == "emergency_stop":
            self.autonomous_enabled = False
            self._stop_motors()
            print("[Web] Emergency stop!")

        elif action == "reset":
            self._reset_all()

        elif action == "set_speed":
            val = float(cmd.get("value", 0.8))
            self.current_speed_limit = max(0.1, min(1.0, val))
            print(f"[Web] Speed limit → {self.current_speed_limit:.2f}")

        elif action == "set_mode":
            mode = str(cmd.get("value", "LANE")).upper()
            self._set_drive_mode(mode)

        elif action == "set_track_target":
            val = cmd.get("value")
            # Empty string or null → clear target (car stops)
            self.track_detector.set_target(val if val else None)

        elif action == "track_click":
            try:
                x = int(cmd.get("x", -1))
                y = int(cmd.get("y", -1))
            except (TypeError, ValueError):
                x = y = -1
            if 0 <= x < 640 and 0 <= y < 480:
                self.track_detector.set_click_target(x, y)
            else:
                print(f"[Web] Invalid track_click coords ({x}, {y})")

        elif action == "arm_sentry":
            self._set_drive_mode("SENTRY")

        elif action == "disarm_sentry":
            if self.drive_mode == "SENTRY":
                self._set_drive_mode("LANE")

        elif action == "clear_sentry_events":
            self.sentry.clear_events()
            pi_server.clear_sentry_events()

        pi_server.clear_command()

    # ── Telemetry ─────────────────────────────────────────────────────────────

    def _build_telemetry(self, left: float, right: float, status: str) -> dict:
        elapsed  = max(time.time() - self._start_time, 0.1)
        # Merge CV confirmed signs + DNN detections for the dashboard
        sign_dets = (
            [{"label": "stop sign (CV)", "conf": round(c, 2)}
             for _, _, c in self.sign_cv.confirmed_signs]
            + [{"label": d[0] + " (DNN)", "conf": round(d[2], 2)}
               for d in self.detector.all_detections]
        )

        return {
            "mode":                  "AUTONOMOUS" if self.autonomous_enabled else "MANUAL",
            "drive_mode":            self.drive_mode,
            "status":                status,
            "car_state":             self.car_state,
            "speed_left":            round(left, 3),
            "speed_right":           round(right, 3),
            "speed_fl":              round(left, 3),
            "speed_rl":              round(left, 3),
            "speed_fr":              round(right, 3),
            "speed_rr":              round(right, 3),
            "base_speed":            round(self.smooth_base_speed.value, 3),
            "steering_error":        round(self._last_steering_error, 1),
            "front_distance_cm":     round(self.front_sensor.distance_cm, 1),
            "fps":                   round(self.frame_count / elapsed, 1),
            "frame_count":           self.frame_count,
            "stop_signs_detected":   self.stop_signs_detected,
            "stop_sign_active":      (bool(self.sign_cv.confirmed_signs)
                                      or self.detector.stop_sign_detected()),
            "dnn_enabled":           self.detector.available,
            "dnn_detections":        sign_dets,
            "speed_limit_kmh":       self.aruco.get_speed_limit(),
            "obstacle_modifier":     round(self.safety.get_speed_modifier(), 2),
            "track_available":       self.track_detector.available,
            "track_click_available": self.track_detector.click_available,
            "track_mode":            self.track_detector.mode,
            "track_target":          self.track_detector.target_class,
            "track_locked":          self.track_detector.target_locked(),
            "track_conf":            round(self.track_detector.last_conf, 2),
            "track_lost_frames":     self.track_detector.lost_frames,
            "track_classes":         list(TRACK_TARGETS.keys()),
            "sentry_armed":          self.sentry.armed,
            "sentry_motion":         self.sentry.motion_active,
            "sentry_person":         self.sentry.person_active,
            "sentry_fire":           self.sentry.fire_active,
        }

    # ── Main perception + decision loop ──────────────────────────────────────

    def process_frame(self, frame):
        self.frame_count += 1
        steering_error, vision_frame = self.perception.process_frame(frame)

        # ── No lane → emergency stop ───────────────────────────────────
        if steering_error is None:
            for s in (self.smooth_left, self.smooth_right,
                      self.smooth_base_speed, self.smooth_pid):
                s.update(0.0)
            return (
                self._create_debug_frame(
                    vision_frame, 0, 0.0, 0.0, 0.0, 0.0, "EMERGENCY STOP"
                ),
                (0.0, 0.0, "EMERGENCY STOP"),
            )

        self.total_error          += abs(steering_error)
        self._last_steering_error  = steering_error

        # ── Stage 1: fast CV (every frame) ────────────────────────────
        self.sign_cv.detect_signs(frame)
        red_hint = self.sign_cv.red_detected()

        # ArUco speed limit (every frame, < 1 ms)
        detected_limit = self.aruco.detect(frame)
        if detected_limit is not None:
            self.active_speed_limit = detected_limit

        # ── Stage 2: DNN confirmation (gated — runs rarely) ───────────
        self.detector.detect(frame, red_hint=red_hint)

        # ── Front ultrasonic obstacle modifier ────────────────────────
        front_dist = self.front_sensor.distance_cm
        self.safety.sensors['front']['distance'] = front_dist
        self.safety._check_obstacles()
        obstacle_modifier = self.safety.get_speed_modifier()

        # ── Stop sign: CV confirmed OR DNN confirmed ───────────────────
        stop_confirmed = (
            bool(self.sign_cv.confirmed_signs)
            or self.detector.stop_sign_detected()
        )
        if self.stop_sign_cooldown > 0:
            self.stop_sign_cooldown -= 1
        if stop_confirmed and self.stop_sign_cooldown == 0:
            if self.stop_sign_timer == 0:
                self.stop_sign_timer    = 60
                self.stop_sign_cooldown = 120
                self.stop_signs_detected += 1
                source = "DNN" if self.detector.stop_sign_detected() else "CV"
                print(f"STOP SIGN [{source}] — holding 2 s")

        # ── Effective speed cap from sign ─────────────────────────────
        if self.active_speed_limit is not None:
            effective_max = self.current_speed_limit * SPEED_LIMIT_MAP[self.active_speed_limit]
        else:
            effective_max = self.current_speed_limit

        # ── Speed decision ────────────────────────────────────────────
        if self.stop_sign_timer > 0:
            base_speed = 0.0
            self.stop_sign_timer -= 1
            status = f"STOPPED: sign ({self.stop_sign_timer})"
        else:
            base_speed = self.speed_control.calculate_speed(steering_error, obstacle_modifier)
            base_speed = min(base_speed, effective_max)
            status     = self.speed_control.get_speed_category(
                abs(steering_error)
            ).replace("_", " ")

        # ── Steering ─────────────────────────────────────────────────
        if self.autonomous_enabled and base_speed > 0:
            pid_output  = self.steering.compute(steering_error)
            left_speed  = max(0.0, min(1.0, base_speed + pid_output))
            right_speed = max(0.0, min(1.0, base_speed - pid_output))
        else:
            pid_output  = 0.0
            left_speed  = 0.0
            right_speed = 0.0

        self.smooth_left.update(left_speed)
        self.smooth_right.update(right_speed)
        self.smooth_base_speed.update(base_speed)
        self.smooth_pid.update(pid_output)

        return (
            self._create_debug_frame(
                vision_frame, steering_error, pid_output,
                base_speed, left_speed, right_speed, status
            ),
            (left_speed, right_speed, status),
        )

    # ── HUD ───────────────────────────────────────────────────────────────────

    def _create_debug_frame(self, vision_frame, error, pid_output,
                            base_speed, left_speed, right_speed,
                            status):
        frame = vision_frame.copy()
        h, w = frame.shape[:2]
        frame = self.safety.draw_overlay(frame)
        frame = self.sign_cv.draw_overlay(frame)
        frame = self.detector.draw_overlay(frame)
        frame = self.aruco.draw_overlay(frame)

        # ── SPEED BADGE (top-right) ──────────────────────────────────
        speed_pct = int(self.smooth_base_speed.value * 100)
        category = self.speed_control.get_speed_category(abs(error)).replace("_", " ").upper()

        cv2.putText(frame, category, (w - 200, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 200, 150), 1)

        badge_text = f"Speed: {speed_pct}%"
        (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        bx1, by1 = w - tw - 24, 28
        bx2, by2 = w - 8, 28 + th + 14
        if speed_pct >= 70:
            badge_bg = (0, 180, 0)
        elif speed_pct >= 40:
            badge_bg = (0, 160, 255)
        else:
            badge_bg = (0, 80, 255)
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), badge_bg, -1)
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (255, 255, 255), 1)
        cv2.putText(frame, badge_text, (bx1 + 8, by2 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # ── STATUS (left side with background) ───────────────────────
        display_status = self._update_display_status(status)
        if "STOP" in display_status:
            status_color, status_bg = (0, 0, 255), (0, 0, 100)
        elif "SLOW" in display_status:
            status_color, status_bg = (0, 180, 255), (0, 60, 100)
        else:
            status_color, status_bg = (200, 255, 200), (0, 60, 0)

        status_text = f"Status: {display_status}"
        (stw, sth), _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (8, 115), (stw + 20, 140), status_bg, -1)
        cv2.rectangle(frame, (8, 115), (stw + 20, 140), status_color, 1)
        cv2.putText(frame, status_text,
                    (12, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

        # Sign detector status
        if self.detector.stop_sign_detected():
            cv2.putText(frame, "STOP SIGN", (12, 162),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
        if self.active_speed_limit is not None:
            mode_map = {10: "SLOW", 20: "AVG", 30: "NORMAL", 50: "MAX"}
            tag = mode_map.get(self.active_speed_limit, "")
            cv2.putText(frame, f"LIMIT {self.active_speed_limit}km/h {tag}",
                        (12, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)

        # ── BOTTOM BAR — front distance obstacle warning ─────────────
        front_dist = self.front_sensor.distance_cm
        modifier   = self.safety.get_speed_modifier()
        if front_dist < SLOW_DISTANCE:
            bar_y = h - 30
            cv2.rectangle(frame, (0, bar_y), (w, h), (0, 0, 0), -1)
            cx = w // 2
            circ_color = (0, 0, 255) if front_dist < STOP_DISTANCE else (0, 165, 255)
            cv2.circle(frame, (cx, bar_y + 15), 10, circ_color, -1)
            dist_label = "STOP" if front_dist < STOP_DISTANCE else "SLOW"
            cv2.putText(frame, f"FRONT {front_dist:.0f}cm -- {dist_label}",
                        (cx - 60, bar_y + 12), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 255), 1)
            if modifier < 1.0:
                obs_text = "OBSTACLE -- SLOWING" if modifier > 0 else "OBSTACLE -- STOPPING"
                obs_color = (0, 165, 255) if modifier > 0 else (0, 0, 255)
                cv2.putText(frame, obs_text, (10, h - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, obs_color, 2)

        # Motor bars
        frame = self._draw_motor_bars(
            frame, self.smooth_left.value, self.smooth_right.value, self.smooth_pid.value)

        return frame

    def _draw_motor_bars(self, frame, left_speed, right_speed, pid_output):
        h, w = frame.shape[:2]
        bw   = 25    # bar width
        bh   = 130   # bar height
        gap  = 4     # gap within a pair
        pgap = 12    # gap between left and right pair
        by   = h - 55          # bottom of all bars

        # Positions from right edge: RR, FR, [pgap], RL, FL
        x_rr = w - 45
        x_fr = x_rr - gap - bw
        x_rl = x_fr - pgap - bw
        x_fl = x_rl - gap - bw

        bar_color = (0, 230, 100) if abs(pid_output) < 0.1 else (0, 180, 255)
        left_fill  = int(bh * max(0.0, min(1.0, left_speed)))
        right_fill = int(bh * max(0.0, min(1.0, right_speed)))

        for bx, fill, label in (
            (x_fl, left_fill,  "FL"),
            (x_rl, left_fill,  "RL"),
            (x_fr, right_fill, "FR"),
            (x_rr, right_fill, "RR"),
        ):
            # background
            cv2.rectangle(frame, (bx, by - bh), (bx + bw, by), (30, 30, 30), -1)
            # fill
            if fill:
                cv2.rectangle(frame, (bx, by - fill), (bx + bw, by), bar_color, -1)
            # border
            cv2.rectangle(frame, (bx, by - bh), (bx + bw, by), (80, 80, 80), 1)
            # label above
            cv2.putText(frame, label, (bx, by - bh - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)

        # Speed values below each pair
        cv2.putText(frame, f"{left_speed:.2f}",
                    (x_fl, by + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)
        cv2.putText(frame, f"{right_speed:.2f}",
                    (x_fr, by + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)
        return frame

    # ── Reset ─────────────────────────────────────────────────────────────────

    def _reset_all(self):
        self.steering.reset()
        self.speed_control.reset()
        self.safety.reset()
        self.perception.reset_smoothing()
        self.sign_cv.reset()
        self.aruco.reset()
        self.detector.reset()
        self.free_roam.reset()
        self.track_detector.reset()
        self._track_spin_frames = 0
        self.sentry.disarm()
        self.car_state           = State.LANE_FOLLOW
        self.drive_mode          = "LANE"
        self.autonomous_enabled  = True
        self.current_speed_limit = 0.8
        self.active_speed_limit  = None
        self.stop_sign_timer     = 0
        self.smooth_left.set_immediate(0.0)
        self.smooth_right.set_immediate(0.0)
        self.smooth_base_speed.set_immediate(0.0)
        self.smooth_pid.set_immediate(0.0)
        self._display_status = "READY"
        print("[System] All systems reset")

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self):
        if self.web_enabled:
            ok = pi_server.start_server(self.web_port)
            if not ok:
                print("[WebServer] Flask not installed — dashboard disabled")
                self.web_enabled = False

        # ── Innomaker 1080P USB2.0 130° camera (front-mounted, upside-down) ──
        self._cap = None
        search = [self.cam_index] if self.cam_index >= 0 else range(4)
        for idx in search:
            _c = cv2.VideoCapture(idx)
            if _c.isOpened():
                self._cap = _c
                print(f"[Camera] Innomaker 130° USB camera found at /dev/video{idx}")
                break
            _c.release()
        if self._cap is None:
            raise RuntimeError(
                "[Camera] Innomaker USB camera not found!\n"
                "  Check cable, then run: ls /dev/video*\n"
                "  If found at a different index, set it with: --cam-index N"
            )
        self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self._cap.set(cv2.CAP_PROP_FPS, 30)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        time.sleep(1)
        print("[Camera] 640×480 ready (front-mounted, 180° rotation applied)")
        if self.show_display:
            print("Keys: [Q] Quit  [SPACE] Auto/Manual  [F] FSD  [R] Reset  [S] Snap  [D] Debug")

        # Drain the camera in a background thread so perception never blocks
        # on USB I/O.  Big win on Pi 3.
        self._grabber = CameraGrabber(self._cap)
        self._grabber.start()

        self._start_time = time.time()

        while True:
            ret, frame = self._grabber.latest()
            if not ret:
                time.sleep(0.005)   # wait for the first frame
                continue
            # camera is mounted right-side up — no rotation needed
            if self.undistort_enabled and self.camera:
                frame = self.camera.undistort(frame)

            # ── Frame processing (mode-dependent) ────────────────────────
            if self.drive_mode == "FSD":
                debug_frame, (left, right, status) = self.process_frame_fsd(frame)
            elif self.drive_mode == "TRACK":
                debug_frame, (left, right, status) = self.process_frame_track(frame)
            elif self.drive_mode == "SENTRY":
                debug_frame, (left, right, status) = self.process_frame_sentry(frame)
            else:
                debug_frame, (left, right, status) = self.process_frame(frame)

            if self.web_enabled:
                self._handle_web_commands()

            # ── Motor control (mode-dependent) ───────────────────────────
            if self.drive_mode == "FSD":
                self._drive_manual(left, right)
            elif self.drive_mode == "TRACK":
                self._drive_manual(left, right)  # supports negative for spin-in-place
            elif self.drive_mode == "SENTRY":
                self._stop_motors()
            elif self.drive_mode == "LANE":
                self._drive(left, right)
            else:  # MANUAL
                if self.web_enabled:
                    self._apply_manual_drive(pi_server.get_manual_keys())
                else:
                    self._stop_motors()

            if self.web_enabled:
                pi_server.push_frame(debug_frame)
                if self.frame_count % 3 == 0:   # dashboard polls every 400ms — 10Hz is plenty
                    pi_server.push_telemetry(self._build_telemetry(left, right, status))
                    if self.drive_mode == "SENTRY":
                        pi_server.push_sentry_events(self.sentry.events)

            if self.show_display:
                state_info = f"  [{self.car_state}]" if self.drive_mode == "LANE" else ""
                title = f"VectorVance  [{self.drive_mode}]{state_info}"
                cv2.imshow(title, debug_frame)

            if self.frame_count % 30 == 0:
                elapsed = time.time() - self._start_time
                fps     = self.frame_count / max(elapsed, 0.1)
                avg_err = self.total_error / max(self.frame_count, 1)
                limit_str = f" Limit:{self.active_speed_limit}km/h" if self.active_speed_limit else ""
                print(f"Frame {self.frame_count:04d} | "
                      f"State:{self.car_state:18s} | "
                      f"L:{left:.2f} R:{right:.2f} | "
                      f"Front:{self.front_sensor.distance_cm:.0f}cm | "
                      f"FPS:{fps:.1f} | AvgErr:{avg_err:.1f}px{limit_str}")

            if self.show_display:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord(' '):
                    # Toggle between current auto mode and MANUAL
                    if self.drive_mode == "MANUAL":
                        self._set_drive_mode("LANE")
                    else:
                        self._set_drive_mode("MANUAL")
                elif key == ord('f'):
                    self._set_drive_mode("FSD")
                elif key == ord('n'):
                    if self.drive_mode == "SENTRY":
                        self._set_drive_mode("LANE")
                    else:
                        self._set_drive_mode("SENTRY")
                elif key == ord('r'):
                    self._reset_all()
                elif key == ord('s'):
                    fname = f"/home/pi/snap_{self.frame_count:04d}.jpg"
                    cv2.imwrite(fname, debug_frame)
                    print(f"Snapshot: {fname}")
                elif key == ord('d'):
                    print(f"State: {self.car_state}")
                    print(f"CV stop: {bool(self.sign_cv.confirmed_signs)} | "
                          f"DNN stop: {self.detector.stop_sign_detected()} | "
                          f"ArUco limit: {self.aruco.get_speed_limit()} km/h | "
                          f"front: {self.front_sensor.distance_cm:.0f}cm")

        if hasattr(self, "_grabber"):
            self._grabber.stop()
        if self._cap:
            self._cap.release()
        if self.show_display:
            cv2.destroyAllWindows()
        self._stop_motors()
        self._cleanup_hardware()
        elapsed = time.time() - self._start_time
        print("=" * 60)
        print(f"Duration    : {elapsed:.1f}s")
        print(f"Frames      : {self.frame_count}")
        print(f"Avg FPS     : {self.frame_count / max(elapsed, 0.1):.1f}")
        print(f"Stop signs  : {self.stop_signs_detected}")
        print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description="VectorVance Autonomous Car")
    p.add_argument("--speed",       type=float, default=1.0)
    p.add_argument("--dnn-model",     type=str,   default="ssd_mobilenet_v2_coco.pb")
    p.add_argument("--dnn-skip",      type=int,   default=5)
    p.add_argument("--port",          type=int,   default=5000)
    p.add_argument("--no-web",        action="store_true")
    p.add_argument("--no-display",    action="store_true")
    p.add_argument("--fov",           type=float, default=130.0,
                   help="Camera FOV in degrees (default: 130 for Innomaker wide-angle)")
    p.add_argument("--no-undistort",  action="store_true", default=True,
                   help="Disable wide-angle lens undistortion (default: OFF until calibrated)")
    p.add_argument("--calibration",   type=str,   default=None,
                   help="Path to calibration .npz file (uses approximation if omitted)")
    p.add_argument("--cam-index",     type=int,   default=-1,
                   help="Force Innomaker USB camera device index (default: auto-detect)")
    args = p.parse_args()

    AutonomousVehicle(
        max_speed        = args.speed,
        dnn_model        = args.dnn_model,
        dnn_skip         = args.dnn_skip,
        enable_web       = not args.no_web,
        web_port         = args.port,
        show_display     = not args.no_display,
        fov_deg          = args.fov,
        undistort        = not args.no_undistort,
        calibration_file = args.calibration,
        cam_index        = args.cam_index,
    ).run()


if __name__ == "__main__":
    main()
