# VectorVance — Autonomous Car Project

## Project Overview

VectorVance is a Raspberry Pi-based autonomous car that follows lanes, detects obstacles and traffic signs, navigates forks using colour-coded tape, and exposes a live web dashboard.

**Entry point:** `main.py`
**Platform:** Raspberry Pi (Linux/lgpio) + Innomaker 1080P USB2.0 UVC 130° wide-angle camera (rear-mounted) + dual L298N motor drivers

---

## Architecture

```
main.py  (AutonomousVehicle + run loop)
├── camera.py            — fisheye undistortion for wide-angle lenses (cv2.fisheye)
├── rear_monitor.py      — rear HC-SR04 collision detection + Telegram/SMS/email alerts
├── perception.py        — lane detection (colour mask → Canny → Hough → EMA fit)
├── controller.py        — PID steering controller
├── speed_controller.py  — adaptive speed based on steering error
├── safety.py            — DNN-fed obstacle detector + HUD overlay (no front ultrasonic)
├── dnn_detector.py      — SSD MobileNet v2 sign/obstacle detection (cv2.dnn, COCO)
├── sentry.py            — stationary surveillance mode (motion/person/fire + ntfy.sh alerts)
├── color_sign_detector.py — colour tape detection (GREEN/BLUE forks, RED destination)
├── intersection_detector.py — fork/intersection detection from vision data
└── pi_server.py         — Flask web dashboard (background thread)
    └── webapp/templates/dashboard.html
```

---

## Hardware

| Component | Details |
|-----------|---------|
| SBC | Raspberry Pi (lgpio GPIO) |
| Camera | Innomaker 1080P USB2.0 UVC 130° Wide Angle Camera, rear-mounted, `cv2.VideoCapture` auto-detect (index 0–3), rotated 180° in code |
| Motors | 4× DC motors via dual L298N — FL, FR, RL, RR |
| Rear ultrasonic | HC-SR04 on TRIG=GPIO4, ECHO=GPIO17 — **rear-facing** for collision/impact detection |
| Rear collision | Alert triggered at < 20 cm; sends Telegram / SMS / email via `rear_monitor.py` |

### GPIO pin map
```
FL_FWD=25, FL_BWD=27
FR_FWD=5,  FR_BWD=15
RL_FWD=26, RL_BWD=20
RR_FWD=16, RR_BWD=6
TRIG=4,    ECHO=17
```

---

## State Machine (`State` enum in `main.py`)

| State | Description |
|-------|-------------|
| `LANE_FOLLOW` | Normal PID lane following |
| `FORK_WAITING` | Fork detected — car stopped, waiting for user to pick GREEN or BLUE via dashboard |
| `COLOR_FOLLOWING` | Steering toward the chosen colour tape |
| `ARRIVED` | RED destination tape confirmed (4 consecutive frames) — car stops |
| `SENTRY` | Car parked, wheels locked, front camera used for surveillance |

---

## Key Modules

### `camera.py` — `FisheyeCamera`
- Camera: Innomaker 1080P USB2.0 UVC 130° wide-angle (replaced ELP 170° due to IR sensor issue)
- Corrects barrel distortion from the 130° wide-angle lens using `cv2.fisheye` undistortion
- Applied in `main.py` immediately after 180° rotation, before any other module sees the frame
- Uses equidistant fisheye model: `r = f·θ` where `f ≈ 212` for 130° at 640×480
- Default K and D are **approximate** — run calibration for best accuracy:
  ```bash
  python camera.py --calibrate --images ./calibration_images/
  python main.py --calibration calibration.npz
  ```
- `balance=0.0` (default): crops black borders, keeps clean image centre
- CLI flags: `--fov 130`, `--no-undistort`, `--calibration path.npz`
- Test undistortion on a single image: `python camera.py --test image.jpg`

### `perception.py` — `LaneDetector`
- Extracts white/yellow lane markings via HSV+HLS colour masks
- Canny edges → Hough lines → fit left/right lanes with `np.polyfit`
- EMA smoothing (α=0.10) on lane fits; weighted rolling average on steering error
- Failsafes: camera-blocked detection, lost-lane fallback
- Returns `(steering_error_px, debug_frame)`

### `controller.py` — `PIDController`
- Kp=0.003, Ki=0.0001, Kd=0.001 (set in `main.py`)
- Anti-windup, filtered derivative (deque of 5), output clamped to ±1.0

### `speed_controller.py` — `AdaptiveSpeedController`
- Straight (<30px error) → 100%, Gentle (<80px) → 75%, Moderate (<150px) → 50%, Sharp → 30%
- Multiplied by `obstacle_modifier` from `YoloDetector`

### `dnn_detector.py` — `DNNDetector`
- **Backend: OpenCV DNN + SSD MobileNet v2 COCO** — no PyTorch, no ultralytics; works on Pi 3
- Tracks: stop signs (11), traffic lights (9), persons (0), bikes (1), cars (2), motorcycles (3), buses (5), trucks (7)
- Model files required in `src/`: `ssd_mobilenet_v2_coco.pb` (~67 MB) + `ssd_mobilenet_v2_coco.pbtxt` (~8 KB)
  - Download once: `python dnn_detector.py --download`
- Input: 300×300 blob, normalized to [-1, 1]; TF COCO class IDs remapped internally to COCO 0-indexed numbering
- Zone-aware: only centre-zone obstacles trigger speed reduction
- Temporal stop-sign filter: confirmed if seen in ≥2 of last 5 frames
- Returns danger level: `CLEAR | CAUTION | DANGER | STOP`
- `skip_frames=5` on Pi (every 5th frame to manage CPU; cached results returned on skipped frames)
- Expected Pi 3 performance: ~3–6 FPS inference; ~15–25 FPS effective with `skip_frames=5`

### `sentry.py` — `SentryMonitor`
- Activated in `SENTRY` state — motors set to 0.0, front camera used for surveillance
- **Detection pipeline** (per frame):
  - Motion: frame differencing + Gaussian blur (21×21) + contour area threshold (≥1200 px²)
  - Person/object: SSD MobileNet v2 shared net, every 8th frame (`DNN_SKIP=8`), confidence ≥50%
  - Fire: HSV colour threshold — red/orange/yellow at high saturation+brightness; triggers if ≥1.5% of frame pixels match (`FIRE_PIXEL_RATIO=0.015`)
- **State flags** on the instance: `motion_active`, `person_active`, `fire_active`
- Alert priority (HUD banner + ntfy): fire > person > motion
- Alert cooldown: 30 s per event type to avoid notification spam
- Snapshots saved to `/home/pi/sentry_snaps/sentry_YYYYMMDD_HHMMSS_####.jpg`
- **Notifications: ntfy.sh** — set `NTFY_TOPIC` env var; optionally `NTFY_URL` for self-hosted server
  - Sends snapshot as image attachment via `PUT {url}/{topic}` with `Title`, `Tags`, `Priority` headers
  - Priority: fire=5 (max), person=4 (high), motion=3 (default)
- Constructor: `SentryMonitor(net=shared_net)` — reads `NTFY_TOPIC`/`NTFY_URL` from env
- `arm()` / `disarm()` reset state; `process(frame)` returns annotated debug frame

### `color_sign_detector.py` — `ColorSignDetector`
- Detects GREEN/BLUE (fork path) and RED (destination) tape via HSV ranges
- ROI: bottom 55% of frame (tape is on the ground)
- `get_steering_offset()` → smoothed -1.0…+1.0 offset toward target tape
- `destination_reached()` → True after RED visible for 4 consecutive frames
- `set_target(color)` called by web command handler when user picks path

### `intersection_detector.py` — `IntersectionDetector`
- 4 signals: line count spike, lane width expansion, asymmetric confidence, diverging fits
- Self-calibrates on first 20 frames to learn baseline
- Fork confirmed when smoothed score > 0.45 sustained over ≥2 of last 4 frames
- 30-frame cooldown after each detection

### `safety.py` — `ObstacleDetector`
- HC-SR04 distance fed in from `main.py` every 3 frames
- Emergency stop < 20 cm; warning/slowdown < 50 cm

### `pi_server.py` — Flask Web Dashboard
- Runs in daemon thread on port 5000
- Routes:
  - `GET /` — dashboard HTML
  - `GET /video_feed` — MJPEG stream
  - `GET /api/status` — telemetry JSON
  - `POST /api/command` — control commands
  - `GET /api/sentry_events` — sentry event log JSON
- Valid commands: `toggle_auto`, `emergency_stop`, `reset`, `set_speed`, `set_target_color`, `arm_sentry`, `disarm_sentry`, `clear_sentry_events`
- Thread-safe shared state via `threading.Lock`
- AI Drive command parsing: keyword/synonym matching only — no external API (`_resolve_drive_via_keywords`)
- AI Track target resolution: keyword/synonym matching only — no external API (`_resolve_via_keywords`)

---

## Environment Variables (`.env`)

| Variable | Purpose |
|----------|---------|
| `NTFY_TOPIC` | ntfy.sh topic for sentry alerts (e.g. `vectorvance-sentry`) |
| `NTFY_URL` | ntfy server base URL (default: `https://ntfy.sh`; override for self-hosted) |

> **Note:** `rear_monitor.py` uses its own Telegram/SMS/email config — separate from sentry alerts.

---

## Usage

```bash
python main.py                          # full autonomous, MobileNet SSD on
python main.py --no-web                 # disable web dashboard
python main.py --no-display             # headless (no cv2 window)
python main.py --speed 0.6              # set max speed (default 0.8)
python main.py --dnn-skip 3             # DNN inference every 3 frames
python main.py --no-undistort           # skip wide-angle correction
python main.py --fov 120                # override FOV (default 130°)
python main.py --calibration calib.npz # use calibrated lens params
```

### Keyboard shortcuts (when display enabled)
| Key | Action |
|-----|--------|
| Q | Quit |
| SPACE | Toggle autonomous mode |
| R | Reset all systems |
| S | Save snapshot to `/home/pi/snap_XXXX.jpg` |
| D | Print debug info |
| G | Set path colour to GREEN |
| B | Set path colour to BLUE |

---

## Dependencies

```
picamera2        # Pi camera
lgpio            # GPIO (Pi)
gpiozero         # Motor control
opencv-python    # cv2 (includes cv2.dnn — no extra install needed)
numpy
flask            # web dashboard (pip install flask --break-system-packages)
```

Model files: `ssd_mobilenet_v2_coco.pb` + `ssd_mobilenet_v2_coco.pbtxt` must be present in `src/`.
Download once: `python dnn_detector.py --download`
No PyTorch or ultralytics required.

---

## Telemetry Fields (`/api/status`)

Key fields pushed to the dashboard every frame:

- `mode` — `AUTONOMOUS` or `MANUAL`
- `car_state` — current state machine state
- `fork_waiting` — bool, True when stopped at fork
- `fork_options` — list of visible path colours e.g. `["GREEN", "BLUE"]`
- `speed_left`, `speed_right` — motor speeds 0.0–1.0
- `distance_cm` — ultrasonic reading
- `dnn_danger` — `CLEAR | CAUTION | DANGER | STOP`
- `dnn_detections` — list of `{label, conf}`
- `color_target` — selected path colour
- `color_target_visible` — bool
- `sentry_armed` — bool
- `sentry_motion` — bool, motion detected this frame
- `sentry_person` — bool, person detected this frame
- `sentry_fire` — bool, fire detected this frame

---

## Development Notes

- Camera is the Innomaker 1080P USB2.0 130° wide-angle, rear-mounted (replaced ELP 170° due to IR sensor issue). Frame is rotated 180° at capture (`cv2.ROTATE_180`) in `main.py` to compensate for upside-down mounting. `perception.py` does NOT rotate — caller handles it.
- Frame pipeline in `main.py`: capture → rotate 180° → wide-angle undistort (130°) → all other modules.
- Default `FisheyeCamera` K/D are approximate for 130° at 640×480. Run `python camera.py --calibrate` for accurate coefficients — **strongly recommended** after swapping to the new camera.
- `mainv2.py` is an alternate entry point (webcam/video file testing) — rotation is source-dependent there.
- Camera auto-detects at `/dev/video0` through `/dev/video3`. Override with `--cam-index N` if needed.
- `safety.py` `simulate_sensors()` method is for development only (derives fake distances from frame brightness). Production uses actual HC-SR04 readings set directly on `self.safety.sensors['front']['distance']`.
- `use_tracking` parameter retained in `DNNDetector` for API compatibility but is unused (SSD is single-shot, no tracking).
- Voice commands use the browser's Web Speech API (Chrome/Edge only) — no server round-trip, no API key needed. On plain HTTP from a remote device, mic may be blocked; enable Chrome's "Insecure origins treated as secure" flag for the Pi's IP if needed.

---

## Dashboard UI Design System (`webapp/templates/dashboard.html`)

Single-file HTML/CSS/JS dashboard served by Flask on port 5000. No build step, no framework — vanilla JS only.

**Design approach**: dark industrial HUD aesthetic, gold (#C8861A) as dominant brand color, cyan for live data values only.

**Animation patterns in use**:
- Card entrance: cards start `opacity: 0` until `#app.app-entered` is set by `dismiss()` (splash exit), then stagger in with `cardRise` keyframe and `cubic-bezier(0.25, 1, 0.5, 1)` easing
- Value flash: `flashIfChanged(el, key, newVal)` — flashes gold on the element when a telemetry value's zone category changes (e.g. dist zone `clear→warn→danger`)
- Button micro-interactions: CSS `:active` `transform: scale(0.92–0.95)` on all interactive controls
- Detection items: auto-animate via CSS `detSlideIn` each time `innerHTML` is rebuilt
- `prefers-reduced-motion` media query disables all animations/transitions

**Banned patterns** (per impeccable skill):
- No `border-left`/`border-right` > 1px as colored stripe on cards
- No gradient text (`background-clip: text`)
- No glassmorphism
- No bounce/elastic easing — use `cubic-bezier(0.25,1,0.5,1)` (ease-out-quart) instead
