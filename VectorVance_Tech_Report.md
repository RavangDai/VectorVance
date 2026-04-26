# VectorVance — Technical Report

**Project:** VectorVance Autonomous Car
**Platform:** Raspberry Pi 3 (Linux, lgpio)
**Author:** Bibek Pathak
**Date:** April 25, 2026
**Codebase:** `C:/autonomous_car/src/` — ~5,500 lines of Python

---

## 1. Project Overview

VectorVance is a Raspberry Pi-based autonomous ground vehicle designed for multi-modal operation. The system supports five distinct drive modes — lane following, free roam obstacle avoidance, object tracking, manual control, and stationary surveillance — unified under a single state machine and controlled through a real-time web dashboard.

The primary goals of the project are:

- Demonstrate camera-based autonomous navigation (lane detection + PID steering)
- Integrate ultrasonic obstacle avoidance as a reliable, low-latency safety layer
- Detect and respond to traffic signs using both classical CV and DNN inference
- Provide remote control and live telemetry via a browser-accessible dashboard
- Support a surveillance mode with motion, person, and fire detection plus push notifications

---

## 2. Hardware

### 2.1 Computing Platform
- **SBC:** Raspberry Pi 3 (ARMv8, 1 GB RAM)
- **OS:** Raspberry Pi OS (Linux)
- **GPIO Library:** `lgpio` (direct hardware PWM)

### 2.2 Camera
- **Model:** Innomaker 1080P USB2.0 UVC Wide-Angle Camera
- **Field of View:** 130° (fisheye lens)
- **Mounting:** Front-facing, upside-down (software-compensated with 180° rotation)
- **Capture Resolution:** 640 × 480 @ 30 FPS
- **Format:** MJPG (hardware-compressed in camera)
- **Interface:** USB 2.0 (`cv2.VideoCapture`, auto-detects index 0–3)

### 2.3 Distance Sensor
- **Model:** HC-SR04 Ultrasonic Sensor
- **Mounting:** Front-facing
- **GPIO Pins:** TRIG = GPIO4, ECHO = GPIO17
- **Poll Interval:** 100 ms (background thread)
- **Max Range:** 300 cm (hardware) / 200 cm (software cutoff)
- **Camera Offset Compensation:** 5.0 cm

### 2.4 Motor Driver & Drive Train
- **Driver:** Dual L298N H-Bridge Motor Drivers (×2 modules)
- **Motors:** 4 DC gear motors (4WD differential drive)

| Motor | Forward Pin | Backward Pin |
|-------|-------------|--------------|
| Front-Left (FL) | GPIO25 | GPIO27 |
| Front-Right (FR) | GPIO5 | GPIO15 |
| Rear-Left (RL) | GPIO26 | GPIO20 |
| Rear-Right (RR) | GPIO16 | GPIO6 |

- **Speed Control:** PWM duty cycle, range 0.0–1.0
- **Turning Model:** Tank-style differential (left/right speed split)

### 2.5 Network
- **Dashboard:** Wi-Fi, Flask HTTP server on port 5000
- **Push Notifications:** ntfy.sh (HTTPS PUT with image attachment)
  - Topic: `NTFY_TOPIC` env var (default: `vectorvance-sentry`)

---

## 3. Software Architecture

### 3.1 Entry Point & State Machine

`main.py` hosts the `AutonomousVehicle` class, which owns all subsystem instances and runs the main control loop. The vehicle operates as a finite state machine with the following states:

| State | Mode Name | Description |
|-------|-----------|-------------|
| `LANE_FOLLOW` | LANE | Camera-based lane following with PID steering |
| `FREE_ROAM` | FSD | Lane-free forward drive with obstacle avoidance |
| `TRACKING` | TRACK | KCF object tracking, pursues a detected COCO target |
| `MANUAL` | MANUAL | Keyboard or web dashboard direct motor control |
| `SENTRY` | SENTRY | Motors locked; camera-based surveillance active |

State transitions are triggered either by the web dashboard command API or keyboard shortcuts during operation.

### 3.2 Thread Architecture

The system uses four concurrent threads:

| Thread | Class | Purpose |
|--------|-------|---------|
| Main | `AutonomousVehicle.run()` | Frame processing, control loop, display |
| Sensor | `FrontSensor` | HC-SR04 polling at 10 Hz |
| Camera | `CameraGrabber` | USB frame draining (prevents stale frames) |
| Web | `pi_server` Flask | HTTP dashboard, stream, and API server |

The `CameraGrabber` class implements a latest-frame policy: it continuously drains the camera buffer in a background thread so the main loop always gets the most recent frame without blocking on `cv2.VideoCapture.read()`.

### 3.3 Data Flow

```
USB Camera (640×480 MJPG)
        │
        ▼
CameraGrabber (latest-frame thread)
        │
        ▼
cv2.ROTATE_180 (upside-down mount compensation)
        │
        ▼
FisheyeCamera.undistort() (130° → rectilinear)
        │
        ├──────────────────────────┬───────────────────────┐
        ▼                          ▼                       ▼
 LaneDetector               SignDetector             SentryMonitor
 (LANE mode)          (stop sign + ArUco speed)     (SENTRY mode)
        │                          │
        ▼                          ▼
 PIDController            speed_limit_kmh
        │
        ▼
 AdaptiveSpeedController ◄── ObstacleDetector (HC-SR04 front)
        │
        ▼
  L298N Motor Output (left / right duty cycles)
```

---

## 4. Perception System

### 4.1 Camera Undistortion (`camera.py`)

The Innomaker 130° fisheye lens introduces significant barrel distortion. `FisheyeCamera` applies OpenCV's equidistant fisheye model before any vision processing:

- **Projection model:** `r = f × θ` (equidistant)
- **Default focal length:** `f ≈ 212 px` (computed from 130° FOV at 640×480)
- **Distortion coefficients (default):** `k1 = −0.15, k2 = 0.04, k3 = k4 = 0.0`
- **Principal point:** `(320, 240)` (image center)
- **Balance factor:** `0.5` (crop/border tradeoff)
- **Remap method:** `cv2.remap()` with `INTER_LINEAR`, precomputed distortion maps
- **Calibration:** Optional external calibration via `.npz` file (9×6 checkerboard, 15–30 images recommended)

### 4.2 Lane Detection (`perception.py`)

`LaneDetector` operates in HLS color space to isolate white lane markings:

**Color Masking (HLS):**

| Channel | Min | Max |
|---------|-----|-----|
| H | 0 | 255 |
| L | 170 | 255 |
| S | 0 | 50 |

White pixels are additionally captured by a grayscale brightness threshold (`gray > 200`). Both masks are combined with bitwise OR and cleaned with morphological close and dilate operations.

**Region of Interest (ROI):**
A trapezoidal mask isolates the lower portion of the frame to exclude sky and distant clutter. Given the 130° FOV, the trapezoid uses a deliberately wide top: ±70% of frame width at 45% height, narrowing to full width at the bottom 55%.

**Lane Fitting:**
- 9 sliding windows (margin: ±70 px, min 20 pixels to re-center)
- Degree-2 polynomial fit: `x = A·y² + B·y + C` via `np.polyfit`
- Control point evaluated at `y = 0.75 × height`
- Steering error = frame center − lane x at control point

**EMA Smoothing:**
- Polynomial coefficients: `α = 0.25`
- Center position: `α = 0.15`
- Steering error: `α = 0.15`
- Confidence display: `α = 0.08`
- Error history: weighted 5-frame deque (weights 0.5 → 1.0, recency-biased)

**Confidence System:**
Lane confidence increases by +0.20 per successful detection frame and decays by −0.05 when the lane is lost (range: 0.0–1.0). Confidence gates the EMA update weight.

### 4.3 PID Steering Controller (`controller.py`)

```
Kp = 0.004    Ki = 0.00008    Kd = 0.002
Output limits: [−1.0, +1.0]
```

- **Derivative filter:** 5-sample rolling average to suppress sensor noise
- **Anti-windup:** Integral accumulation paused at saturation, or when error pushes back into range
- **Integral clamp:** ±200.0
- **dt:** measured per-frame, clamped to [0.001, 1.0] s; defaults to 20 ms on first frame

### 4.4 Adaptive Speed Control (`speed_controller.py`)

Speed is reduced based on the magnitude of the current steering error (a proxy for curvature):

| Curvature Zone | Error Threshold | Speed Multiplier |
|----------------|-----------------|------------------|
| Straight | < 30 px | 100% |
| Gentle curve | 30–80 px | 85% |
| Moderate curve | 80–150 px | 65% |
| Sharp curve | > 150 px | 50% |

Final speed: `base_speed × curvature_multiplier × obstacle_modifier`, clamped to `[min_speed=0.25, max_speed=0.8]`.

---

## 5. Safety System (`safety.py`)

`ObstacleDetector` consumes HC-SR04 readings from `FrontSensor` and provides three outputs:

| Distance | Action | Speed Modifier |
|----------|--------|---------------|
| < 20 cm | Emergency stop | 0.0 |
| 20–50 cm | Progressive slow | Linear ramp 0.0 → 1.0 |
| > 50 cm | Normal operation | 1.0 |

The sensor dict supports three virtual sensor positions — front, front-left, front-right — though only the front HC-SR04 is physically installed. The emergency stop threshold of 20 cm also triggers the `FreeRoamController` backup maneuver in FSD mode.

---

## 6. Sign Detection (`sign_detector.py`)

Sign detection uses a two-stage pipeline: classical CV for STOP signs (fast, no DNN overhead) and ArUco markers for speed limit signs (< 1 ms).

### 6.1 STOP Sign Detection

**Stage 1 — Red HSV Segmentation:**
Detects red blobs in two hue ranges to handle the H-channel wrap-around:
- Range 1: H ∈ [0°, 10°], S ≥ 50, V ≥ 50
- Range 2: H ∈ [170°, 180°], S ≥ 50, V ≥ 50

**Stage 2 — Shape Validation:**

| Check | Criterion |
|-------|-----------|
| Area | 400–60,000 px² |
| Aspect ratio | 0.75 < w/h < 1.35 |
| Side count | 6–12 (ε = 0.025 × perimeter) |
| Solidity | ≥ 0.75 |
| White interior | ≥ 5% of center region (gray > 160) |
| Circularity | ≥ 0.55 |

**Stage 3 — Temporal Filtering:**
- History: 8-frame deque
- Minimum 2 consecutive detections within 50 px of each other
- Confidence score: weighted sum across shape metrics (minimum 0.45)

### 6.2 ArUco Speed Limit Detection

| Marker ID | Speed Limit | Motor Fraction |
|-----------|-------------|---------------|
| 1 | 10 km/h | 0.30 |
| 2 | 20 km/h | 0.50 |
| 3 | 30 km/h | 0.75 |
| 4 | 50 km/h | 1.00 |

- **Dictionary:** `DICT_4X4_1000`
- **Hold duration:** 45 frames after last sighting
- **Detection latency:** < 1 ms (OpenCV native ArUco)

---

## 7. Object Tracking (`item_tracker.py`)

In `TRACKING` mode, `ItemTracker` uses OpenCV's KCF (Kernelized Correlation Filters) tracker to lock onto a user-selected or DNN-detected COCO object.

- **Initial detection:** SSD MobileNet v2 COCO (67 COCO classes supported)
- **Tracking algorithm:** KCF (fast, CPU-efficient)
- **Lost target behavior:** Pan-and-search rotation to find target; resume chase on re-acquisition
- **Target selection:** Via dashboard click (`track_click` command) or natural-language synonym map (e.g., "ball" → "sports ball")

---

## 8. Surveillance Mode (`sentry.py`)

When in `SENTRY` state, all motors are locked and the camera monitors for three event types:

### 8.1 Motion Detection
- Frame differencing (Gaussian blur 21×21, threshold 25)
- Morphological dilation (2 iterations)
- Minimum contour area: 1,200 px²

### 8.2 Person/Object Detection
- **Model:** SSD MobileNet v2 COCO (shared DNN instance)
- **Confidence threshold:** 0.50
- **Inference cadence:** Every 8 frames (performance budget for Pi 3)
- **Tracked classes:** person, bicycle, car, motorcycle, bus, truck, bird, cat, dog

### 8.3 Fire Detection
- HSV flame color thresholding:
  - Orange/yellow core: H ∈ [0°, 40°], S ≥ 120, V ≥ 150
  - Red wrap: H ∈ [170°, 180°], S ≥ 120, V ≥ 150
- Activation threshold: ≥ 1.5% of frame pixels match

### 8.4 Alert System

| Event | ntfy Priority | Tags |
|-------|-------------|------|
| Fire | 5 (urgent) | rotating_light, fire |
| Person detected | 4 (high) | bust_in_silhouette, warning |
| Motion detected | 3 (default) | wave |

- **Cooldown:** 30 s per event type (prevents alert flooding)
- **Snapshots:** Saved to `/home/pi/sentry_snaps/` as timestamped JPEGs
- **Event log:** Capped at 100 entries, served via `/api/sentry_events`

---

## 9. Web Dashboard (`pi_server.py`)

A Flask-based dashboard runs on port 5000, accessible from any browser on the local network.

### 9.1 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Dashboard HTML (full UI) |
| `/video_feed` | GET | MJPEG stream (15 FPS, JPEG quality 50) |
| `/api/status` | GET | Full telemetry JSON |
| `/api/command` | POST | Issue control commands |
| `/api/ai_drive` | POST | Natural language drive command |
| `/api/ai_track` | POST | Natural language target selection |
| `/api/sentry_events` | GET | Sentry event log |
| `/api/ping` | GET | Health check |

### 9.2 Telemetry Fields (sampled at 10 Hz)

The `/api/status` response includes: drive mode, car state, motor speeds (L/R), base speed, steering error (px), front distance (cm), FPS, frame count, stop sign state, ArUco speed limit, obstacle modifier, tracking state and target, and sentry detection flags (motion/person/fire).

### 9.3 Command API

Supported command actions: `toggle_auto`, `emergency_stop`, `reset`, `set_speed`, `manual_drive` (WASD), `set_mode`, `set_track_target`, `track_click`, `arm_sentry`, `disarm_sentry`, `clear_sentry_events`.

### 9.4 Voice Control
The dashboard uses the browser Web Speech API. Spoken commands are parsed client-side and sent to `/api/ai_drive` or `/api/ai_track`. Keyword-based NLU handles direction, duration qualifiers ("quickly" → 0.5 s, "for a long time" → 3.0 s), and target synonyms.

---

## 10. FSD (Free Self-Drive) Mode

`FreeRoamController` implements lane-free navigation using only the front ultrasonic sensor:

| Parameter | Value |
|-----------|-------|
| `BASE_SPEED` | 0.55 |
| `TURN_SPEED` | 0.65 |
| `COMMIT_FRAMES` | 22 (≈ 0.8 s at 28 FPS) |
| `BACKUP_FRAMES` | 12 |

Logic: drive forward until obstacle detected < 20 cm → backup for `BACKUP_FRAMES` → select turn direction (away from obstacle) → commit to turn for `COMMIT_FRAMES` → resume forward.

---

## 11. DNN Inference Engine (`dnn_detector.py`)

- **Model:** SSD MobileNet v2 COCO (TensorFlow frozen graph)
  - `ssd_mobilenet_v2_coco.pb` (~67 MB)
  - `ssd_mobilenet_v2_coco.pbtxt` (~8 KB)
- **Inference backend:** OpenCV DNN (`cv2.dnn.readNetFromTensorflow`)
- **Input:** 300×300 RGB, mean subtraction (127.5, 127.5, 127.5), scale 1/127.5
- **Default skip:** Every 8 frames (tunable via `--dnn-skip`)
- **Shared instance:** `SignDetector` and `SentryMonitor` share one loaded net to avoid duplicate memory use (~250 MB on Pi 3)

---

## 12. Key Software Parameters Summary

| Parameter | Value | Module |
|-----------|-------|--------|
| Camera resolution | 640 × 480 | `main.py` |
| Camera FPS target | 30 | `main.py` |
| Frame rotate | 180° | `main.py` |
| Fisheye FOV | 130° | `camera.py` |
| Balance factor | 0.5 | `camera.py` |
| HC-SR04 poll interval | 100 ms | `main.py` (FrontSensor) |
| Emergency stop distance | 20 cm | `safety.py` |
| Slow-down distance | 50 cm | `safety.py` |
| PID Kp / Ki / Kd | 0.004 / 0.00008 / 0.002 | `main.py` |
| Adaptive speed min / max | 0.25 / 0.80 | `speed_controller.py` |
| Lane EMA α (error) | 0.15 | `perception.py` |
| Lane sliding windows | 9 | `perception.py` |
| STOP sign confidence min | 0.45 | `sign_detector.py` |
| STOP sign temporal frames | 8 (need 2 hits) | `sign_detector.py` |
| DNN inference skip | 8 frames | `dnn_detector.py` |
| Dashboard stream FPS | 15 | `pi_server.py` |
| Sentry alert cooldown | 30 s | `sentry.py` |
| Sentry fire pixel ratio | 1.5% | `sentry.py` |
| Motor EMA α (display) | 0.28 | `main.py` |

---

## 13. Dependencies

```bash
pip install opencv-python numpy flask gpiozero lgpio --break-system-packages
```

| Package | Usage |
|---------|-------|
| `opencv-python` | Camera capture, fisheye undistortion, lane CV, DNN inference, ArUco |
| `numpy` | Array math, polynomial fitting, HSV masking |
| `flask` | Web dashboard, REST API, MJPEG stream |
| `lgpio` | GPIO PWM (HC-SR04, L298N motor control) |
| `gpiozero` | GPIO abstraction layer |

DNN model files are downloaded separately via:
```bash
python src/dnn_detector.py --download
```

---

## 14. Limitations & Known Constraints

- **Platform:** Raspberry Pi 3 constrains frame rate; DNN inference is run every 8 frames to maintain ≥ 25 FPS in the main loop.
- **Single camera:** Only one USB camera is present (front-facing). No rear view during forward travel.
- **Speed limit DNN:** ArUco markers substitute for a real speed-limit sign model; a custom-trained classifier would be needed for printed signs.
- **Lane detection:** Relies on white tape markings under consistent indoor lighting. Performance degrades under shadows or reflective surfaces.
- **Obstacle avoidance:** Single front ultrasonic sensor provides no lateral obstacle awareness.
- **No wheel odometry:** No encoders; speed is open-loop duty cycle only.
- **`rear_monitor.py`:** Exists in `src/` but is not imported or used. Considered dead code.

---

*VectorVance — April 2026*
