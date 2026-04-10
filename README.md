# VectorVance 🚗

An autonomous lane-following car built on Raspberry Pi. Uses computer vision and a MobileNet SSD to navigate a track, stop at signs, and choose coloured tape paths at forks.

## Features

- **Lane following** — PID steering with EMA-smoothed Hough lane detection
- **Object detection** — stop signs, traffic lights, pedestrians, vehicles (SSD MobileNet v2, OpenCV DNN)
- **Fork navigation** — stops at forks, user picks GREEN or BLUE path via dashboard
- **Destination** — automatically stops when RED tape is reached
- **Obstacle avoidance** — HC-SR04 ultrasonic + DNN proximity for adaptive speed
- **Fisheye correction** — undistorts 160° FOV lens before any vision processing
- **FSD free-roam mode** — obstacle-avoiding drive without lane detection
- **Live dashboard** — web UI with MJPEG stream, telemetry, and remote controls

## Hardware

- Raspberry Pi (lgpio GPIO)
- DORHEA USB fisheye camera — 160° FOV, CMOS, 640×480
- 4× DC motors via dual L298N drivers (FL, FR, RL, RR)
- HC-SR04 ultrasonic sensor (TRIG=GPIO4, ECHO=GPIO17)

## Run

```bash
python main.py                        # full autonomous
python main.py --no-display           # headless (no cv2 window)
python main.py --no-web               # disable web dashboard
python main.py --speed 0.6            # set max speed (default 0.8)
python main.py --fov 160              # FOV override (default 160 for DORHEA)
python main.py --dnn-skip 3           # DNN inference every N frames
python main.py --no-undistort         # skip fisheye correction
python main.py --calibration calib.npz  # use calibrated lens params
```

Open dashboard at `http://<pi-ip>:5000/`

For PC testing (webcam or video file):

```bash
python mainv2.py
python mainv2.py --video path/to/video.mp4
```

## Install

```bash
pip install opencv-python numpy flask gpiozero lgpio --break-system-packages
```

Download DNN model files once:

```bash
python src/dnn_detector.py --download
```

This places `ssd_mobilenet_v2_coco.pb` (~67 MB) and `ssd_mobilenet_v2_coco.pbtxt` in `src/`.

## Fisheye Calibration (optional but recommended)

```bash
python src/camera.py --calibrate --images ./calibration_images/ --fov 160
python main.py --calibration calibration.npz
```

## Keyboard Shortcuts (display enabled)

| Key   | Action                        |
|-------|-------------------------------|
| Q     | Quit                          |
| SPACE | Toggle autonomous / manual    |
| F     | Toggle FSD free-roam mode     |
| R     | Reset all systems             |
| S     | Save snapshot                 |
| D     | Print debug info              |
| G     | Set fork path to GREEN        |
| B     | Set fork path to BLUE         |

## Project Structure

```
src/
├── main.py                  # entry point (Pi)
├── mainv2.py                # entry point (PC testing)
├── camera.py                # fisheye undistortion (160° FOV)
├── perception.py            # lane detection
├── controller.py            # PID steering
├── speed_controller.py      # adaptive speed
├── dnn_detector.py          # SSD MobileNet v2 sign & obstacle detection
├── color_sign_detector.py   # colour tape navigation
├── intersection_detector.py # fork detection
├── safety.py                # ultrasonic obstacle detector
├── pi_server.py             # Flask web dashboard
└── webapp/templates/
    └── dashboard.html
```

## Dashboard Controls

| Action            | Button / Key         |
|-------------------|----------------------|
| Toggle autonomous | Dashboard / SPACE    |
| Emergency stop    | Dashboard button     |
| Pick fork path    | GREEN / BLUE buttons |
| Reset all systems | Dashboard / R        |
| Set max speed     | Speed slider         |
| Manual drive      | WASD keys            |

## State Machine

| State            | Description                                          |
|------------------|------------------------------------------------------|
| `LANE_FOLLOW`    | Normal PID lane following                            |
| `FORK_WAITING`   | Stopped at fork, waiting for user to pick a path     |
| `COLOR_FOLLOWING`| Steering toward chosen colour tape                   |
| `ARRIVED`        | RED destination tape confirmed — car stops           |
| `FREE_ROAM`      | FSD mode: obstacle-avoiding drive, no lane detection |
