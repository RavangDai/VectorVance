# VectorVance 🚗                                                                                                                                                                                                                  
  An autonomous lane-following car built on Raspberry Pi. Uses computer vision and YOLO to navigate a track, stop   at signs, and choose coloured tape paths at forks.
                                                                                                                  
  ## Features

  - **Lane following**  PID steering with EMA-smoothed Hough lane detection
  - **YOLO detection**  stop signs, traffic lights, pedestrians, vehicles (YOLOv8n)
  - **Fork navigation** stops at forks, user picks GREEN or BLUE path via dashboard
  - **Destination** automatically stops when RED tape is reached
  - **Obstacle avoidance** HC-SR04 ultrasonic + YOLO proximity for adaptive speed
  - **Live dashboard** web UI with MJPEG stream, telemetry, and remote controls

  ## Hardware

  - Raspberry Pi + Picamera2
  - 4× DC motors via dual L298N drivers
  - HC-SR04 ultrasonic sensor

  ## Run

  ```bash
  python main.py                  # full autonomous
  python main.py --no-display     # headless
  python main.py --no-web         # no dashboard

  Open dashboard at http://<pi-ip>:5000/

  For PC testing (webcam or video file):

  python mainv2.py
  python mainv2.py --yolo --video path/to/video.mp4

  Install

  pip install opencv-python numpy flask ultralytics --break-system-packages

  Place yolov8n.pt in the src/ directory.

  Project Structure

  src/
  ├── main.py                  # entry point (Pi)
  ├── mainv2.py                # entry point (PC testing)
  ├── perception.py            # lane detection
  ├── controller.py            # PID steering
  ├── speed_controller.py      # adaptive speed
  ├── yolo_detector.py         # YOLO sign & obstacle detection
  ├── color_sign_detector.py   # colour tape navigation
  ├── intersection_detector.py # fork detection
  ├── safety.py                # ultrasonic obstacle detector
  ├── pi_server.py             # Flask web dashboard
  └── webapp/templates/
      └── dashboard.html

  Dashboard Controls

  ┌───────────────────┬──────────────────────┐
  │      Action       │     Button / Key     │
  ├───────────────────┼──────────────────────┤
  │ Toggle autonomous │ Dashboard / SPACE    │
  ├───────────────────┼──────────────────────┤
  │ Emergency stop    │ Dashboard / —        │
  ├───────────────────┼──────────────────────┤
  │ Pick fork path    │ GREEN / BLUE buttons │
  ├───────────────────┼──────────────────────┤
  │ Reset all systems │ Dashboard / R        │
  ├───────────────────┼──────────────────────┤
  │ Set max speed     │ Speed slider         │
  ├───────────────────┼──────────────────────┤
  │ ```               │                      │
  └───────────────────┴──────────────────────┘

──────────────────────
