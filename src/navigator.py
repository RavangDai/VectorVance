"""
navigator.py - VectorVance Turn-by-Turn Navigation
───────────────────────────────────────────────────
Takes a route from pathfinder.py and feeds commands to the car
one step at a time as it drives.

The navigator is a state machine:
  LANE_FOLLOW  → normal lane-following (perception.py handles steering)
  APPROACHING  → intersection detected, prepare for turn
  TURNING      → executing a turn (override steering)
  ARRIVED      → destination reached, stop

USAGE:
    from track_map import build_default_track
    from pathfinder import find_shortest_path
    from navigator import Navigator

    track = build_default_track()
    route = find_shortest_path(track, "START", "DEST")
    nav = Navigator(route)

    # In your main loop:
    command = nav.get_current_command()
    nav.update(distance_traveled, intersection_detected)
"""

import time


class NavigatorState:
    LANE_FOLLOW = "LANE_FOLLOW"
    APPROACHING = "APPROACHING"
    TURNING     = "TURNING"
    ARRIVED     = "ARRIVED"
    NO_ROUTE    = "NO_ROUTE"


class Navigator:
    """
    Real-time navigation controller.
    Feeds one command at a time to the driving loop.
    """

    def __init__(self, route=None):
        """
        route: output from pathfinder.find_shortest_path()
               Must have 'found', 'commands', 'path', 'distance'.
        """
        self.route = route
        self.commands = route["commands"] if route and route.get("found") else []
        self.path = route["path"] if route and route.get("found") else []
        self.total_distance = route["distance"] if route and route.get("found") else 0

        self.current_step = 0           # index into self.commands
        self.state = NavigatorState.NO_ROUTE

        # Distance tracking within current segment
        self.segment_distance = 0.0     # how far traveled on current segment
        self.total_traveled = 0.0       # total traveled so far

        # Timing for turn execution
        self._turn_start_time = None
        self.turn_duration = 1.5        # seconds to hold a turn (tune for your car)

        # Approach zone: how far before expected distance to start looking for fork
        self.approach_margin = 10.0     # cm — start looking 10cm before expected fork

        if self.commands:
            self.state = NavigatorState.LANE_FOLLOW
            print(f"Navigator: Route loaded — {len(self.commands)} segments, "
                  f"{self.total_distance:.0f}cm total")
            self._print_current_step()
        else:
            print("Navigator: No route loaded")

    # ─────────────────────────────────────────────────────────────────
    #  MAIN UPDATE (call every frame)
    # ─────────────────────────────────────────────────────────────────
    def update(self, distance_delta=0.0, intersection_detected=False):
        """
        Call every frame with:
          distance_delta:        estimated cm traveled since last frame
          intersection_detected: True if perception sees a lane fork

        Returns the current NavigatorState.
        """
        if self.state == NavigatorState.ARRIVED:
            return self.state
        if self.state == NavigatorState.NO_ROUTE:
            return self.state

        self.segment_distance += distance_delta
        self.total_traveled += distance_delta

        cmd = self._current_cmd()
        if cmd is None:
            self.state = NavigatorState.ARRIVED
            return self.state

        # ── STATE MACHINE ────────────────────────────────────────────

        if self.state == NavigatorState.LANE_FOLLOW:
            # Check if we're approaching the expected distance for next node
            expected = cmd["distance"]
            remaining = expected - self.segment_distance

            if remaining < self.approach_margin:
                # Getting close — start watching for the fork
                self.state = NavigatorState.APPROACHING

            # Also trigger if intersection is detected early
            if intersection_detected and cmd["action"] != "STRAIGHT":
                self.state = NavigatorState.APPROACHING

        elif self.state == NavigatorState.APPROACHING:
            if intersection_detected or self.segment_distance >= cmd["distance"]:
                # Fork detected or distance reached — execute turn
                if cmd["action"] == "STRAIGHT":
                    # No turn needed, advance to next segment
                    self._advance_segment()
                else:
                    # Start turning
                    self.state = NavigatorState.TURNING
                    self._turn_start_time = time.time()
                    print(f"Navigator: Executing {cmd['action']} → {cmd['to']}")

        elif self.state == NavigatorState.TURNING:
            # Hold the turn for a fixed duration
            elapsed = time.time() - self._turn_start_time
            if elapsed >= self.turn_duration:
                # Turn complete — move to next segment
                self._advance_segment()

        return self.state

    # ─────────────────────────────────────────────────────────────────
    #  QUERIES (read current state)
    # ─────────────────────────────────────────────────────────────────
    def get_current_command(self):
        """
        Returns the current command dict, or None if done.
        {"action": "TURN_LEFT", "to": "WP_LEFT", "distance": 25}
        """
        return self._current_cmd()

    def get_turn_override(self):
        """
        If currently TURNING, return the turn direction.
        The main loop should override PID steering with this.
        Returns: "TURN_LEFT" | "TURN_RIGHT" | None
        """
        if self.state == NavigatorState.TURNING:
            cmd = self._current_cmd()
            if cmd and cmd["action"] in ("TURN_LEFT", "TURN_RIGHT"):
                return cmd["action"]
        return None

    def get_progress(self):
        """Return navigation progress as a dict."""
        total_steps = len(self.commands)
        return {
            "state": self.state,
            "step": self.current_step + 1,
            "total_steps": total_steps,
            "segment_distance": self.segment_distance,
            "total_traveled": self.total_traveled,
            "total_route": self.total_distance,
            "percent": (self.total_traveled / self.total_distance * 100)
                       if self.total_distance > 0 else 0,
            "current_target": self.commands[self.current_step]["to"]
                              if self.current_step < total_steps else "DONE",
            "next_action": self.commands[self.current_step]["action"]
                           if self.current_step < total_steps else "ARRIVED",
        }

    def is_finished(self):
        return self.state == NavigatorState.ARRIVED

    # ─────────────────────────────────────────────────────────────────
    #  INTERNALS
    # ─────────────────────────────────────────────────────────────────
    def _current_cmd(self):
        if 0 <= self.current_step < len(self.commands):
            return self.commands[self.current_step]
        return None

    def _advance_segment(self):
        """Move to the next route segment."""
        self.current_step += 1
        self.segment_distance = 0.0
        self._turn_start_time = None

        if self.current_step >= len(self.commands):
            self.state = NavigatorState.ARRIVED
            print(f"Navigator: ✓ ARRIVED at destination! "
                  f"Total: {self.total_traveled:.1f}cm")
        else:
            self.state = NavigatorState.LANE_FOLLOW
            self._print_current_step()

    def _print_current_step(self):
        cmd = self._current_cmd()
        if cmd:
            icon = {"STRAIGHT": "⬆️",
                    "TURN_LEFT": "⬅️",
                    "TURN_RIGHT": "➡️"}.get(cmd["action"], "  ")
            print(f"Navigator: Step {self.current_step+1}/{len(self.commands)} "
                  f"— {icon} {cmd['action']} → {cmd['to']} ({cmd['distance']}cm)")

    def reset(self):
        """Reset navigation to start of route."""
        self.current_step = 0
        self.segment_distance = 0.0
        self.total_traveled = 0.0
        self._turn_start_time = None
        if self.commands:
            self.state = NavigatorState.LANE_FOLLOW
        else:
            self.state = NavigatorState.NO_ROUTE

    def draw_overlay(self, frame):
        """Draw navigation HUD on the debug frame."""
        import cv2

        h, w = frame.shape[:2]
        progress = self.get_progress()

        # ── Navigation box (top-center) ──────────────────────────────
        box_w, box_h = 280, 70
        box_x = (w - box_w) // 2
        box_y = 5

        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (box_x, box_y),
                      (box_x + box_w, box_y + box_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        cv2.rectangle(frame, (box_x, box_y),
                      (box_x + box_w, box_y + box_h), (0, 200, 255), 1)

        # State color
        state_colors = {
            NavigatorState.LANE_FOLLOW: (0, 255, 0),
            NavigatorState.APPROACHING: (0, 255, 255),
            NavigatorState.TURNING: (0, 165, 255),
            NavigatorState.ARRIVED: (255, 200, 0),
            NavigatorState.NO_ROUTE: (100, 100, 100),
        }
        color = state_colors.get(progress["state"], (255, 255, 255))

        # State label
        cv2.putText(frame, f"NAV: {progress['state']}",
                    (box_x + 8, box_y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Current target + action
        action_text = f"{progress['next_action']} -> {progress['current_target']}"
        cv2.putText(frame, action_text,
                    (box_x + 8, box_y + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        # Progress bar
        bar_x = box_x + 8
        bar_y2 = box_y + box_h - 10
        bar_w2 = box_w - 16
        cv2.rectangle(frame, (bar_x, bar_y2), (bar_x + bar_w2, bar_y2 + 6),
                      (50, 50, 50), -1)
        fill = int(bar_w2 * progress["percent"] / 100)
        cv2.rectangle(frame, (bar_x, bar_y2), (bar_x + fill, bar_y2 + 6),
                      color, -1)

        # Step counter
        step_text = f"Step {progress['step']}/{progress['total_steps']}"
        cv2.putText(frame, step_text,
                    (box_x + box_w - 100, box_y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

        # Distance
        dist_text = f"{progress['total_traveled']:.0f}/{progress['total_route']:.0f}cm"
        cv2.putText(frame, dist_text,
                    (box_x + box_w - 100, box_y + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

        # ── Big turn arrow (when turning) ────────────────────────────
        if progress["state"] == NavigatorState.TURNING:
            arrow = progress["next_action"]
            if arrow == "TURN_LEFT":
                cv2.arrowedLine(frame, (w//2 + 40, h//2),
                                (w//2 - 60, h//2), (0, 165, 255), 4, tipLength=0.3)
                cv2.putText(frame, "TURN LEFT",
                            (w//2 - 80, h//2 - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
            elif arrow == "TURN_RIGHT":
                cv2.arrowedLine(frame, (w//2 - 40, h//2),
                                (w//2 + 60, h//2), (0, 165, 255), 4, tipLength=0.3)
                cv2.putText(frame, "TURN RIGHT",
                            (w//2 - 80, h//2 - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

        elif progress["state"] == NavigatorState.ARRIVED:
            cv2.putText(frame, "ARRIVED!",
                        (w//2 - 80, h//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 100), 3)

        return frame


# ─────────────────────────────────────────────────────────────────────
#  SELF-TEST (simulate a drive)
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from track_map import build_default_track
    from pathfinder import find_shortest_path, print_route

    track = build_default_track()
    route = find_shortest_path(track, "START", "DEST")
    print_route(route, "ROUTE TO FOLLOW")

    nav = Navigator(route)

    print("\n── SIMULATING DRIVE ──")
    frame_count = 0
    speed_cm_per_frame = 1.0   # simulate ~1cm per frame

    while not nav.is_finished():
        frame_count += 1

        # Simulate intersection detection at roughly the right distance
        cmd = nav.get_current_command()
        fake_intersection = False
        if cmd and nav.segment_distance >= cmd["distance"] - 5:
            if cmd["action"] != "STRAIGHT":
                fake_intersection = True

        state = nav.update(
            distance_delta=speed_cm_per_frame,
            intersection_detected=fake_intersection
        )

        # Check if navigator wants to override steering
        turn = nav.get_turn_override()
        if turn:
            print(f"  Frame {frame_count}: OVERRIDE steering → {turn}")

        # Simulate time passing for turns
        if state == NavigatorState.TURNING:
            time.sleep(0.05)  # speed up simulation

        if frame_count > 500:
            print("  (safety cutoff — too many frames)")
            break

    progress = nav.get_progress()
    print(f"\nFinal: {progress['state']} — "
          f"Traveled {progress['total_traveled']:.1f}cm / "
          f"{progress['total_route']:.1f}cm")