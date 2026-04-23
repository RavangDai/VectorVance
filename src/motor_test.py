#!/usr/bin/env python3
"""
motor_test.py — Standalone motor diagnostic for VectorVance (Raspberry Pi 3)

Run directly on the Pi:
    python motor_test.py

Tests every motor for direction, dead-zone (minimum start speed),
side symmetry, and pivot turns. Prints a report at the end — share
that output to get accurate PID / turning values.
"""

import time
import sys

try:
    from gpiozero import Motor
except ImportError:
    print("ERROR: gpiozero not installed.")
    print("       Run: pip install gpiozero --break-system-packages")
    sys.exit(1)

# ── Pin assignments — must match main.py ────────────────────────────────────
#   Motor(forward=FWD_PIN, backward=BWD_PIN)
#   NOTE: .backward(spd) = car moves FORWARD  (wiring convention in this build)
#         .forward(spd)  = car moves BACKWARD
MOTORS = {
    "LF": (25, 27),
    "LB": (26, 20),
    "RF": ( 5, 15),
    "RB": (16,  6),
}

HOLD   = 1.5   # seconds to hold each test
PAUSE  = 0.6   # seconds between tests

# ── Helpers ──────────────────────────────────────────────────────────────────

def sep(char="─", n=56): print(char * n)

def ask(q):
    while True:
        a = input(f"    {q}  [y/n]: ").strip().lower()
        if a in ("y","yes"): return True
        if a in ("n","no"):  return False

def wait(msg=""):
    input(f"\n  >>> {msg}  Press ENTER to continue <<<")

def make_motor(name):
    fwd, bwd = MOTORS[name]
    return Motor(forward=fwd, backward=bwd)

def stop_all(motors: dict):
    for m in motors.values():
        try: m.stop()
        except: pass

def close_all(motors: dict):
    stop_all(motors)
    for m in motors.values():
        try: m.close()
        except: pass

# ── Drive helpers (mirrors main.py _drive_manual convention) ─────────────────

def drive_side(motor_a, motor_b, speed: float):
    """speed > 0 = car forward, speed < 0 = car backward."""
    if abs(speed) < 0.02:
        motor_a.stop(); motor_b.stop()
    elif speed > 0:
        motor_a.backward(speed); motor_b.backward(speed)
    else:
        motor_a.forward(-speed); motor_b.forward(-speed)

def drive_all(lf, lb, rf, rb, left: float, right: float):
    drive_side(lf, lb, left)
    drive_side(rf, rb, right)

# ── Phase 1: Individual motor tests ──────────────────────────────────────────

def test_individual(name):
    sep()
    fwd, bwd = MOTORS[name]
    print(f"  MOTOR {name}   GPIO fwd={fwd}  bwd={bwd}")
    sep()
    result = {"name": name, "fwd_ok": None, "bwd_ok": None, "dead_zone": None}
    m = make_motor(name)
    try:
        # Forward direction
        print(f"\n  [FORWARD] Running at 70% for {HOLD}s ...")
        m.backward(0.70)
        time.sleep(HOLD)
        m.stop(); time.sleep(PAUSE)
        result["fwd_ok"] = ask("Motor spun in the FORWARD (car-forward) direction?")

        # Backward direction
        print(f"\n  [BACKWARD] Running at 70% for {HOLD}s ...")
        m.forward(0.70)
        time.sleep(HOLD)
        m.stop(); time.sleep(PAUSE)
        result["bwd_ok"] = ask("Motor spun in the BACKWARD direction?")

        # Dead-zone sweep — find minimum speed that actually moves the motor
        print("\n  [DEAD-ZONE] Sweeping up from 10% until motor moves ...")
        dz = None
        for pct in range(10, 105, 5):
            spd = pct / 100.0
            print(f"    {pct}% ... ", end="", flush=True)
            m.backward(spd)
            time.sleep(0.5)
            m.stop()
            time.sleep(0.2)
            moved = ask("Did the motor move?")
            if moved:
                dz = spd
                print(f"    → Dead-zone threshold: {pct}%")
                break
        result["dead_zone"] = dz if dz is not None else "> 100%"

    finally:
        m.stop(); m.close()
    return result

# ── Phase 2: Side symmetry ────────────────────────────────────────────────────

def test_symmetry():
    sep()
    print("  PHASE 2 — Side Symmetry (wheels free in air or car on floor)")
    sep()
    results = {}

    for side, names in [("LEFT", ("LF","LB")), ("RIGHT", ("RF","RB"))]:
        print(f"\n  Running {side} side (both wheels) at 60% for {HOLD}s ...")
        motors = {n: make_motor(n) for n in names}
        try:
            for m in motors.values(): m.backward(0.60)
            time.sleep(HOLD)
            stop_all(motors); time.sleep(PAUSE)
            results[side] = ask(f"Do {names[0]} and {names[1]} spin at the SAME speed?")
        finally:
            close_all(motors)

    # Left vs right balance
    print(f"\n  Running ALL 4 motors at 60% (straight forward) for {HOLD*1.5:.0f}s ...")
    motors = {n: make_motor(n) for n in MOTORS}
    try:
        for m in motors.values(): m.backward(0.60)
        time.sleep(HOLD * 1.5)
        stop_all(motors); time.sleep(PAUSE)
        results["BALANCE"] = ask("Does the car go STRAIGHT (no drift left/right)?")
        if not results["BALANCE"]:
            drift = input("    Which way does it drift?  [L]eft / [R]ight: ").strip().upper()
            results["DRIFT"] = "LEFT" if drift.startswith("L") else "RIGHT"
        else:
            results["DRIFT"] = "NONE"
    finally:
        close_all(motors)

    return results

# ── Phase 3: Turning tests ────────────────────────────────────────────────────

def test_turns():
    sep()
    print("  PHASE 3 — Turning Tests  (place car on floor with space)")
    sep()
    wait("Place car on floor")
    results = []

    TURN_TESTS = [
        ("Straight forward (60/60)",      0.60,  0.60, 2.0),
        ("Gentle RIGHT lane turn (60/30)", 0.60,  0.30, 2.0),
        ("Gentle LEFT  lane turn (30/60)", 0.30,  0.60, 2.0),
        ("Sharp RIGHT  lane turn (60/10)", 0.60,  0.10, 2.0),
        ("Sharp LEFT   lane turn (10/60)", 0.10,  0.60, 2.0),
        ("Pivot RIGHT spin (CW)  (40/-40)",0.40, -0.40, 2.5),
        ("Pivot LEFT  spin (CCW)(-40/40)",-0.40,  0.40, 2.5),
    ]

    for label, l_spd, r_spd, hold in TURN_TESTS:
        print(f"\n  TEST: {label}")
        print(f"        Left={int(l_spd*100)}%  Right={int(r_spd*100)}%  for {hold}s")
        motors = {n: make_motor(n) for n in MOTORS}
        lf, lb = motors["LF"], motors["LB"]
        rf, rb = motors["RF"], motors["RB"]
        try:
            drive_all(lf, lb, rf, rb, l_spd, r_spd)
            time.sleep(hold)
            stop_all(motors); time.sleep(PAUSE)
            ok = ask("Result was correct / car behaved as expected?")
            note = ""
            if not ok:
                note = input("    Brief note (e.g. 'drifts left', 'too sharp'): ").strip()
            results.append({"label": label, "l": l_spd, "r": r_spd, "ok": ok, "note": note})
        finally:
            close_all(motors)

    return results

# ── Report ────────────────────────────────────────────────────────────────────

def print_report(ind, sym, turns):
    print()
    sep("═")
    print("  VECTORVANCE MOTOR DIAGNOSTIC REPORT")
    print(f"  {time.strftime('%Y-%m-%d  %H:%M:%S')}")
    sep("═")

    print("\n  1. INDIVIDUAL MOTORS")
    print(f"  {'Motor':<6} {'FWD dir':<14} {'BWD dir':<14} {'Dead-zone'}")
    sep()
    for r in ind:
        fwd = "✓ correct" if r['fwd_ok']  else "✗ WRONG"
        bwd = "✓ correct" if r['bwd_ok']  else "✗ WRONG"
        dz  = f"{int(float(r['dead_zone'])*100)}%" if isinstance(r['dead_zone'], float) else str(r['dead_zone'])
        print(f"  {r['name']:<6} {fwd:<14} {bwd:<14} {dz}")

    print("\n  2. SIDE SYMMETRY")
    sep()
    for key, label in [("LEFT","Left side (LF=LB)"), ("RIGHT","Right side (RF=RB)"),
                        ("BALANCE","Both sides balanced")]:
        val = sym.get(key)
        tag = "✓ matched" if val else "✗ MISMATCH"
        print(f"  {label:<28}: {tag}")
    print(f"  {'Straight drift':<28}: {sym.get('DRIFT','?')}")

    print("\n  3. TURNING RESULTS")
    sep()
    for t in turns:
        tag  = "✓ OK" if t['ok'] else "✗ FAIL"
        note = f"  → {t['note']}" if t['note'] else ""
        print(f"  {t['label']:<40} {tag}{note}")

    print()
    sep("═")
    print("  COPY EVERYTHING ABOVE AND SHARE IT FOR MOTOR TUNING.")
    sep("═")
    print()

# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    sep("═")
    print("  VectorVance — Motor Diagnostic")
    print("  Raspberry Pi 3  |  Dual L298N  |  4× DC motors")
    sep("═")
    print("""
  This test runs in 3 phases:
    1. Each motor alone  — direction + minimum start speed (dead-zone)
    2. Side symmetry     — do both wheels on each side match?
    3. Turning tests     — straight, gentle turn, pivot spin

  PHASE 1+2: Put the car on a raised surface so wheels spin freely.
  PHASE 3:   Put the car on the floor with room to move.

  Answer y/n after each movement prompt.
  At the end you get a report — share it for tuning.
""")
    wait("Ready? Place car on raised surface (wheels free)")

    individual = []
    for name in MOTORS:
        individual.append(test_individual(name))
        time.sleep(0.5)

    symmetry = test_symmetry()
    turns    = test_turns()

    print_report(individual, symmetry, turns)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n  Ctrl+C — aborted.  Motors stopped.")
        sys.exit(0)
