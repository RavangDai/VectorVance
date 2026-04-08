"""
camera.py — Fisheye lens undistortion for 160° wide-angle camera
─────────────────────────────────────────────────────────────────
A 160° FOV camera introduces heavy barrel distortion that bends
straight lane lines into curves, throwing off Hough detection and
PID steering.  This module corrects that before any other module
sees the frame.

How it works
────────────
Uses OpenCV's fisheye model (equidistant projection):
  r = f * θ   (r = pixel radius, f = focal length, θ = angle from centre)

Default K and D are APPROXIMATE for a generic 160° lens at 640×480.
They will work but for best accuracy run a one-time calibration:

  python camera.py --calibrate --images ./calibration_images/
  python camera.py --calibrate --images ./calibration_images/ --board 9x6

Then pass the saved .npz file to main.py:
  python main.py --calibration calibration.npz

balance parameter (0.0–1.0)
  0.0 → crop to only valid pixels (no black borders, smaller FOV)
  1.0 → keep all pixels (black corners, maximum FOV)
  Recommended: 0.0–0.3 for lane following (cleaner image centre)
"""

import math
import os
import cv2
import numpy as np


# ── Default approximate parameters for a 160° lens at 640×480 ────────────────
# Equidistant model: f = half_width / (half_fov_rad)
# half_fov_rad = 80° = 1.396 rad → f = 320 / 1.396 ≈ 229
_DEFAULT_FOV_DEG = 160
_DEFAULT_W, _DEFAULT_H = 640, 480

def _default_K(w=_DEFAULT_W, h=_DEFAULT_H, fov_deg=_DEFAULT_FOV_DEG) -> np.ndarray:
    f = (w / 2) / math.tan(math.radians(fov_deg / 2))
    return np.array([[f,   0., w / 2],
                     [0.,  f,  h / 2],
                     [0.,  0., 1.   ]], dtype=np.float64)

# Estimated distortion coefficients (k1, k2, k3, k4) for fisheye model
_DEFAULT_D = np.array([[-0.38], [0.14], [-0.03], [0.0]], dtype=np.float64)


class FisheyeCamera:
    """
    Handles fisheye undistortion for a wide-angle camera.

    Usage:
        cam = FisheyeCamera()                         # approximate params
        cam = FisheyeCamera.from_file("calib.npz")   # calibrated params
        corrected = cam.undistort(raw_frame)
    """

    def __init__(self,
                 width: int   = _DEFAULT_W,
                 height: int  = _DEFAULT_H,
                 fov_deg: float = _DEFAULT_FOV_DEG,
                 K: np.ndarray = None,
                 D: np.ndarray = None,
                 balance: float = 0.0):
        """
        width, height : frame resolution (must match camera config)
        fov_deg       : camera field of view in degrees (160 for your lens)
        K             : 3×3 camera matrix (uses approximation if None)
        D             : 4×1 fisheye distortion coefficients (approx if None)
        balance       : 0.0 = crop black borders, 1.0 = keep all pixels
        """
        self.width   = width
        self.height  = height
        self.fov_deg = fov_deg
        self.balance = balance

        self.K = K if K is not None else _default_K(width, height, fov_deg)
        self.D = D if D is not None else _DEFAULT_D.copy()

        self._build_maps()

        mode = "calibrated" if K is not None else "approximate"
        print(f"[Camera] FisheyeCamera — {fov_deg}° FOV, {mode} params, "
              f"balance={balance:.1f}")

    def _build_maps(self):
        """Precompute undistortion maps (done once, fast to apply each frame)."""
        size = (self.width, self.height)
        self._new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            self.K, self.D, size, np.eye(3), balance=self.balance
        )
        self._map1, self._map2 = cv2.fisheye.initUndistortRectifyMap(
            self.K, self.D, np.eye(3), self._new_K, size, cv2.CV_16SC2
        )

    def undistort(self, frame: np.ndarray) -> np.ndarray:
        """Return undistorted copy of frame.  Input shape must match width×height."""
        return cv2.remap(
            frame, self._map1, self._map2,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )

    @classmethod
    def from_file(cls, path: str, **kwargs) -> "FisheyeCamera":
        """
        Load calibrated K and D from a .npz file saved by calibrate().
        Extra kwargs (fov_deg, balance, etc.) override saved values.
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Calibration file not found: {path}")
        data = np.load(path)
        w    = int(data.get("width",  _DEFAULT_W))
        h    = int(data.get("height", _DEFAULT_H))
        fov  = float(data.get("fov_deg", _DEFAULT_FOV_DEG))
        return cls(width=w, height=h, fov_deg=fov,
                   K=data["K"], D=data["D"], **kwargs)

    def save(self, path: str):
        """Save current K and D to .npz for later reuse."""
        np.savez(path,
                 K=self.K, D=self.D,
                 width=self.width, height=self.height, fov_deg=self.fov_deg)
        print(f"[Camera] Calibration saved → {path}")


# ── Calibration utility ───────────────────────────────────────────────────────

def calibrate(image_dir: str,
              board_size: tuple = (9, 6),
              output: str = "calibration.npz",
              width: int  = _DEFAULT_W,
              height: int = _DEFAULT_H,
              fov_deg: float = _DEFAULT_FOV_DEG) -> FisheyeCamera:
    """
    Run fisheye calibration from a directory of checkerboard images.

    Requirements:
      - 15–30 images of a printed checkerboard (9×6 inner corners default)
      - Shoot from different angles/distances, covering the full FOV
      - Images must be the same resolution as the live camera

    Args:
        image_dir  : path to folder of .jpg / .png calibration images
        board_size : (cols, rows) of inner corners on the checkerboard
        output     : path to save calibration .npz
        width, height : expected image resolution
        fov_deg    : camera FOV (informational, stored in output file)

    Returns a ready-to-use FisheyeCamera with calibrated parameters.
    """
    import glob

    objp = np.zeros((1, board_size[0] * board_size[1], 3), np.float32)
    objp[0, :, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)

    obj_points = []   # 3D points in real world
    img_points = []   # 2D points in image plane

    patterns = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        patterns.extend(glob.glob(os.path.join(image_dir, ext)))

    if not patterns:
        raise RuntimeError(f"No images found in {image_dir}")

    print(f"[Calibration] Found {len(patterns)} images, board={board_size}")

    found = 0
    for path in sorted(patterns):
        img  = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(
            gray, board_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        if ret:
            obj_points.append(objp)
            cv2.cornerSubPix(gray, corners, (3, 3), (-1, -1),
                             (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                              30, 0.1))
            img_points.append(corners)
            found += 1
            print(f"  ✓ {os.path.basename(path)}")
        else:
            print(f"  ✗ {os.path.basename(path)} — corners not found")

    if found < 5:
        raise RuntimeError(
            f"Only {found} usable images — need at least 5. "
            "Use more images or check board_size."
        )

    print(f"[Calibration] Running fisheye calibration on {found} images...")

    K = np.zeros((3, 3))
    D = np.zeros((4, 1))
    flags = (cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
             + cv2.fisheye.CALIB_CHECK_COND
             + cv2.fisheye.CALIB_FIX_SKEW)

    rms, K, D, _, _ = cv2.fisheye.calibrate(
        obj_points, img_points, (width, height), K, D,
        flags=flags,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
    )

    print(f"[Calibration] Done — RMS reprojection error: {rms:.4f} px")
    print(f"  K =\n{K}")
    print(f"  D = {D.T}")

    cam = FisheyeCamera(width=width, height=height, fov_deg=fov_deg, K=K, D=D)
    cam.save(output)
    return cam


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Fisheye camera calibration tool")
    p.add_argument("--calibrate",  action="store_true",
                   help="Run calibration from checkerboard images")
    p.add_argument("--images",     type=str, default="calibration_images",
                   help="Directory of checkerboard images")
    p.add_argument("--board",      type=str, default="9x6",
                   help="Board inner corners as WxH  (default: 9x6)")
    p.add_argument("--output",     type=str, default="calibration.npz",
                   help="Output .npz path  (default: calibration.npz)")
    p.add_argument("--fov",        type=float, default=_DEFAULT_FOV_DEG,
                   help=f"Camera FOV in degrees (default: {_DEFAULT_FOV_DEG})")
    p.add_argument("--test",       type=str, default=None,
                   help="Undistort a single image and save result_undistorted.jpg")
    args = p.parse_args()

    if args.calibrate:
        bw, bh = map(int, args.board.split("x"))
        calibrate(args.images, board_size=(bw, bh),
                  output=args.output, fov_deg=args.fov)

    elif args.test:
        img = cv2.imread(args.test)
        if img is None:
            print(f"Cannot read {args.test}")
        else:
            h, w = img.shape[:2]
            cam = FisheyeCamera(width=w, height=h, fov_deg=args.fov)
            out = cam.undistort(img)
            side_by_side = np.hstack([img, out])
            out_path = "result_undistorted.jpg"
            cv2.imwrite(out_path, side_by_side)
            print(f"Saved side-by-side → {out_path}")

    else:
        p.print_help()
