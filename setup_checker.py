"""Setup checker for the Three-Model comparison project.

- Verifies Python version, packages, model files, webcam access
- Checks whether OpenPose bindings are importable via openpose_wrapper.import_pyopenpose()

Run:
  python setup_checker.py
"""

from __future__ import annotations

import os
import sys


def _check_import(name: str) -> tuple[bool, str]:
    try:
        __import__(name)
        return True, "✓ Installed"
    except Exception:
        return False, "✗ Not found"


def _check_file(path: str) -> tuple[bool, str]:
    if os.path.exists(path):
        size_mb = os.path.getsize(path) / (1024 * 1024)
        return True, f"✓ Found ({size_mb:.1f} MB)"
    return False, "✗ Missing"


def main() -> int:
    print("=" * 70)
    print("Three-Model Pose Comparison - Setup Verification")
    print("=" * 70)

    print(f"\nPython Version: {sys.version}")

    print("\n" + "=" * 70)
    print("Checking Required Packages...")
    print("=" * 70)

    required = {
        "opencv-python": "cv2",
        "mediapipe": "mediapipe",
        "numpy": "numpy",
    }

    all_required = True
    for pkg, mod in required.items():
        ok, msg = _check_import(mod)
        print(f"  {pkg:20} {msg}")
        all_required &= ok

    if all_required:
        print("\n✓ All required packages installed!")
    else:
        print("\n⚠ Missing required packages!")
        print("  Run: pip install -r requirements_python.txt")

    print("\n" + "=" * 70)
    print("Checking Optional Packages...")
    print("=" * 70)

    openpose_ok = False
    openpose_msg = "✗ Not available"
    try:
        import openpose_wrapper as ow
        try:
            ow.import_pyopenpose()
            openpose_ok = True
            openpose_msg = "✓ Available"
        except Exception as e:
            openpose_msg = f"✗ Not available ({type(e).__name__})"
    except Exception:
        openpose_msg = "✗ Wrapper import failed"

    print(f"  {'OpenPose (pyopenpose)':20} {openpose_msg}")
    if not openpose_ok:
        print("\n  ℹ OpenPose is optional. Models 1 & 2 (MediaPipe) will work without it.")
        print("  To enable Model 3 (OpenPose), build OpenPose with Python bindings.")
        print(f"  OPENPOSE_HOME={os.environ.get('OPENPOSE_HOME')}")
        print(f"  OPENPOSE_PYTHON_PATH={os.environ.get('OPENPOSE_PYTHON_PATH')}")
        print(f"  OPENPOSE_BIN_PATH={os.environ.get('OPENPOSE_BIN_PATH')}")

    print("\n" + "=" * 70)
    print("Checking MediaPipe Model Files...")
    print("=" * 70)

    models = {
        "Pose Model": "pose_landmarker_full.task",
        "Face Model": "face_landmarker.task",
    }

    all_models = True
    for name, path in models.items():
        ok, msg = _check_file(path)
        print(f"  {name:20} {msg}")
        all_models &= ok

    if all_models:
        print("\n✓ All model files present!")
    else:
        print("\n⚠ Missing model files!")

    print("\n" + "=" * 70)
    print("Checking Webcam...")
    print("=" * 70)

    try:
        import cv2

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("  ✗ Cannot open webcam")
            print("  Try: cv2.VideoCapture(1) or cv2.VideoCapture(2)")
        else:
            ok, frame = cap.read()
            if ok and frame is not None:
                h, w = frame.shape[:2]
                print(f"  ✓ Webcam detected ({w}x{h})")
            else:
                print("  ⚠ Webcam opens but cannot read frames")
        cap.release()
    except Exception as e:
        print(f"  ✗ Error: {e}")

    print("\n" + "=" * 70)
    print("Setup Summary")
    print("=" * 70)

    if all_required and all_models:
        print("\n🎉 Your system is ready!")
        if not openpose_ok:
            print("\n  Note: OpenPose (Model 3) not available.")
            print("        Models 1 & 2 (MediaPipe) are ready to use.")
    else:
        print("\n⚠ Setup incomplete. Fix issues above.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
