"""
OpenPose Wrapper Module
-----------------------
Clean Python wrapper for OpenPose inference using pyopenpose API.
Configures OpenPose for body and face detection without hands.

Academic Project - No Training Required
Uses pre-trained OpenPose models for real-time pose and face landmark detection.

Author: RUPP AI Final Year Project
Date: 2026
"""

import cv2
import numpy as np
import sys
import os


# OpenPose BODY_25 keypoint names (index -> name)
BODY_25_NAMES = [
    "nose",            # 0
    "neck",            # 1
    "right_shoulder",  # 2
    "right_elbow",     # 3
    "right_wrist",     # 4
    "left_shoulder",   # 5
    "left_elbow",      # 6
    "left_wrist",      # 7
    "mid_hip",         # 8
    "right_hip",       # 9
    "right_knee",      # 10
    "right_ankle",     # 11
    "left_hip",        # 12
    "left_knee",       # 13
    "left_ankle",      # 14
    "right_eye",       # 15
    "left_eye",        # 16
    "right_ear",       # 17
    "left_ear",        # 18
    "left_big_toe",    # 19
    "left_small_toe",  # 20
    "left_heel",       # 21
    "right_big_toe",   # 22
    "right_small_toe", # 23
    "right_heel",      # 24
]


# Common BODY_25 skeleton edges (start_idx, end_idx)
BODY_25_EDGES = [
    # Head
    (0, 1),
    (0, 15), (15, 17),
    (0, 16), (16, 18),
    # Torso
    (1, 2), (2, 3), (3, 4),
    (1, 5), (5, 6), (6, 7),
    (1, 8),
    (8, 9), (9, 10), (10, 11),
    (8, 12), (12, 13), (13, 14),
    # Feet
    (14, 19), (19, 20),
    (14, 21),
    (11, 22), (22, 23),
    (11, 24),
]


def _draw_text_with_outline(img, text, org, font_scale, color, thickness=1):
    """Draw readable text with a thin black outline."""
    cv2.putText(
        img,
        text,
        org,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (0, 0, 0),
        thickness + 2,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        text,
        org,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def _draw_keypoint_legend(img, names, title="Keypoints", x=10, y=10, max_rows=13):
    """Draw a compact keypoint index->name legend (top-left panel)."""
    if not names:
        return

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    line_h = 18
    pad = 10

    rows = min(max_rows, max(1, len(names)))
    cols = int(np.ceil(len(names) / rows))
    col_w = 190
    panel_w = pad * 2 + col_w * cols
    panel_h = pad * 2 + line_h * (rows + 1)

    x2 = min(img.shape[1] - 1, x + panel_w)
    y2 = min(img.shape[0] - 1, y + panel_h)

    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x2, y2), (0, 0, 0), -1)
    img[:] = cv2.addWeighted(overlay, 0.55, img, 0.45, 0)
    cv2.rectangle(img, (x, y), (x2, y2), (255, 255, 255), 1)

    _draw_text_with_outline(img, title, (x + pad, y + pad + 12), 0.55, (255, 255, 255), 1)

    start_y = y + pad + line_h + 6
    for idx, name in enumerate(names):
        col = idx // rows
        row = idx % rows
        tx = x + pad + col * col_w
        ty = start_y + row * line_h
        _draw_text_with_outline(img, f"{idx:>2}. {name}", (tx, ty), font_scale, (230, 230, 230), 1)

def _candidate_openpose_roots():
    """Return likely OpenPose install roots on Windows."""
    candidates = []

    env_root = os.environ.get("OPENPOSE_HOME")
    if env_root:
        candidates.append(env_root)

    candidates += [
        "C:/openpose",
        "C:/OpenPose",
        "C:/Program Files/openpose",
        "C:/Program Files/OpenPose",
        "../openpose",
        "./openpose",
    ]
    return candidates


def _add_windows_dll_directories(dll_dirs):
    """Best-effort DLL search path setup (Windows only)."""
    if os.name != "nt":
        return

    for dll_dir in dll_dirs:
        if not dll_dir or not os.path.isdir(dll_dir):
            continue
        try:
            # Python 3.8+: preferred way
            os.add_dll_directory(dll_dir)
        except Exception:
            # Fallback for older behavior
            os.environ["PATH"] = dll_dir + os.pathsep + os.environ.get("PATH", "")


def _try_configure_openpose_paths():
    """
    Try to make `import pyopenpose` work by adding common OpenPose build paths.

    OpenPose's Python bindings are NOT installable via pip.
    They are generated when building OpenPose with `BUILD_PYTHON=ON`.
    """
    python_dirs = []
    dll_dirs = []

    # Allow explicit overrides
    extra_python = os.environ.get("OPENPOSE_PYTHON_PATH")
    if extra_python:
        python_dirs.append(extra_python)

    extra_bin = os.environ.get("OPENPOSE_BIN_PATH")
    if extra_bin:
        dll_dirs.append(extra_bin)

    for root in _candidate_openpose_roots():
        if not root:
            continue
        root = root.replace("\\", "/")
        if not os.path.isdir(root):
            continue

        # Common OpenPose Python binding folders (depending on build/toolchain)
        python_dirs += [
            # Some builds expose a Python package `openpose` under build/python
            f"{root}/build/python",
            f"{root}/python",
            f"{root}/build/python/openpose/Release",
            f"{root}/build/python/openpose",
            f"{root}/python/openpose",
        ]

        # Common DLL folders
        dll_dirs += [
            f"{root}/build/x64/Release",
            f"{root}/build/bin",
            f"{root}/bin",
        ]

    # DLL search paths first (Windows)
    _add_windows_dll_directories(dll_dirs)

    # Python module search path
    for p in python_dirs:
        if p and os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)


def import_pyopenpose():
    """Import pyopenpose with helpful error messages and path auto-configuration."""
    try:
        import pyopenpose as op  # type: ignore
        return op
    except Exception:
        # Attempt to configure paths and retry
        _try_configure_openpose_paths()
        try:
            import pyopenpose as op  # type: ignore
            return op
        except Exception:
            # Some OpenPose builds require importing via the `openpose` package
            try:
                from openpose import pyopenpose as op  # type: ignore
                return op
            except Exception:
                _try_configure_openpose_paths()
                try:
                    from openpose import pyopenpose as op  # type: ignore
                    return op
                except Exception as e:
                    raise ImportError(
                        "pyopenpose could not be imported. OpenPose is not installable via pip; "
                        "you must build OpenPose with Python support (BUILD_PYTHON=ON). "
                        "On Windows, ensure OpenPose DLL folders are discoverable (set OPENPOSE_HOME, "
                        "or add OpenPose build/bin folders to PATH). "
                        "See the OpenPose notes in Three_Model_Comparison.ipynb for setup guidance. "
                        f"Details: {e}"
                    )


class OpenPoseWrapper:
    """
    Wrapper for OpenPose inference with body and face detection.
    Simplifies configuration and provides a clean interface for academic use.
    """
    
    def __init__(self, model_folder="openpose/models/", display_output=False):
        """
        Initialize OpenPose with body and face detection enabled.
        
        Args:
            model_folder (str): Path to OpenPose models directory
            display_output (bool): Whether to use OpenPose's built-in visualization
        """
        self.model_folder = model_folder
        self.display_output = display_output

        # Import OpenPose binding lazily so importing this module doesn't crash
        # projects that don't have OpenPose installed.
        self.op = import_pyopenpose()
        
        # Validate model folder
        if not os.path.exists(model_folder):
            print(f"WARNING: Model folder not found at {model_folder}")
            print("Trying default OpenPose installation paths...")

            env_root = os.environ.get("OPENPOSE_HOME")
            if env_root:
                env_models = os.path.join(env_root, "models")
                if os.path.exists(env_models):
                    self.model_folder = env_models.replace("\\", "/") + "/"
                    print(f"Found models via OPENPOSE_HOME: {self.model_folder}")
            
            # Common OpenPose installation paths
            possible_paths = [
                "C:/openpose/models/",
                "C:/Program Files/openpose/models/",
                "../openpose/models/",
                "./models/",
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    self.model_folder = path
                    print(f"Found models at: {path}")
                    break
            else:
                print("ERROR: Could not find OpenPose models directory!")
                print(f"Please set model_folder parameter or place models at: {model_folder}")
        
        # Configure OpenPose parameters
        self.params = self._configure_params()
        
        # Initialize OpenPose Wrapper
        print("Initializing OpenPose...")
        try:
            self.opWrapper = self.op.WrapperPython()
            self.opWrapper.configure(self.params)
            self.opWrapper.start()
            print("✓ OpenPose initialized successfully!")
        except Exception as e:
            print(f"ERROR initializing OpenPose: {e}")
            raise
        
        # Store last detection results
        self.last_body_keypoints = None
        self.last_face_keypoints = None
        self.body_landmark_count = 0
        self.face_landmark_count = 0
        
    def _configure_params(self):
        """
        Configure OpenPose parameters for academic evaluation.
        
        Returns:
            dict: OpenPose configuration parameters
        """
        params = {
            # Model paths
            "model_folder": self.model_folder,
            
            # Body detection
            "body": 1,  # Enable body detection (BODY_25 model)
            "model_pose": "BODY_25",  # 25 body keypoints
            
            # Face detection
            "face": True,  # Enable face landmark detection (70 points)
            
            # Hand detection - DISABLED for this comparison
            "hand": False,
            
            # Display settings
            "disable_blending": False,  # Blend skeleton with original image
            
            # Performance settings
            "num_gpu": 1,  # Use GPU if available (fallback to CPU automatically)
            "num_gpu_start": 0,  # Start from GPU 0
            
            # Output settings - disable built-in display (we'll use OpenCV)
            "display": 0 if not self.display_output else 1,
        }
        
        return params
    
    def process_frame(self, frame):
        """
        Process a single frame through OpenPose.
        
        Args:
            frame (np.ndarray): Input BGR image from OpenCV
            
        Returns:
            tuple: (output_frame, body_keypoints, face_keypoints)
                - output_frame: Rendered frame with skeleton overlay
                - body_keypoints: NumPy array of body keypoints [N x 25 x 3]
                - face_keypoints: NumPy array of face keypoints [N x 70 x 3]
        """
        # Create OpenPose datum
        datum = self.op.Datum()
        datum.cvInputData = frame
        
        # Process frame
        self.opWrapper.emplaceAndPop(self.op.VectorDatum([datum]))
        
        # Extract results
        output_frame = datum.cvOutputData
        body_keypoints = datum.poseKeypoints
        face_keypoints = datum.faceKeypoints
        
        # Store for later access
        self.last_body_keypoints = body_keypoints
        self.last_face_keypoints = face_keypoints
        
        # Count landmarks
        self.body_landmark_count = 0
        self.face_landmark_count = 0
        
        if body_keypoints is not None and len(body_keypoints.shape) >= 2:
            # body_keypoints shape: [num_people, num_keypoints, 3]
            self.body_landmark_count = int(np.sum(body_keypoints[:, :, 2] > 0))
        
        if face_keypoints is not None and len(face_keypoints.shape) >= 2:
            # face_keypoints shape: [num_people, 70, 3]
            self.face_landmark_count = int(np.sum(face_keypoints[:, :, 2] > 0))
        
        return output_frame, body_keypoints, face_keypoints
    
    def get_landmark_count(self):
        """
        Get total detected landmark count from last processed frame.
        
        Returns:
            int: Total number of detected landmarks (body + face)
        """
        return self.body_landmark_count + self.face_landmark_count
    
    def draw_custom_skeleton(
        self,
        frame,
        body_keypoints,
        face_keypoints,
        body_color=(0, 255, 0),
        face_color=(255, 255, 0),
        *,
        draw_connections=True,
        draw_points=True,
        draw_indices=False,
        draw_names=False,
        show_legend=False,
        min_conf=0.1,
        line_thickness=2,
        point_radius=5,
        face_radius=2,
    ):
        """
        Draw custom skeleton visualization on frame.
        Useful for consistent visualization across models.
        
        Args:
            frame (np.ndarray): Input image
            body_keypoints: Body keypoints array
            face_keypoints: Face keypoints array
            body_color: BGR color for body points
            face_color: BGR color for face points
            
        Returns:
            np.ndarray: Frame with skeleton drawn
        """
        output = frame.copy()

        if show_legend:
            _draw_keypoint_legend(output, BODY_25_NAMES, title="OpenPose BODY_25")

        # Draw BODY_25
        if body_keypoints is not None and len(body_keypoints.shape) >= 2:
            for person in body_keypoints:
                if person is None or len(person) == 0:
                    continue

                # Connections first
                if draw_connections:
                    for s, e in BODY_25_EDGES:
                        if s >= len(person) or e >= len(person):
                            continue
                        x1, y1, c1 = person[s]
                        x2, y2, c2 = person[e]
                        if c1 <= min_conf or c2 <= min_conf:
                            continue
                        pt1 = (int(x1), int(y1))
                        pt2 = (int(x2), int(y2))
                        cv2.line(output, pt1, pt2, (255, 0, 0), line_thickness, cv2.LINE_AA)

                # Points + optional labels
                if draw_points:
                    for i, kp in enumerate(person):
                        x, y, conf = kp
                        if conf <= min_conf:
                            continue
                        cx, cy = int(x), int(y)
                        cv2.circle(output, (cx, cy), point_radius, body_color, -1, cv2.LINE_AA)

                        if draw_indices or draw_names:
                            label = str(i)
                            if draw_names and i < len(BODY_25_NAMES):
                                label = f"{i}:{BODY_25_NAMES[i]}"
                            _draw_text_with_outline(output, label, (cx + 6, cy - 6), 0.4, (255, 255, 255), 1)
        
        # Draw face keypoints (points only; OpenPose face is dense)
        if face_keypoints is not None and len(face_keypoints.shape) >= 2 and draw_points:
            for person in face_keypoints:
                for kp in person:
                    x, y, conf = kp
                    if conf <= min_conf:
                        continue
                    cv2.circle(output, (int(x), int(y)), face_radius, face_color, -1, cv2.LINE_AA)
        
        return output
    
    def cleanup(self):
        """
        Clean up OpenPose resources.
        Call this when shutting down the application.
        """
        if hasattr(self, 'opWrapper'):
            self.opWrapper.stop()
            print("✓ OpenPose resources released")


def test_openpose():
    """
    Simple test function to verify OpenPose installation.
    Captures webcam feed and displays OpenPose output.
    """
    print("=" * 60)
    print("OpenPose Wrapper Test")
    print("=" * 60)
    
    # Initialize OpenPose
    try:
        pose_detector = OpenPoseWrapper()
    except Exception as e:
        print(f"Failed to initialize OpenPose: {e}")
        return
    
    # Open webcam
    cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    if not cap.isOpened():
        print("ERROR: Cannot open webcam")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("\nControls:")
    print("  [Q] - Quit")
    print("  [S] - Save screenshot")
    print("  [L] - Toggle legend")
    print("  [I] - Toggle indices")
    print("  [N] - Toggle names")
    print("\nProcessing...")
    
    import time
    prev_time = time.time()
    
    show_legend = False
    show_indices = False
    show_names = False

    while True:
        success, frame = cap.read()
        if not success:
            continue
        
        # Process with OpenPose
        _output_frame, body_kp, face_kp = pose_detector.process_frame(frame)

        # Use improved custom rendering (closer to reference diagram style)
        output_frame = pose_detector.draw_custom_skeleton(
            frame,
            body_kp,
            face_kp,
            draw_connections=True,
            draw_points=True,
            draw_indices=show_indices,
            draw_names=show_names,
            show_legend=show_legend,
        )
        
        # Calculate FPS
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time) if prev_time > 0 else 0
        prev_time = curr_time
        
        # Display info
        if output_frame is not None:
            landmark_count = pose_detector.get_landmark_count()
            
            # Add FPS and info overlay
            cv2.rectangle(output_frame, (10, 10), (300, 100), (0, 0, 0), -1)
            cv2.putText(output_frame, "OpenPose Test", (20, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(output_frame, f"FPS: {int(fps)}", (20, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(output_frame, f"Landmarks: {landmark_count}", (20, 85), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow("OpenPose Test", output_frame)
        
        # Handle key presses
        key = cv2.waitKey(5) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"openpose_test_{timestamp}.png"
            cv2.imwrite(filename, output_frame)
            print(f"Screenshot saved: {filename}")
        elif key == ord('l'):
            show_legend = not show_legend
        elif key == ord('i'):
            show_indices = not show_indices
            if show_indices:
                show_names = False
        elif key == ord('n'):
            show_names = not show_names
            if show_names:
                show_indices = False
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    pose_detector.cleanup()
    print("\n✓ Test completed successfully!")


if __name__ == "__main__":
    test_openpose()
