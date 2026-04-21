"""Multi-person pose detection demo for YOLOv8 or MediaPipe."""
from __future__ import annotations

import argparse
import time
from typing import Tuple

import cv2
import numpy as np


def _draw_keypoints(image: np.ndarray, xy: np.ndarray, color: Tuple[int, int, int]) -> None:
    for x, y in xy:
        cv2.circle(image, (int(x), int(y)), 3, color, -1)


def run_yolo(source: str, model_path: str, device: str) -> None:
    from ultralytics import YOLO

    cap = cv2.VideoCapture(0 if source.isdigit() else source)
    if not cap.isOpened():
        raise RuntimeError("Failed to open video source.")

    model = YOLO(model_path)

    last = time.time()
    fps = 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = model.predict(frame, device=device, verbose=False)
        if results:
            keypoints = results[0].keypoints
            if keypoints is not None:
                xy = keypoints.xy.cpu().numpy()
                for person in xy:
                    _draw_keypoints(frame, person, (0, 255, 0))

        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(1e-6, now - last))
        last = now

        cv2.putText(frame, f"YOLOv8 Multi-Person | FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("YOLOv8 Multi-Person", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


def run_mediapipe(source: str, model_path: str, max_poses: int) -> None:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    cap = cv2.VideoCapture(0 if source.isdigit() else source)
    if not cap.isOpened():
        raise RuntimeError("Failed to open video source.")

    base = python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base,
        output_segmentation_masks=False,
        num_poses=max_poses,
        running_mode=vision.RunningMode.VIDEO,
    )

    drawing = mp.solutions.drawing_utils
    connections = mp.solutions.pose.POSE_CONNECTIONS

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        last = time.time()
        fps = 0.0
        frame_index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_index += 1

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(frame_index * 1000 / max(1.0, cap.get(cv2.CAP_PROP_FPS)))
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            for pose in result.pose_landmarks:
                drawing.draw_landmarks(frame, pose, connections)

            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(1e-6, now - last))
            last = now

            cv2.putText(frame, f"MediaPipe Multi-Person | FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow("MediaPipe Multi-Person", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-person pose detection demo.")
    parser.add_argument("--backend", choices=["yolo", "mediapipe"], default="yolo")
    parser.add_argument("--source", default="0", help="Webcam index or video file path.")
    parser.add_argument("--yolo-model", default="yolov8m-pose.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mediapipe-model", default="pose_landmarker_full.task")
    parser.add_argument("--max-poses", type=int, default=4)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.backend == "yolo":
        run_yolo(args.source, args.yolo_model, args.device)
    else:
        run_mediapipe(args.source, args.mediapipe_model, args.max_poses)
