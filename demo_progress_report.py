"""Generate a short, teacher-friendly project progress report.

This script runs YOLOv8 pose inference on a small number of frames from a video
and prints simple metrics you can present during a demo.
"""
from __future__ import annotations

import argparse
import time

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a short pose progress report.")
    parser.add_argument("--source", default="sample_video.mp4", help="Video path or webcam index.")
    parser.add_argument("--model", default="yolo11s-pose.pt", help="YOLOv8 pose model path.")
    parser.add_argument("--device", default="cuda:0", help="CUDA device only, for example cuda or cuda:0")
    parser.add_argument("--frames", type=int, default=30, help="How many frames to evaluate.")
    return parser.parse_args()


def _require_cuda_device(requested: str) -> str:
    device = requested.strip().lower()
    if device in {"auto", "cpu"}:
        raise ValueError(
            f"GPU-only mode is enabled. Unsupported device '{requested}'. "
            "Use a CUDA device such as 'cuda' or 'cuda:0'."
        )
    if not device.startswith("cuda"):
        raise ValueError(
            f"Unsupported device '{requested}'. Only CUDA devices are allowed in GPU-only mode."
        )

    try:
        import torch
    except Exception as exc:
        raise RuntimeError(
            "PyTorch with CUDA support is required, but torch could not be imported."
        ) from exc

    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError(
            "GPU-only mode is enabled, but no CUDA-capable GPU is available to PyTorch."
        )

    return "cuda:0" if device == "cuda" else device


def main() -> int:
    args = parse_args()
    device = _require_cuda_device(args.device)

    from ultralytics import YOLO

    cap = cv2.VideoCapture(0 if args.source.isdigit() else args.source)
    if not cap.isOpened():
        print("[ERROR] Cannot open source:", args.source)
        return 1

    model = YOLO(args.model)

    processed = 0
    total_detections = 0
    total_people_with_keypoints = 0

    t0 = time.time()
    while processed < args.frames:
        ok, frame = cap.read()
        if not ok:
            break

        result = model.predict(frame, device=device, verbose=False)[0]

        detections = len(result.boxes) if result.boxes is not None else 0
        people_kp = int(result.keypoints.xy.shape[0]) if result.keypoints is not None else 0

        total_detections += detections
        total_people_with_keypoints += people_kp
        processed += 1

    elapsed_s = max(1e-6, time.time() - t0)
    cap.release()

    print("=" * 70)
    print("PROJECT PROGRESS REPORT")
    print("=" * 70)
    print(f"Source: {args.source}")
    print(f"Model: {args.model}")
    print(f"Device used: {device}")
    print(f"Frames requested: {args.frames}")
    print(f"Frames processed: {processed}")
    print("-" * 70)
    print(f"Total detections: {total_detections}")
    print(f"Total people with keypoints: {total_people_with_keypoints}")
    print(f"Average detections/frame: {total_detections / max(1, processed):.3f}")
    print(f"Average people with keypoints/frame: {total_people_with_keypoints / max(1, processed):.3f}")
    print(f"Elapsed time (s): {elapsed_s:.2f}")
    print(f"Effective FPS: {processed / elapsed_s:.2f}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
