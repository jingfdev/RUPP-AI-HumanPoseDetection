"""Asynchronous video pipeline: capture and inference in separate threads."""
from __future__ import annotations

import argparse
import os
import queue
import threading
import time
from typing import Tuple

import cv2
import numpy as np


def run_async(source: str, model_path: str, device: str, conf: float) -> None:
    from ultralytics import YOLO

    print(
        f"[INFO] Starting async pose demo | source={source} | model={model_path} | "
        f"device={device} | conf={conf}"
    )
    cap = cv2.VideoCapture(0 if source.isdigit() else source)
    if not cap.isOpened():
        raise RuntimeError("Failed to open video source.")

    model = YOLO(model_path)
    frame_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=1)
    result_queue: queue.Queue[Tuple[np.ndarray, float]] = queue.Queue(maxsize=1)
    stop_event = threading.Event()

    def capture_loop() -> None:
        while not stop_event.is_set():
            ok, frame = cap.read()
            if not ok:
                stop_event.set()
                break
            if frame_queue.full():
                try:
                    frame_queue.get_nowait()
                except queue.Empty:
                    pass
            frame_queue.put(frame)

    def infer_loop() -> None:
        while not stop_event.is_set():
            try:
                frame = frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            t0 = time.time()
            results = model.predict(frame, device=device, conf=conf, verbose=False)
            inf_ms = (time.time() - t0) * 1000.0

            annotated_frame = frame
            if results:
                # Draw connected skeletons instead of only keypoint dots.
                annotated_frame = results[0].plot(
                    boxes=False,
                    labels=False,
                    kpt_line=True,
                    kpt_radius=4,
                )

            if result_queue.full():
                try:
                    result_queue.get_nowait()
                except queue.Empty:
                    pass
            result_queue.put((annotated_frame, inf_ms))

    capture_thread = threading.Thread(target=capture_loop, daemon=True)
    infer_thread = threading.Thread(target=infer_loop, daemon=True)
    capture_thread.start()
    infer_thread.start()

    fps = 0.0
    last = time.time()
    frame_count = 0
    while not stop_event.is_set():
        try:
            frame, inf_ms = result_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(1e-6, now - last))
        last = now
        frame_count += 1

        if frame_count % 120 == 0:
            print(f"[PROGRESS] frames={frame_count} | fps={fps:.1f} | inference_ms={inf_ms:.1f}")

        cv2.putText(
            frame,
            f"Async {os.path.basename(model_path)} | FPS: {fps:.1f} | Inference: {inf_ms:.1f} ms",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("Async YOLOv8 Pipeline", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            stop_event.set()
            break

    stop_event.set()
    capture_thread.join(timeout=1.0)
    infer_thread.join(timeout=1.0)
    cap.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Demo finished | total_frames={frame_count}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Asynchronous video pipeline demo.")
    parser.add_argument("--source", default="0", help="Webcam index or video file path.")
    parser.add_argument("--model", default="yolov8m-pose.pt")
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda")
    parser.add_argument("--conf", type=float, default=0.35, help="Detection confidence threshold.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    selected_device = args.device
    if selected_device == "auto":
        try:
            import torch

            selected_device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            selected_device = "cpu"
    run_async(args.source, args.model, selected_device, args.conf)
