"""Benchmark YOLOv8 PyTorch vs ONNX Runtime on identical inputs."""
from __future__ import annotations

import argparse
import os
import time
from typing import List, Tuple

import numpy as np


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _list_images(images_dir: str, max_images: int) -> List[str]:
    files = []
    for root, _dirs, names in os.walk(images_dir):
        for name in names:
            if os.path.splitext(name)[1].lower() in IMG_EXTS:
                files.append(os.path.join(root, name))
    files.sort()
    if max_images > 0:
        files = files[:max_images]
    return files


def _preprocess_image(image_bgr: np.ndarray, imgsz: int) -> np.ndarray:
    import cv2

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
    img = resized.astype(np.float32) / 255.0
    chw = np.transpose(img, (2, 0, 1))
    return chw


def _build_batches(
    images_dir: str,
    batch: int,
    imgsz: int,
    max_images: int,
) -> List[np.ndarray]:
    import cv2

    if not images_dir:
        dummy = np.random.rand(batch, 3, imgsz, imgsz).astype(np.float32)
        return [dummy]

    files = _list_images(images_dir, max_images)
    if not files:
        dummy = np.random.rand(batch, 3, imgsz, imgsz).astype(np.float32)
        return [dummy]

    tensors = []
    for path in files:
        img = cv2.imread(path)
        if img is None:
            continue
        tensors.append(_preprocess_image(img, imgsz))

    if not tensors:
        dummy = np.random.rand(batch, 3, imgsz, imgsz).astype(np.float32)
        return [dummy]

    batches = []
    idx = 0
    while idx < len(tensors):
        chunk = tensors[idx : idx + batch]
        if len(chunk) < batch:
            # Repeat last item to fill the batch for consistent timing
            chunk = chunk + [chunk[-1]] * (batch - len(chunk))
        batches.append(np.stack(chunk, axis=0))
        idx += batch

    return batches


def _benchmark_torch(
    model_path: str,
    device: str,
    inputs: List[np.ndarray],
    runs: int,
    warmup: int,
) -> List[float]:
    import torch
    from ultralytics import YOLO

    model = YOLO(model_path)
    torch_device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    model.model.to(torch_device)
    model.model.eval()

    def _sync() -> None:
        if torch_device.type == "cuda":
            torch.cuda.synchronize()

    with torch.no_grad():
        for i in range(warmup):
            batch = inputs[i % len(inputs)]
            tensor = torch.from_numpy(batch).to(torch_device)
            _ = model.model(tensor)
            _sync()

        times = []
        for i in range(runs):
            batch = inputs[i % len(inputs)]
            tensor = torch.from_numpy(batch).to(torch_device)
            t0 = time.perf_counter()
            _ = model.model(tensor)
            _sync()
            times.append((time.perf_counter() - t0) * 1000.0)

    return times


def _benchmark_onnx(
    model_path: str,
    device: str,
    inputs: List[np.ndarray],
    runs: int,
    warmup: int,
) -> List[float]:
    import onnxruntime as ort

    providers = ["CPUExecutionProvider"]
    if device.lower() == "cuda":
        if "CUDAExecutionProvider" in ort.get_available_providers():
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            print("CUDAExecutionProvider not available. Falling back to CPU.")

    session = ort.InferenceSession(model_path, providers=providers)
    input_name = session.get_inputs()[0].name

    for i in range(warmup):
        batch = inputs[i % len(inputs)]
        _ = session.run(None, {input_name: batch})

    times = []
    for i in range(runs):
        batch = inputs[i % len(inputs)]
        t0 = time.perf_counter()
        _ = session.run(None, {input_name: batch})
        times.append((time.perf_counter() - t0) * 1000.0)

    return times


def _summarize(times: List[float]) -> Tuple[float, float, float]:
    mean_ms = float(np.mean(times)) if times else 0.0
    std_ms = float(np.std(times)) if times else 0.0
    fps = 1000.0 / mean_ms if mean_ms > 0 else 0.0
    return mean_ms, std_ms, fps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare PyTorch vs ONNX Runtime speed.")
    parser.add_argument("--pt-model", default="yolov8m-pose.pt", help="YOLOv8 PyTorch model.")
    parser.add_argument("--onnx-model", default="yolov8m-pose.onnx", help="YOLOv8 ONNX model.")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--images-dir", default="", help="Optional directory of images to use.")
    parser.add_argument("--max-images", type=int, default=0, help="Limit images (0 = all).")
    parser.add_argument("--out-csv", default="", help="Optional CSV output path.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    inputs = _build_batches(args.images_dir, args.batch, args.imgsz, args.max_images)

    torch_times = _benchmark_torch(
        model_path=args.pt_model,
        device=args.device,
        inputs=inputs,
        runs=args.runs,
        warmup=args.warmup,
    )
    onnx_times = _benchmark_onnx(
        model_path=args.onnx_model,
        device=args.device,
        inputs=inputs,
        runs=args.runs,
        warmup=args.warmup,
    )

    torch_mean, torch_std, torch_fps = _summarize(torch_times)
    onnx_mean, onnx_std, onnx_fps = _summarize(onnx_times)

    print("PyTorch vs ONNX Runtime Benchmark")
    print("=" * 70)
    print(f"Device: {args.device}")
    print(f"Batch: {args.batch}")
    print(f"Runs: {args.runs}")
    print(f"PyTorch model: {args.pt_model}")
    print(f"ONNX model: {args.onnx_model}")
    if args.images_dir:
        print(f"Images: {args.images_dir}")
    print("-" * 70)
    print(f"PyTorch mean: {torch_mean:.2f} ms | std: {torch_std:.2f} ms | FPS: {torch_fps:.2f}")
    print(f"ONNX mean:    {onnx_mean:.2f} ms | std: {onnx_std:.2f} ms | FPS: {onnx_fps:.2f}")

    if args.out_csv:
        import csv

        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "device",
                    "batch",
                    "runs",
                    "pt_model",
                    "onnx_model",
                    "pt_mean_ms",
                    "pt_std_ms",
                    "pt_fps",
                    "onnx_mean_ms",
                    "onnx_std_ms",
                    "onnx_fps",
                ]
            )
            writer.writerow(
                [
                    args.device,
                    args.batch,
                    args.runs,
                    args.pt_model,
                    args.onnx_model,
                    f"{torch_mean:.4f}",
                    f"{torch_std:.4f}",
                    f"{torch_fps:.4f}",
                    f"{onnx_mean:.4f}",
                    f"{onnx_std:.4f}",
                    f"{onnx_fps:.4f}",
                ]
            )
        print(f"Saved: {args.out_csv}")
