"""Benchmark YOLOv8 ONNX inference with ONNX Runtime (CPU/GPU)."""
from __future__ import annotations

import argparse
import csv
import time
from typing import List

import numpy as np


def benchmark(model_path: str, device: str, runs: int, warmup: int, batch: int) -> List[float]:
    import onnxruntime as ort

    providers = ["CPUExecutionProvider"]
    if device.lower() == "cuda":
        if "CUDAExecutionProvider" in ort.get_available_providers():
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            print("CUDAExecutionProvider not available. Falling back to CPU.")

    session = ort.InferenceSession(model_path, providers=providers)
    input_name = session.get_inputs()[0].name

    dummy = np.random.rand(batch, 3, 640, 640).astype(np.float32)

    for _ in range(warmup):
        _ = session.run(None, {input_name: dummy})

    times = []
    for _ in range(runs):
        t0 = time.time()
        _ = session.run(None, {input_name: dummy})
        times.append((time.time() - t0) * 1000.0)

    return times


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark ONNX pose model.")
    parser.add_argument("--model", default="yolov8m-pose.onnx")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--out-csv", default="")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    times = benchmark(args.model, args.device, args.runs, args.warmup, args.batch)

    mean_ms = float(np.mean(times))
    std_ms = float(np.std(times))
    fps = 1000.0 / mean_ms if mean_ms > 0 else 0.0

    print("ONNX Runtime Benchmark")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Batch: {args.batch}")
    print(f"Runs: {args.runs}")
    print(f"Mean: {mean_ms:.2f} ms")
    print(f"Std: {std_ms:.2f} ms")
    print(f"FPS: {fps:.2f}")

    if args.out_csv:
        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["model", "device", "batch", "runs", "mean_ms", "std_ms", "fps"])
            writer.writerow([args.model, args.device, args.batch, args.runs, f"{mean_ms:.4f}", f"{std_ms:.4f}", f"{fps:.4f}"])
        print(f"Saved: {args.out_csv}")
