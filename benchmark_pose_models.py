"""Benchmark YOLOv8, YOLO11, and YOLO26 pose models on the local GPU.

The script measures local inference speed on this project's video/webcam input
and combines it with official Ultralytics COCO pose metrics for report graphs.
"""
from __future__ import annotations

import argparse
import csv
import statistics
import time
from pathlib import Path
from typing import Dict, List

import cv2
import matplotlib.pyplot as plt
import numpy as np

from async_video_pipeline import require_cuda_device


MODEL_METRICS: Dict[str, Dict[str, float | str]] = {
    "yolov8n-pose.pt": {"family": "YOLOv8", "scale": "n", "map50_95": 50.4, "map50": 80.1, "official_speed_ms": 1.18, "speed_hardware": "A100 TensorRT", "params_m": 3.3, "flops_b": 9.2},
    "yolov8s-pose.pt": {"family": "YOLOv8", "scale": "s", "map50_95": 60.0, "map50": 86.2, "official_speed_ms": 1.42, "speed_hardware": "A100 TensorRT", "params_m": 11.6, "flops_b": 30.2},
    "yolov8m-pose.pt": {"family": "YOLOv8", "scale": "m", "map50_95": 65.0, "map50": 88.8, "official_speed_ms": 2.00, "speed_hardware": "A100 TensorRT", "params_m": 26.4, "flops_b": 81.0},
    "yolov8l-pose.pt": {"family": "YOLOv8", "scale": "l", "map50_95": 67.6, "map50": 90.0, "official_speed_ms": 2.59, "speed_hardware": "A100 TensorRT", "params_m": 44.4, "flops_b": 168.6},
    "yolov8x-pose.pt": {"family": "YOLOv8", "scale": "x", "map50_95": 69.2, "map50": 90.2, "official_speed_ms": 3.73, "speed_hardware": "A100 TensorRT", "params_m": 69.4, "flops_b": 263.2},
    "yolo11n-pose.pt": {"family": "YOLO11", "scale": "n", "map50_95": 50.0, "map50": 81.0, "official_speed_ms": 1.7, "speed_hardware": "T4 TensorRT10", "params_m": 2.9, "flops_b": 7.6},
    "yolo11s-pose.pt": {"family": "YOLO11", "scale": "s", "map50_95": 58.9, "map50": 86.3, "official_speed_ms": 2.6, "speed_hardware": "T4 TensorRT10", "params_m": 9.9, "flops_b": 23.2},
    "yolo11m-pose.pt": {"family": "YOLO11", "scale": "m", "map50_95": 64.9, "map50": 89.4, "official_speed_ms": 4.9, "speed_hardware": "T4 TensorRT10", "params_m": 20.9, "flops_b": 71.7},
    "yolo11l-pose.pt": {"family": "YOLO11", "scale": "l", "map50_95": 66.1, "map50": 89.9, "official_speed_ms": 6.4, "speed_hardware": "T4 TensorRT10", "params_m": 26.2, "flops_b": 90.7},
    "yolo11x-pose.pt": {"family": "YOLO11", "scale": "x", "map50_95": 69.5, "map50": 91.1, "official_speed_ms": 12.1, "speed_hardware": "T4 TensorRT10", "params_m": 58.8, "flops_b": 203.3},
    "yolo26n-pose.pt": {"family": "YOLO26", "scale": "n", "map50_95": 57.2, "map50": 83.3, "official_speed_ms": 1.8, "speed_hardware": "T4 TensorRT10", "params_m": 2.9, "flops_b": 7.5},
    "yolo26s-pose.pt": {"family": "YOLO26", "scale": "s", "map50_95": 63.0, "map50": 86.6, "official_speed_ms": 2.7, "speed_hardware": "T4 TensorRT10", "params_m": 10.4, "flops_b": 23.9},
    "yolo26m-pose.pt": {"family": "YOLO26", "scale": "m", "map50_95": 68.8, "map50": 89.6, "official_speed_ms": 5.0, "speed_hardware": "T4 TensorRT10", "params_m": 21.5, "flops_b": 73.1},
    "yolo26l-pose.pt": {"family": "YOLO26", "scale": "l", "map50_95": 70.4, "map50": 90.5, "official_speed_ms": 6.5, "speed_hardware": "T4 TensorRT10", "params_m": 25.9, "flops_b": 91.3},
    "yolo26x-pose.pt": {"family": "YOLO26", "scale": "x", "map50_95": 71.6, "map50": 91.6, "official_speed_ms": 12.2, "speed_hardware": "T4 TensorRT10", "params_m": 57.6, "flops_b": 201.7},
}

SOURCE_URLS = {
    "YOLOv8": "https://huggingface.co/Ultralytics/YOLOv8",
    "YOLO11": "https://huggingface.co/Ultralytics/YOLO11",
    "YOLO26": "https://docs.ultralytics.com/tasks/pose/",
    "YOLO12": "https://docs.ultralytics.com/models/yolo12/",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark official Ultralytics pose models on CUDA.")
    parser.add_argument("--source", default="sample_video.mp4", help="Video file used for local speed benchmarking.")
    parser.add_argument("--device", default="cuda:0", help="CUDA device only, for example cuda or cuda:0.")
    parser.add_argument("--conf", type=float, default=0.35, help="Inference confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--frames", type=int, default=90, help="Maximum sampled frames per model.")
    parser.add_argument("--sample-step", type=int, default=3, help="Analyze every Nth frame from source video.")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup predictions before timing each model.")
    parser.add_argument("--output-dir", default="benchmark_results", help="Directory for CSV, plots, and report.")
    parser.add_argument(
        "--benchmark-mode",
        choices=["sampled-video", "realtime"],
        default="sampled-video",
        help="sampled-video reuses the same frames for every model; realtime reads frames live for each model.",
    )
    parser.add_argument(
        "--show-window",
        action="store_true",
        help="Show a live annotated preview window during realtime benchmarking.",
    )
    parser.add_argument(
        "--window-scale",
        type=float,
        default=1.25,
        help="Scale factor for the realtime benchmark preview window.",
    )
    return parser.parse_args()


def load_sampled_frames(source: str, frames: int, sample_step: int) -> List[np.ndarray]:
    cap = cv2.VideoCapture(0 if source.isdigit() else source)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open benchmark source: {source}")

    sampled: List[np.ndarray] = []
    seen = 0
    while len(sampled) < frames:
        ok, frame = cap.read()
        if not ok:
            break
        seen += 1
        if seen % max(1, sample_step) == 0:
            sampled.append(frame)

    cap.release()
    if not sampled:
        raise RuntimeError("No frames were sampled from the benchmark source.")
    return sampled


def cuda_synchronize() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def benchmark_model(model_name: str, frames: List[np.ndarray], args: argparse.Namespace, device: str) -> Dict[str, float | str]:
    from ultralytics import YOLO

    model = YOLO(model_name)
    for _ in range(args.warmup):
        model.predict(frames[0], device=device, conf=args.conf, imgsz=args.imgsz, verbose=False)
    cuda_synchronize()

    timings_ms: List[float] = []
    detections = 0
    people_with_keypoints = 0

    for frame in frames:
        start = time.perf_counter()
        result = model.predict(frame, device=device, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
        cuda_synchronize()
        timings_ms.append((time.perf_counter() - start) * 1000.0)

        detections += len(result.boxes) if result.boxes is not None else 0
        people_with_keypoints += int(result.keypoints.xy.shape[0]) if result.keypoints is not None else 0

    mean_ms = statistics.mean(timings_ms)
    median_ms = statistics.median(timings_ms)
    fps = 1000.0 / mean_ms if mean_ms > 0 else 0.0
    official = MODEL_METRICS[model_name]

    return {
        "model": model_name,
        "family": official["family"],
        "scale": official["scale"],
        "status": "ok",
        "frames": len(frames),
        "local_mean_ms": round(mean_ms, 3),
        "local_median_ms": round(median_ms, 3),
        "local_fps": round(fps, 3),
        "end_to_end_fps": "",
        "avg_detections_per_frame": round(detections / len(frames), 3),
        "avg_people_keypoints_per_frame": round(people_with_keypoints / len(frames), 3),
        "official_pose_map50_95": official["map50_95"],
        "official_pose_map50": official["map50"],
        "official_speed_ms": official["official_speed_ms"],
        "official_speed_hardware": official["speed_hardware"],
        "params_m": official["params_m"],
        "flops_b": official["flops_b"],
        "source_url": SOURCE_URLS[str(official["family"])],
    }


def benchmark_model_realtime(model_name: str, args: argparse.Namespace, device: str) -> Dict[str, float | str]:
    from ultralytics import YOLO

    cap = cv2.VideoCapture(0 if args.source.isdigit() else args.source)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open realtime benchmark source: {args.source}")

    model = YOLO(model_name)

    warmup_done = 0
    while warmup_done < args.warmup:
        ok, frame = cap.read()
        if not ok:
            cap.release()
            raise RuntimeError("Realtime source ended during warmup.")
        model.predict(frame, device=device, conf=args.conf, imgsz=args.imgsz, verbose=False)
        warmup_done += 1
    cuda_synchronize()

    timings_ms: List[float] = []
    detections = 0
    people_with_keypoints = 0
    wall_start = time.perf_counter()
    window_name = "Pose Benchmark Preview"
    window_ready = False

    if args.show_window:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while len(timings_ms) < args.frames:
        ok, frame = cap.read()
        if not ok:
            break

        start = time.perf_counter()
        result = model.predict(frame, device=device, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
        cuda_synchronize()
        timings_ms.append((time.perf_counter() - start) * 1000.0)

        detections += len(result.boxes) if result.boxes is not None else 0
        people_with_keypoints += int(result.keypoints.xy.shape[0]) if result.keypoints is not None else 0

        if args.show_window:
            preview = result.plot(boxes=True, labels=False, kpt_line=True, kpt_radius=4)
            cv2.putText(
                preview,
                f"{model_name} | frame {len(timings_ms)}/{args.frames} | {timings_ms[-1]:.1f} ms",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            if not window_ready:
                height, width = preview.shape[:2]
                cv2.resizeWindow(
                    window_name,
                    max(640, int(width * args.window_scale)),
                    max(480, int(height * args.window_scale)),
                )
                window_ready = True
            cv2.imshow(window_name, preview)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    wall_elapsed = max(1e-6, time.perf_counter() - wall_start)
    cap.release()
    if args.show_window:
        cv2.destroyWindow(window_name)

    if not timings_ms:
        raise RuntimeError("No realtime frames were processed.")

    mean_ms = statistics.mean(timings_ms)
    median_ms = statistics.median(timings_ms)
    inference_fps = 1000.0 / mean_ms if mean_ms > 0 else 0.0
    end_to_end_fps = len(timings_ms) / wall_elapsed
    official = MODEL_METRICS[model_name]

    return {
        "model": model_name,
        "family": official["family"],
        "scale": official["scale"],
        "status": "ok",
        "frames": len(timings_ms),
        "local_mean_ms": round(mean_ms, 3),
        "local_median_ms": round(median_ms, 3),
        "local_fps": round(inference_fps, 3),
        "end_to_end_fps": round(end_to_end_fps, 3),
        "avg_detections_per_frame": round(detections / len(timings_ms), 3),
        "avg_people_keypoints_per_frame": round(people_with_keypoints / len(timings_ms), 3),
        "official_pose_map50_95": official["map50_95"],
        "official_pose_map50": official["map50"],
        "official_speed_ms": official["official_speed_ms"],
        "official_speed_hardware": official["speed_hardware"],
        "params_m": official["params_m"],
        "flops_b": official["flops_b"],
        "source_url": SOURCE_URLS[str(official["family"])],
    }


def write_csv(rows: List[Dict[str, float | str]], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_bar(rows: List[Dict[str, float | str]], key: str, title: str, ylabel: str, path: Path) -> None:
    labels = [str(row["model"]).replace("-pose.pt", "") for row in rows]
    values = [float(row[key]) for row in rows]
    palette = {"YOLOv8": "#4c78a8", "YOLO11": "#59a14f", "YOLO26": "#f28e2b"}
    colors = [palette[str(row["family"])] for row in rows]

    plt.figure(figsize=(14, 6))
    plt.bar(labels, values, color=colors)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_tradeoff(rows: List[Dict[str, float | str]], path: Path) -> None:
    palette = {"YOLOv8": "#4c78a8", "YOLO11": "#59a14f", "YOLO26": "#f28e2b"}
    plt.figure(figsize=(10, 6))

    for family in ["YOLOv8", "YOLO11", "YOLO26"]:
        family_rows = [row for row in rows if row["family"] == family]
        x_values = [float(row["local_mean_ms"]) for row in family_rows]
        y_values = [float(row["official_pose_map50_95"]) for row in family_rows]
        scales = [str(row["scale"]) for row in family_rows]
        plt.scatter(x_values, y_values, label=family, color=palette[family], s=90)
        for x_value, y_value, scale in zip(x_values, y_values, scales):
            plt.annotate(scale, (x_value, y_value), textcoords="offset points", xytext=(5, 5))

    plt.title("Accuracy vs Local GPU Inference Time")
    plt.xlabel("Local mean inference time (ms/frame)")
    plt.ylabel("Official COCO pose mAP50-95")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def write_report(rows: List[Dict[str, float | str]], output_path: Path, args: argparse.Namespace) -> None:
    best_fps = max(rows, key=lambda row: float(row["local_fps"]))
    best_accuracy = max(rows, key=lambda row: float(row["official_pose_map50_95"]))
    best_balance = max(rows, key=lambda row: float(row["official_pose_map50_95"]) / max(1.0, float(row["local_mean_ms"])))

    lines = [
        "# Pose Model Benchmark Report",
        "",
        "## Benchmark Setup",
        "",
        f"- Source: `{args.source}`",
        f"- Benchmark mode: `{args.benchmark_mode}`",
        f"- Device: `{args.device}`",
        f"- Image size: `{args.imgsz}`",
        f"- Frames per model: `{args.frames}` sampled every `{args.sample_step}` frame(s)",
        f"- Confidence threshold: `{args.conf}`",
        f"- Preview window: `{args.show_window}`",
        "",
        "## Important Notes",
        "",
        "- Local speed/FPS is measured on the project machine and is the fairest speed comparison for this demo.",
        "- `sampled-video` mode gives every model the same frames, so it is the fairest reproducible comparison.",
        "- `realtime` mode measures live camera/video processing behavior, but the scene can change between models.",
        "- If `--show-window` is used, local inference time remains model-only, but end-to-end FPS can include preview overhead.",
        "- Official mAP, parameters, and FLOPs are taken from Ultralytics model documentation/model cards.",
        "- Official YOLOv8 speed uses A100 TensorRT, while YOLO11 and YOLO26 list T4 TensorRT10, so official speed values should not be directly compared across all generations.",
        "- YOLO12-pose is excluded because official pretrained YOLO12-pose weights are not available; training YOLO12-pose would make the comparison dependent on custom training settings.",
        "",
        "## Summary",
        "",
        f"- Fastest local model: `{best_fps['model']}` at `{best_fps['local_fps']}` FPS.",
        f"- Highest official pose mAP50-95: `{best_accuracy['model']}` at `{best_accuracy['official_pose_map50_95']}`.",
        f"- Best simple accuracy/speed balance in this run: `{best_balance['model']}`.",
        "",
        "## Generated Graphs",
        "",
        "- `local_fps.png`: local GPU FPS comparison.",
        "- `local_mean_ms.png`: local GPU inference time comparison.",
        "- `official_pose_map.png`: official COCO pose accuracy comparison.",
        "- `accuracy_speed_tradeoff.png`: local speed vs official accuracy.",
        "- `params_m.png`: parameter count comparison.",
        "- `flops_b.png`: computational cost comparison.",
        "",
        "## References",
        "",
        f"- YOLOv8 model card: {SOURCE_URLS['YOLOv8']}",
        f"- YOLO11 model card: {SOURCE_URLS['YOLO11']}",
        f"- YOLO26 pose documentation: {SOURCE_URLS['YOLO26']}",
        f"- YOLO12 documentation: {SOURCE_URLS['YOLO12']}",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    device = require_cuda_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = None
    if args.benchmark_mode == "sampled-video":
        frames = load_sampled_frames(args.source, args.frames, args.sample_step)
    rows: List[Dict[str, float | str]] = []

    for model_name in MODEL_METRICS:
        print(f"[INFO] Benchmarking {model_name} on {device}...")
        try:
            if args.benchmark_mode == "sampled-video":
                assert frames is not None
                rows.append(benchmark_model(model_name, frames, args, device))
            else:
                rows.append(benchmark_model_realtime(model_name, args, device))
        except Exception as exc:
            official = MODEL_METRICS[model_name]
            frame_count = len(frames) if frames is not None else args.frames
            rows.append(
                {
                    "model": model_name,
                    "family": official["family"],
                    "scale": official["scale"],
                    "status": f"failed: {exc}",
                    "frames": frame_count,
                    "local_mean_ms": 0,
                    "local_median_ms": 0,
                    "local_fps": 0,
                    "end_to_end_fps": 0,
                    "avg_detections_per_frame": 0,
                    "avg_people_keypoints_per_frame": 0,
                    "official_pose_map50_95": official["map50_95"],
                    "official_pose_map50": official["map50"],
                    "official_speed_ms": official["official_speed_ms"],
                    "official_speed_hardware": official["speed_hardware"],
                    "params_m": official["params_m"],
                    "flops_b": official["flops_b"],
                    "source_url": SOURCE_URLS[str(official["family"])],
                }
            )
            print(f"[WARN] {model_name} failed: {exc}")

    write_csv(rows, output_dir / "pose_model_comparison.csv")
    ok_rows = [row for row in rows if row["status"] == "ok"]

    if ok_rows:
        plot_bar(ok_rows, "local_fps", "Local GPU FPS by Pose Model", "FPS", output_dir / "local_fps.png")
        plot_bar(ok_rows, "local_mean_ms", "Local GPU Inference Time by Pose Model", "ms/frame", output_dir / "local_mean_ms.png")
        plot_bar(ok_rows, "official_pose_map50_95", "Official COCO Pose Accuracy", "mAP50-95", output_dir / "official_pose_map.png")
        plot_bar(ok_rows, "params_m", "Model Parameter Count", "Parameters (M)", output_dir / "params_m.png")
        plot_bar(ok_rows, "flops_b", "Model FLOPs", "FLOPs (B)", output_dir / "flops_b.png")
        plot_tradeoff(ok_rows, output_dir / "accuracy_speed_tradeoff.png")
        write_report(ok_rows, output_dir / "pose_model_benchmark_report.md", args)

    print(f"[INFO] Results written to: {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
