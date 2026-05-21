"""Benchmark YOLO pose models (YOLOv8, YOLO11, YOLO26) on local GPU.

GPU-ONLY MODE: This script requires CUDA GPU and will refuse to run on CPU.

Measures accuracy (mAP50-95), running time (ms), and FPS for each model generation and size.
Generates line graphs comparing all models and a combined summary graph.
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

from async_video_pipeline import SleepPoseClassifier, draw_pose_labels, require_cuda_device


MODEL_METRICS: Dict[str, Dict[str, float | str]] = {
    # YOLOv8 Pose models
    "yolov8n-pose.pt": {"family": "YOLOv8", "scale": "n", "map50_95": 50.4, "official_speed_ms": 1.18, "speed_hardware": "A100 TensorRT"},
    "yolov8s-pose.pt": {"family": "YOLOv8", "scale": "s", "map50_95": 60.0, "official_speed_ms": 1.42, "speed_hardware": "A100 TensorRT"},
    "yolov8m-pose.pt": {"family": "YOLOv8", "scale": "m", "map50_95": 65.0, "official_speed_ms": 2.00, "speed_hardware": "A100 TensorRT"},
    "yolov8l-pose.pt": {"family": "YOLOv8", "scale": "l", "map50_95": 67.6, "official_speed_ms": 2.59, "speed_hardware": "A100 TensorRT"},
    "yolov8x-pose.pt": {"family": "YOLOv8", "scale": "x", "map50_95": 69.2, "official_speed_ms": 3.73, "speed_hardware": "A100 TensorRT"},
    
    # YOLO11 Pose models
    "yolo11n-pose.pt": {"family": "YOLO11", "scale": "n", "map50_95": 50.0, "official_speed_ms": 1.7, "speed_hardware": "T4 TensorRT10"},
    "yolo11s-pose.pt": {"family": "YOLO11", "scale": "s", "map50_95": 58.9, "official_speed_ms": 2.6, "speed_hardware": "T4 TensorRT10"},
    "yolo11m-pose.pt": {"family": "YOLO11", "scale": "m", "map50_95": 64.9, "official_speed_ms": 4.9, "speed_hardware": "T4 TensorRT10"},
    "yolo11l-pose.pt": {"family": "YOLO11", "scale": "l", "map50_95": 66.1, "official_speed_ms": 6.4, "speed_hardware": "T4 TensorRT10"},
    "yolo11x-pose.pt": {"family": "YOLO11", "scale": "x", "map50_95": 69.5, "official_speed_ms": 12.1, "speed_hardware": "T4 TensorRT10"},
    
    # YOLO26 Pose models
    "yolo26n-pose.pt": {"family": "YOLO26", "scale": "n", "map50_95": 57.2, "official_speed_ms": 1.8, "speed_hardware": "T4 TensorRT10"},
    "yolo26s-pose.pt": {"family": "YOLO26", "scale": "s", "map50_95": 63.0, "official_speed_ms": 2.7, "speed_hardware": "T4 TensorRT10"},
    "yolo26m-pose.pt": {"family": "YOLO26", "scale": "m", "map50_95": 68.8, "official_speed_ms": 5.0, "speed_hardware": "T4 TensorRT10"},
    "yolo26l-pose.pt": {"family": "YOLO26", "scale": "l", "map50_95": 70.4, "official_speed_ms": 6.5, "speed_hardware": "T4 TensorRT10"},
    "yolo26x-pose.pt": {"family": "YOLO26", "scale": "x", "map50_95": 71.6, "official_speed_ms": 12.2, "speed_hardware": "T4 TensorRT10"},
}

SOURCE_URLS = {
    "YOLOv8": "https://huggingface.co/Ultralytics/YOLOv8",
    "YOLO11": "https://huggingface.co/Ultralytics/YOLO11",
    "YOLO26": "https://docs.ultralytics.com/tasks/pose/",
}

HISTORICAL_GENERATION_REVIEW = [
    {
        "generation": "YOLOv5",
        "main_benchmark": "No",
        "result": "Excluded from direct benchmark",
        "reason": "No official Ultralytics pretrained COCO 17-keypoint pose weights in the same YOLO(...)-pose workflow.",
    },
    {
        "generation": "YOLOv6",
        "main_benchmark": "No",
        "result": "Excluded from direct benchmark",
        "reason": "Not an official Ultralytics pretrained pose family compatible with this pipeline.",
    },
    {
        "generation": "YOLOv7",
        "main_benchmark": "No",
        "result": "Excluded from direct benchmark",
        "reason": "Different older ecosystem; not a direct official Ultralytics pretrained pose model family for this workflow.",
    },
    {
        "generation": "YOLOv8",
        "main_benchmark": "Yes",
        "result": "Benchmarked",
        "reason": "Official pretrained Ultralytics pose family with compatible COCO 17-keypoint output.",
    },
    {
        "generation": "YOLOv9",
        "main_benchmark": "No",
        "result": "Excluded from direct benchmark",
        "reason": "Not part of the official compatible pretrained pose model set used by this project.",
    },
    {
        "generation": "YOLOv10",
        "main_benchmark": "No",
        "result": "Excluded from direct benchmark",
        "reason": "Not part of the official compatible pretrained pose model set used by this project.",
    },
    {
        "generation": "YOLO11",
        "main_benchmark": "Yes",
        "result": "Benchmarked",
        "reason": "Current project model family with official pretrained Ultralytics pose weights.",
    },
    {
        "generation": "YOLO12",
        "main_benchmark": "No",
        "result": "Excluded from direct benchmark",
        "reason": "Pose architecture exists, but official pretrained YOLO12-pose weights are not available for fair direct comparison.",
    },
    {
        "generation": "YOLO26",
        "main_benchmark": "Yes",
        "result": "Benchmarked",
        "reason": "Newer official Ultralytics pretrained pose family with compatible COCO 17-keypoint output.",
    },
]

RESULT_FIELDS = ["model", "family", "scale", "status", "accuracy_map50_95", "running_time_ms", "fps"]
HISTORICAL_REVIEW_FIELDS = ["generation", "main_benchmark", "result", "reason"]
OBSOLETE_OUTPUTS = [
    "accuracy_speed_tradeoff.png",
    "flops_b.png",
    "flops_b_line.png",
    "local_fps.png",
    "local_fps_line.png",
    "local_mean_ms.png",
    "local_mean_ms_line.png",
    "official_pose_map.png",
    "official_pose_map_line.png",
    "params_m.png",
    "params_m_line.png",
    "comparison_summary_graph.png",
]

FAMILY_TEMPLATES = {
    "YOLOv8": "yolov8{scale}-pose.pt",
    "YOLO11": "yolo11{scale}-pose.pt",
    "YOLO26": "yolo26{scale}-pose.pt",
}

SCALE_KEYS = {
    "1": "n",
    "2": "s",
    "3": "m",
    "4": "l",
    "5": "x",
    "n": "n",
    "s": "s",
    "m": "m",
    "l": "l",
    "x": "x",
}

# Model families for comparison and their color palette
MODEL_FAMILIES = ["YOLOv8", "YOLO11", "YOLO26"]
FAMILY_COLORS = {
    "YOLOv8": "#d62728",  # red
    "YOLO11": "#9467bd",  # purple
    "YOLO26": "#8c564b",  # brown
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark official Ultralytics pose models on CUDA (GPU ONLY - CPU not supported).")
    parser.add_argument("--source", default="sample_video.mp4", help="Video file used for local speed benchmarking.")
    parser.add_argument("--device", default="cuda:0", help="CUDA GPU device (e.g., cuda or cuda:0). CPU is NOT supported.")
    parser.add_argument("--conf", type=float, default=0.35, help="Inference confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--frames", type=int, default=90, help="Maximum sampled frames per model.")
    parser.add_argument("--sample-step", type=int, default=3, help="Analyze every Nth frame from source video.")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup predictions before timing each model.")
    parser.add_argument("--output-dir", default="benchmark_results", help="Directory for CSV, plots, and report.")
    parser.add_argument(
        "--from-csv",
        default="",
        help="Regenerate plots/report from an existing pose_model_comparison.csv without rerunning models.",
    )
    parser.add_argument(
        "--benchmark-mode",
        choices=["sampled-video", "realtime", "live-compare"],
        default="sampled-video",
        help="sampled-video reuses frames; realtime benchmarks live input; live-compare opens one dashboard window with three model panels.",
    )
    parser.add_argument(
        "--show-window",
        action="store_true",
        help="Show a live annotated preview window during realtime benchmarking.",
    )
    parser.add_argument(
        "--window-scale",
        type=float,
        default=1.0,
        help="Scale factor for preview windows when explicit width/height are not used.",
    )
    parser.add_argument("--window-width", type=int, default=1080, help="Popup window width for preview modes.")
    parser.add_argument("--window-height", type=int, default=1080, help="Popup window height for preview modes.")
    parser.add_argument(
        "--scale",
        default="s",
        choices=["n", "s", "m", "l", "x"],
        help="Initial model size for live-compare mode.",
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


def validate_gpu_device(device: str) -> None:
    """Ensure GPU-only execution. Refuse CPU fallback."""
    if not device.lower().startswith("cuda"):
        raise RuntimeError(f"GPU-ONLY mode: device '{device}' is not CUDA. CPU execution is not supported.")
    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("GPU-ONLY mode: CUDA not available on this system. Cannot proceed without GPU.")
    except ImportError:
        raise RuntimeError("PyTorch not available. Cannot check GPU availability.")


def release_cuda_memory() -> None:
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def comparison_model_name(family: str, scale: str) -> str:
    return FAMILY_TEMPLATES[family].format(scale=scale)


def load_comparison_models(scale: str, device: str) -> Dict[str, object]:
    from ultralytics import YOLO

    release_cuda_memory()
    models: Dict[str, object] = {}
    # Only load YOLOv8, YOLO11, YOLO26 for live comparison (fits in 2x2 grid)
    for family in ["YOLOv8", "YOLO11", "YOLO26"]:
        name = comparison_model_name(family, scale)
        print(f"[INFO] Loading {name} on {device}...")
        models[family] = YOLO(name)
    return models


def draw_comparison_overlay(frame: np.ndarray, family: str, scale: str, inference_ms: float) -> None:
    name = comparison_model_name(family, scale)
    metrics = MODEL_METRICS[name]
    lines = [
        f"{name}",
        f"{inference_ms:.1f} ms | {1000.0 / max(1e-6, inference_ms):.1f} FPS",
        f"mAP50-95 {metrics['map50_95']}",
        "1=n 2=s 3=m 4=l 5=x | q=quit",
    ]

    y = 22
    for line in lines:
        cv2.putText(
            frame,
            line,
            (8, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y += 20


def fit_panel(panel: np.ndarray, cell_width: int, cell_height: int) -> np.ndarray:
    cell = np.zeros((cell_height, cell_width, 3), dtype=np.uint8)
    height, width = panel.shape[:2]
    scale = min(cell_width / max(1, width), cell_height / max(1, height))
    resized_width = max(1, int(width * scale))
    resized_height = max(1, int(height * scale))
    resized = cv2.resize(panel, (resized_width, resized_height))
    x1 = (cell_width - resized_width) // 2
    y1 = (cell_height - resized_height) // 2
    cell[y1:y1 + resized_height, x1:x1 + resized_width] = resized
    return cell


def make_info_panel(width: int, height: int, scale: str) -> np.ndarray:
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    lines = [
        "YOLO Pose Version Comparison",
        "",
        "Panels:",
        "YOLOv8   YOLO11",
        "YOLO26  Controls",
        "",
        "Keys:",
        "1 = nano",
        "2 = small",
        "3 = medium",
        "4 = large",
        "5 = extra-large",
        "",
        f"Current size: {scale}",
        "q / ESC = quit",
    ]
    y = 40
    for line in lines:
        cv2.putText(panel, line, (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        y += 34
    return panel


def make_live_dashboard(panels: List[np.ndarray], canvas_width: int, canvas_height: int, scale: str) -> np.ndarray:
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    cell_width = canvas_width // 2
    cell_height = canvas_height // 2
    cells = [fit_panel(panel, cell_width, cell_height) for panel in panels]
    cells.append(make_info_panel(cell_width, cell_height, scale))

    positions = [(0, 0), (cell_width, 0), (0, cell_height), (cell_width, cell_height)]
    for cell, (x, y) in zip(cells, positions):
        canvas[y:y + cell_height, x:x + cell_width] = cell
        cv2.rectangle(canvas, (x, y), (x + cell_width - 1, y + cell_height - 1), (70, 70, 70), 1)

    return canvas


def draw_live_compare_labels(frame: np.ndarray, result, classifier: SleepPoseClassifier, now: float) -> np.ndarray:
    labels = classifier.classify(result, now, frame.shape)
    return draw_pose_labels(frame, labels, show_score=True, show_reason=False)


def draw_panel_title(frame: np.ndarray, family: str, scale: str, inference_ms: float) -> None:
    name = comparison_model_name(family, scale)
    metrics = MODEL_METRICS[name]
    lines = [
        f"{name} | {inference_ms:.1f} ms | {1000.0 / max(1e-6, inference_ms):.1f} FPS",
        f"mAP50-95 {metrics['map50_95']}",
    ]
    y = 30
    for line in lines:
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        y += 28


def run_live_compare(args: argparse.Namespace, device: str) -> int:
    cap = cv2.VideoCapture(0 if args.source.isdigit() else args.source)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open live comparison source: {args.source}")

    current_scale = args.scale
    models = load_comparison_models(current_scale, device)
    classifiers = {
        family: SleepPoseClassifier(
            sleep_threshold=0.55,
            persist_seconds=0.0,
            min_box_area_ratio=0.02,
            min_keypoints=5,
        )
        for family in ["YOLOv8", "YOLO11", "YOLO26"]
    }
    window_name = "YOLO Pose Version Comparison"
    window_ready = False
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("[INFO] Live compare controls: 1=n, 2=s, 3=m, 4=l, 5=x, or n/s/m/l/x. Press q or ESC to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        panels: List[np.ndarray] = []
        now = time.time()
        for family, model in models.items():
            start = time.perf_counter()
            result = model.predict(frame.copy(), device=device, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
            cuda_synchronize()
            inference_ms = (time.perf_counter() - start) * 1000.0

            preview = result.plot(boxes=False, labels=False, kpt_line=True, kpt_radius=4)
            preview = draw_live_compare_labels(preview, result, classifiers[family], now)
            draw_panel_title(preview, family, current_scale, inference_ms)
            panels.append(preview)

        dashboard = make_live_dashboard(panels, args.window_width, args.window_height, current_scale)
        if not window_ready:
            cv2.resizeWindow(
                window_name,
                args.window_width,
                args.window_height,
            )
            window_ready = True

        cv2.imshow(window_name, dashboard)

        key = cv2.waitKey(1) & 0xFF
        if key in {27, ord("q")}:
            break
        if key != 255:
            pressed = chr(key).lower()
            if pressed in SCALE_KEYS:
                next_scale = SCALE_KEYS[pressed]
                if next_scale != current_scale:
                    print(f"[INFO] Switching comparison scale: {current_scale} -> {next_scale}")
                    models = load_comparison_models(next_scale, device)
                    classifiers = {
                        family: SleepPoseClassifier(
                            sleep_threshold=0.55,
                            persist_seconds=0.0,
                            min_box_area_ratio=0.02,
                            min_keypoints=5,
                        )
                        for family in ["YOLOv8", "YOLO11", "YOLO26"]
                    }
                    current_scale = next_scale

    cap.release()
    cv2.destroyAllWindows()
    return 0


def benchmark_model(model_name: str, frames: List[np.ndarray], args: argparse.Namespace, device: str) -> Dict[str, float | str]:
    from ultralytics import YOLO

    model = YOLO(model_name)
    for _ in range(args.warmup):
        model.predict(frames[0], device=device, conf=args.conf, imgsz=args.imgsz, verbose=False)
    cuda_synchronize()

    timings_ms: List[float] = []

    for frame in frames:
        start = time.perf_counter()
        result = model.predict(frame, device=device, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
        cuda_synchronize()
        timings_ms.append((time.perf_counter() - start) * 1000.0)

    mean_ms = statistics.mean(timings_ms)
    fps = 1000.0 / mean_ms if mean_ms > 0 else 0.0
    official = MODEL_METRICS[model_name]

    return {
        "model": model_name,
        "family": official["family"],
        "scale": official["scale"],
        "status": "ok",
        "running_time_ms": round(mean_ms, 3),
        "fps": round(fps, 3),
        "accuracy_map50_95": official["map50_95"],
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
                cv2.resizeWindow(
                    window_name,
                    args.window_width,
                    args.window_height,
                )
                window_ready = True
            cv2.imshow(window_name, preview)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    if args.show_window:
        cv2.destroyWindow(window_name)

    if not timings_ms:
        raise RuntimeError("No realtime frames were processed.")

    mean_ms = statistics.mean(timings_ms)
    fps = 1000.0 / mean_ms if mean_ms > 0 else 0.0
    official = MODEL_METRICS[model_name]

    return {
        "model": model_name,
        "family": official["family"],
        "scale": official["scale"],
        "status": "ok",
        "running_time_ms": round(mean_ms, 3),
        "fps": round(fps, 3),
        "accuracy_map50_95": official["map50_95"],
    }


def write_csv(rows: List[Dict[str, float | str]], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_historical_generation_review(output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=HISTORICAL_REVIEW_FIELDS)
        writer.writeheader()
        writer.writerows(HISTORICAL_GENERATION_REVIEW)


def read_csv_rows(path: Path) -> List[Dict[str, float | str]]:
    with path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def normalize_rows(rows: List[Dict[str, float | str]]) -> List[Dict[str, float | str]]:
    normalized: List[Dict[str, float | str]] = []
    for row in rows:
        normalized.append(
            {
                "model": row["model"],
                "family": row["family"],
                "scale": row["scale"],
                "status": row.get("status", "ok"),
                "accuracy_map50_95": row.get("accuracy_map50_95", row.get("official_pose_map50_95", 0)),
                "running_time_ms": row.get("running_time_ms", row.get("local_mean_ms", 0)),
                "fps": row.get("fps", row.get("local_fps", 0)),
            }
        )
    return normalized


def cleanup_old_outputs(output_dir: Path) -> None:
    for filename in OBSOLETE_OUTPUTS:
        path = output_dir / filename
        if path.exists():
            path.unlink()


def write_outputs(rows: List[Dict[str, float | str]], output_dir: Path, args: argparse.Namespace) -> None:
    cleanup_old_outputs(output_dir)
    rows = normalize_rows(rows)
    write_csv(rows, output_dir / "pose_model_comparison.csv")
    write_historical_generation_review(output_dir / "historical_yolo_generation_review.csv")
    ok_rows = [row for row in rows if row["status"] == "ok"]

    if ok_rows:
        plot_line_by_scale(ok_rows, "fps", "FPS Comparison by Model Size", "FPS (higher is better)", output_dir / "fps_graph.png")
        plot_line_by_scale(ok_rows, "running_time_ms", "Running Time Comparison by Model Size", "ms/frame (lower is better)", output_dir / "running_time_graph.png")
        plot_line_by_scale(ok_rows, "accuracy_map50_95", "Accuracy Comparison by Model Size", "mAP50-95 (higher is better)", output_dir / "accuracy_graph.png")
        plot_accuracy_runtime_tradeoff(ok_rows, output_dir / "accuracy_runtime_tradeoff_graph.png")
        plot_combined_summary(ok_rows, output_dir / "combined_summary_graph.png")
        write_report(ok_rows, output_dir / "pose_model_benchmark_report.md", args)


def compact_model_label(row: Dict[str, float | str]) -> str:
    family = str(row["family"]).replace("YOLOv", "").replace("YOLO", "")
    return f"{family}{row['scale']}"


def annotate_point(axis, x_value, y_value, label: str, color: str, index: int, value: float | None = None) -> None:
    offsets = [(6, 8), (6, -14), (-22, 8), (-22, -14), (8, 0)]
    offset = offsets[index % len(offsets)]
    text = label if value is None else f"{label}\n{value:.1f}"
    axis.annotate(
        text,
        (x_value, y_value),
        textcoords="offset points",
        xytext=offset,
        fontsize=8,
        color=color,
        fontweight="bold",
        alpha=0.9,
    )


def plot_line_by_scale(rows: List[Dict[str, float | str]], key: str, title: str, ylabel: str, path: Path) -> None:
    """Plot line graphs for each metric by model size."""
    scale_order = ["n", "s", "m", "l", "x"]

    fig, axis = plt.subplots(figsize=(12, 7))
    for family in MODEL_FAMILIES:
        family_rows = {str(row["scale"]): row for row in rows if row["family"] == family}
        values = [float(family_rows[scale][key]) for scale in scale_order if scale in family_rows]
        labels = [scale for scale in scale_order if scale in family_rows]
        if values:
            color = FAMILY_COLORS[family]
            axis.plot(labels, values, marker="o", markersize=6, linewidth=2.5, label=family, color=color)
            for index, (scale, value) in enumerate(zip(labels, values)):
                annotate_point(axis, scale, value, compact_model_label(family_rows[scale]), color, index, value)

    axis.set_title(title, fontsize=14, fontweight="bold")
    axis.set_xlabel("Model Size", fontsize=12)
    axis.set_ylabel(ylabel, fontsize=12)
    axis.grid(True, alpha=0.3, linestyle="--")
    axis.legend(loc="best", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_accuracy_runtime_tradeoff(rows: List[Dict[str, float | str]], path: Path) -> None:
    """Plot accuracy against local running time with connected model-size lines."""
    scale_order = ["n", "s", "m", "l", "x"]

    fig, axis = plt.subplots(figsize=(13, 7.5))
    for family in MODEL_FAMILIES:
        family_rows = {str(row["scale"]): row for row in rows if row["family"] == family}
        ordered_rows = [family_rows[scale] for scale in scale_order if scale in family_rows]
        if not ordered_rows:
            continue
        x_values = [float(row["running_time_ms"]) for row in ordered_rows]
        y_values = [float(row["accuracy_map50_95"]) for row in ordered_rows]
        color = FAMILY_COLORS[family]
        axis.plot(x_values, y_values, marker="o", markersize=7, linewidth=2.8, label=family, color=color)
        for index, (row, x_value, y_value) in enumerate(zip(ordered_rows, x_values, y_values)):
            annotate_point(axis, x_value, y_value, compact_model_label(row), color, index)

    axis.set_title("Accuracy vs Running Time Tradeoff", fontsize=15, fontweight="bold")
    axis.set_xlabel("Local GPU running time (ms/frame, lower is better)", fontsize=12)
    axis.set_ylabel("Official COCO pose mAP50-95 (higher is better)", fontsize=12)
    axis.grid(True, alpha=0.3, linestyle="--")
    axis.legend(loc="best", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_combined_summary(rows: List[Dict[str, float | str]], path: Path) -> None:
    """Plot all three metrics in a single figure with subplots."""
    scale_order = ["n", "s", "m", "l", "x"]
    metrics = [
        ("fps", "FPS Comparison", "FPS (higher is better)"),
        ("running_time_ms", "Running Time Comparison", "ms/frame (lower is better)"),
        ("accuracy_map50_95", "Accuracy Comparison", "mAP50-95 (higher is better)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle("YOLO Pose Model Comparison: Accuracy, Running Time, FPS", fontsize=16, fontweight="bold")
    
    for axis, (key, title, ylabel) in zip(axes, metrics):
        for family in MODEL_FAMILIES:
            family_rows = {str(row["scale"]): row for row in rows if row["family"] == family}
            values = [float(family_rows[scale][key]) for scale in scale_order if scale in family_rows]
            labels = [scale for scale in scale_order if scale in family_rows]
            if values:
                color = FAMILY_COLORS[family]
                axis.plot(labels, values, marker="o", markersize=4, linewidth=2.2, label=family, color=color)
                for index, (scale, value) in enumerate(zip(labels, values)):
                    annotate_point(axis, scale, value, compact_model_label(family_rows[scale]), color, index)
        
        axis.set_title(title, fontsize=12, fontweight="bold")
        axis.set_xlabel("Model Size", fontsize=11)
        axis.set_ylabel(ylabel, fontsize=11)
        axis.grid(True, alpha=0.3, linestyle="--")
        axis.legend(loc="best", fontsize=10)

    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def write_report(rows: List[Dict[str, float | str]], output_path: Path, args: argparse.Namespace) -> None:
    best_fps = max(rows, key=lambda row: float(row["fps"]))
    best_accuracy = max(rows, key=lambda row: float(row["accuracy_map50_95"]))
    best_runtime = min(rows, key=lambda row: float(row["running_time_ms"]))

    lines = [
        "# YOLO Pose Model Benchmark Report",
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
        "## Metrics Measured",
        "",
        "This benchmark measures **only three key metrics** for each model:",
        "",
        "1. **Accuracy (mAP50-95)**: Official COCO pose accuracy from Ultralytics documentation",
        "2. **Running Time (ms)**: Local inference time per frame measured on the project machine",
        "3. **FPS**: Frames per second (1000 / running_time_ms)",
        "",
        "## Important Notes",
        "",
        "- FPS and running time are measured on the local GPU device specified.",
        "- `sampled-video` mode feeds every model the same frames, providing a fair and reproducible comparison.",
        "- `realtime` mode measures live video/camera processing but scenes may differ between model runs.",
        "- Accuracy values are official metrics from Ultralytics model cards (COCO pose mAP50-95).",
        "- Models included: YOLOv8, YOLO11, YOLO26 across nano (n), small (s), medium (m), large (l), and extra-large (x) sizes.",
        "- Older YOLO generations are reviewed separately in `historical_yolo_generation_review.csv`; they are not mixed into the direct benchmark because they are not equivalent official pretrained pose targets for this pipeline.",
        "",
        "## Summary - Best Performers",
        "",
        f"- **Highest FPS**: `{best_fps['model']}` ({best_fps['family']} {best_fps['scale']}) at `{best_fps['fps']}` FPS",
        f"- **Lowest Running Time**: `{best_runtime['model']}` ({best_runtime['family']} {best_runtime['scale']}) at `{best_runtime['running_time_ms']}` ms/frame",
        f"- **Highest Accuracy**: `{best_accuracy['model']}` ({best_accuracy['family']} {best_accuracy['scale']}) at `{best_accuracy['accuracy_map50_95']}` mAP50-95",
        "",
        "## Generated Graphs",
        "",
        "- `fps_graph.png`: FPS comparison across all models and sizes (line graph)",
        "- `running_time_graph.png`: Running time comparison across all models and sizes (line graph)",
        "- `accuracy_graph.png`: Accuracy comparison across all models and sizes (line graph)",
        "- `accuracy_runtime_tradeoff_graph.png`: Accuracy compared with local running time using connected model-size lines",
        "- `combined_summary_graph.png`: All three metrics in a single figure with three subplots (line graphs)",
        "- `historical_yolo_generation_review.csv`: Review result for other YOLO generations considered but not directly benchmarked",
        "",
        "## Historical YOLO Generations Reviewed",
        "",
        "| Generation | Main Benchmark | Review Result | Reason |",
        "| --- | --- | --- | --- |",
        *[
            f"| {row['generation']} | {row['main_benchmark']} | {row['result']} | {row['reason']} |"
            for row in HISTORICAL_GENERATION_REVIEW
        ],
        "",
        "## Model References",
        "",
        f"- YOLOv8: {SOURCE_URLS['YOLOv8']}",
        f"- YOLO11: {SOURCE_URLS['YOLO11']}",
        f"- YOLO26: {SOURCE_URLS['YOLO26']}",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.from_csv:
        rows = read_csv_rows(Path(args.from_csv))
        write_outputs(rows, output_dir, args)
        print(f"[INFO] Graphs/report regenerated from CSV into: {output_dir.resolve()}")
        return 0

    # GPU-ONLY enforcement for all modes that run model inference.
    validate_gpu_device(args.device)

    device = require_cuda_device(args.device)
    if args.benchmark_mode == "live-compare":
        return run_live_compare(args, device)

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
            rows.append(
                {
                    "model": model_name,
                    "family": official["family"],
                    "scale": official["scale"],
                    "status": f"failed: {exc}",
                    "running_time_ms": 0,
                    "fps": 0,
                    "accuracy_map50_95": official["map50_95"],
                }
            )
            print(f"[WARN] {model_name} failed: {exc}")

    write_outputs(rows, output_dir, args)

    print(f"[INFO] Results written to: {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
