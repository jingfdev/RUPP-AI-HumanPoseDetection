"""Extract tabular posture features from labeled images using YOLO pose keypoints."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np

from async_video_pipeline import (
    LEFT_ELBOW,
    LEFT_EYE,
    LEFT_HIP,
    LEFT_SHOULDER,
    LEFT_WRIST,
    NOSE,
    RIGHT_ELBOW,
    RIGHT_EYE,
    RIGHT_HIP,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
    require_cuda_device,
)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
FEATURE_FIELDS = [
    "image_path",
    "label",
    "mapped_label",
    "model",
    "box_area_ratio",
    "box_aspect_ratio",
    "visible_keypoints",
    "head_low_ratio",
    "torso_angle_abs",
    "torso_vertical_span",
    "head_to_wrists",
    "head_to_elbows",
    "shoulder_slope",
    "hip_slope",
    "shoulder_width_ratio",
    "hip_width_ratio",
    "head_to_hips_y",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract YOLO pose features from a labeled posture dataset.")
    parser.add_argument("--dataset-dir", default="posture_dataset/raw", help="Folder containing label subfolders.")
    parser.add_argument("--output", default="posture_dataset/features/posture_features.csv", help="Output CSV path.")
    parser.add_argument("--model", default="yolo11s-pose.pt", help="YOLO pose model used as keypoint extractor.")
    parser.add_argument("--device", default="cuda:0", help="CUDA device only.")
    parser.add_argument("--conf", type=float, default=0.35, help="YOLO confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO image size.")
    parser.add_argument(
        "--confusing-as",
        default="normal",
        choices=["normal", "sleeping", "drop"],
        help="How to map confusing samples for binary training.",
    )
    return parser.parse_args()


def image_paths(dataset_dir: Path) -> List[Tuple[Path, str]]:
    items: List[Tuple[Path, str]] = []
    for label_dir in sorted(path for path in dataset_dir.iterdir() if path.is_dir()):
        label = label_dir.name.lower()
        for path in sorted(label_dir.rglob("*")):
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                items.append((path, label))
    return items


def mean_point(kp: np.ndarray, conf: np.ndarray, indices: Iterable[int]) -> np.ndarray | None:
    points = []
    for index in indices:
        if index < len(kp) and conf[index] >= 0.25 and kp[index][0] > 0 and kp[index][1] > 0:
            points.append(kp[index])
    if not points:
        return None
    return np.mean(np.asarray(points, dtype=np.float32), axis=0)


def pair_slope(kp: np.ndarray, conf: np.ndarray, left: int, right: int, box_h: float) -> float:
    if conf[left] < 0.25 or conf[right] < 0.25:
        return -1.0
    return float((kp[right][1] - kp[left][1]) / max(1.0, box_h))


def pair_width(kp: np.ndarray, conf: np.ndarray, left: int, right: int, box_w: float) -> float:
    if conf[left] < 0.25 or conf[right] < 0.25:
        return -1.0
    return float(abs(kp[right][0] - kp[left][0]) / max(1.0, box_w))


def normalized_distance(a: np.ndarray | None, b: np.ndarray | None, scale: float) -> float:
    if a is None or b is None:
        return -1.0
    return float(np.linalg.norm(a - b) / max(1.0, scale))


def keypoint_confidences(result) -> np.ndarray:
    if result.keypoints.conf is not None:
        return result.keypoints.conf.cpu().numpy()
    keypoints = result.keypoints.xy.cpu().numpy()
    return np.ones((keypoints.shape[0], keypoints.shape[1]), dtype=np.float32)


def largest_person_index(boxes: np.ndarray) -> int:
    areas = (boxes[:, 2] - boxes[:, 0]).clip(min=0) * (boxes[:, 3] - boxes[:, 1]).clip(min=0)
    return int(np.argmax(areas))


def extract_features(result, frame_shape: Tuple[int, int, int]) -> Dict[str, float] | None:
    if result.boxes is None or result.keypoints is None or len(result.boxes) == 0:
        return None

    boxes = result.boxes.xyxy.cpu().numpy()
    keypoints = result.keypoints.xy.cpu().numpy()
    confidences = keypoint_confidences(result)
    person_index = largest_person_index(boxes)
    if person_index >= len(keypoints):
        return None

    box = boxes[person_index]
    kp = keypoints[person_index]
    conf = confidences[person_index]
    x1, y1, x2, y2 = box
    box_w = max(1.0, float(x2 - x1))
    box_h = max(1.0, float(y2 - y1))
    frame_area = float(frame_shape[0] * frame_shape[1])
    box_area = box_w * box_h

    head = mean_point(kp, conf, [NOSE, LEFT_EYE, RIGHT_EYE])
    shoulders = mean_point(kp, conf, [LEFT_SHOULDER, RIGHT_SHOULDER])
    hips = mean_point(kp, conf, [LEFT_HIP, RIGHT_HIP])
    wrists = mean_point(kp, conf, [LEFT_WRIST, RIGHT_WRIST])
    elbows = mean_point(kp, conf, [LEFT_ELBOW, RIGHT_ELBOW])

    head_low_ratio = -1.0
    if head is not None and shoulders is not None:
        head_low_ratio = float((head[1] - shoulders[1]) / box_h)

    torso_angle_abs = -1.0
    torso_vertical_span = -1.0
    if shoulders is not None and hips is not None:
        torso_vec = hips - shoulders
        torso_angle_abs = abs(float(np.degrees(np.arctan2(torso_vec[1], torso_vec[0]))))
        torso_vertical_span = float(abs(hips[1] - shoulders[1]) / box_h)

    head_to_hips_y = -1.0
    if head is not None and hips is not None:
        head_to_hips_y = float((head[1] - hips[1]) / box_h)

    return {
        "box_area_ratio": float(box_area / max(1.0, frame_area)),
        "box_aspect_ratio": float(box_w / box_h),
        "visible_keypoints": float(np.count_nonzero(conf >= 0.25)),
        "head_low_ratio": head_low_ratio,
        "torso_angle_abs": torso_angle_abs,
        "torso_vertical_span": torso_vertical_span,
        "head_to_wrists": normalized_distance(head, wrists, max(box_w, box_h)),
        "head_to_elbows": normalized_distance(head, elbows, max(box_w, box_h)),
        "shoulder_slope": pair_slope(kp, conf, LEFT_SHOULDER, RIGHT_SHOULDER, box_h),
        "hip_slope": pair_slope(kp, conf, LEFT_HIP, RIGHT_HIP, box_h),
        "shoulder_width_ratio": pair_width(kp, conf, LEFT_SHOULDER, RIGHT_SHOULDER, box_w),
        "hip_width_ratio": pair_width(kp, conf, LEFT_HIP, RIGHT_HIP, box_w),
        "head_to_hips_y": head_to_hips_y,
    }


def mapped_label(label: str, confusing_as: str) -> str | None:
    if label in {"normal", "sleeping"}:
        return label
    if label == "confusing":
        return None if confusing_as == "drop" else confusing_as
    return None


def main() -> int:
    args = parse_args()
    device = require_cuda_device(args.device)
    dataset_dir = Path(args.dataset_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO

    model = YOLO(args.model)
    rows: List[Dict[str, float | str]] = []
    skipped = 0

    for path, label in image_paths(dataset_dir):
        target_label = mapped_label(label, args.confusing_as)
        if target_label is None:
            skipped += 1
            continue
        frame = cv2.imread(str(path))
        if frame is None:
            skipped += 1
            continue
        result = model.predict(frame, device=device, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
        features = extract_features(result, frame.shape)
        if features is None:
            skipped += 1
            continue
        row: Dict[str, float | str] = {
            "image_path": str(path),
            "label": label,
            "mapped_label": target_label,
            "model": args.model,
        }
        row.update(features)
        rows.append(row)

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FEATURE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[INFO] Wrote {len(rows)} feature row(s) to {output_path}")
    print(f"[INFO] Skipped {skipped} image(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
