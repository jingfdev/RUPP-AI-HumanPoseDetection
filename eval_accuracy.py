"""Evaluate pose keypoint accuracy (PCK and OKS) for YOLOv8 vs MediaPipe.

Expected dataset: COCO keypoints-style JSON with images/annotations.
Images must be available locally under --images-dir.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import numpy as np


COCO_KEYPOINTS = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

# COCO sigmas for OKS (order matches COCO_KEYPOINTS)
COCO_SIGMAS = np.array(
    [
        0.26,
        0.25,
        0.25,
        0.35,
        0.35,
        0.79,
        0.79,
        0.72,
        0.72,
        0.62,
        0.62,
        1.07,
        1.07,
        0.87,
        0.87,
        0.89,
        0.89,
    ],
    dtype=np.float32,
) / 10.0

# MediaPipe Pose Landmarker 33 indices mapped to COCO 17 indices
# https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
MP_TO_COCO = {
    0: 0,   # nose
    2: 1,   # left_eye
    5: 2,   # right_eye
    7: 3,   # left_ear
    8: 4,   # right_ear
    11: 5,  # left_shoulder
    12: 6,  # right_shoulder
    13: 7,  # left_elbow
    14: 8,  # right_elbow
    15: 9,  # left_wrist
    16: 10, # right_wrist
    23: 11, # left_hip
    24: 12, # right_hip
    25: 13, # left_knee
    26: 14, # right_knee
    27: 15, # left_ankle
    28: 16, # right_ankle
}


@dataclass
class PersonKeypoints:
    xy: np.ndarray  # (17, 2)
    conf: np.ndarray  # (17,)


@dataclass
class CocoAnnotation:
    image_id: int
    bbox: Tuple[float, float, float, float]
    keypoints: np.ndarray  # (17, 3)


def load_coco_annotations(json_path: str) -> Tuple[Dict[int, str], Dict[int, List[CocoAnnotation]]]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    images = {img["id"]: img["file_name"] for img in data.get("images", [])}
    annotations: Dict[int, List[CocoAnnotation]] = {}

    for ann in data.get("annotations", []):
        if ann.get("num_keypoints", 0) == 0:
            continue
        if "keypoints" not in ann or "bbox" not in ann:
            continue
        kp = np.array(ann["keypoints"], dtype=np.float32).reshape(-1, 3)
        if kp.shape[0] != 17:
            continue
        item = CocoAnnotation(
            image_id=ann["image_id"],
            bbox=tuple(ann["bbox"]),
            keypoints=kp,
        )
        annotations.setdefault(ann["image_id"], []).append(item)

    return images, annotations


def _pck(person_gt: CocoAnnotation, person_pred: PersonKeypoints, alpha: float) -> float:
    bbox_w = max(1.0, float(person_gt.bbox[2]))
    bbox_h = max(1.0, float(person_gt.bbox[3]))
    threshold = alpha * max(bbox_w, bbox_h)

    gt_xy = person_gt.keypoints[:, :2]
    gt_vis = person_gt.keypoints[:, 2] > 0

    pred_xy = person_pred.xy

    dists = np.linalg.norm(pred_xy - gt_xy, axis=1)
    correct = (dists < threshold) & gt_vis
    if gt_vis.sum() == 0:
        return 0.0
    return float(correct.sum() / gt_vis.sum())


def _oks(person_gt: CocoAnnotation, person_pred: PersonKeypoints) -> float:
    area = max(1.0, float(person_gt.bbox[2] * person_gt.bbox[3]))
    gt_xy = person_gt.keypoints[:, :2]
    gt_vis = person_gt.keypoints[:, 2] > 0
    pred_xy = person_pred.xy

    if gt_vis.sum() == 0:
        return 0.0

    d2 = np.sum((pred_xy - gt_xy) ** 2, axis=1)
    sigmas = COCO_SIGMAS
    vars_ = (sigmas * 2) ** 2
    e = d2 / (2 * vars_ * area + 1e-10)
    oks = np.exp(-e)
    return float(np.sum(oks[gt_vis]) / gt_vis.sum())


def _match_predictions(
    gts: List[CocoAnnotation],
    preds: List[PersonKeypoints],
    oks_threshold: float,
) -> List[Tuple[CocoAnnotation, PersonKeypoints | None]]:
    if not gts:
        return []
    if not preds:
        return [(gt, None) for gt in gts]

    oks_matrix = np.zeros((len(gts), len(preds)), dtype=np.float32)
    for i, gt in enumerate(gts):
        for j, pred in enumerate(preds):
            oks_matrix[i, j] = _oks(gt, pred)

    matched = []
    used_preds = set()
    for i, gt in enumerate(gts):
        best_j = int(np.argmax(oks_matrix[i]))
        best_oks = float(oks_matrix[i, best_j])
        if best_j in used_preds or best_oks < oks_threshold:
            matched.append((gt, None))
            continue
        used_preds.add(best_j)
        matched.append((gt, preds[best_j]))

    return matched


def run_yolov8(model, image_bgr: np.ndarray, device: str) -> List[PersonKeypoints]:
    results = model.predict(image_bgr, device=device, verbose=False)
    if not results:
        return []

    keypoints = results[0].keypoints
    if keypoints is None:
        return []

    xy = keypoints.xy.cpu().numpy()
    conf = keypoints.conf.cpu().numpy() if keypoints.conf is not None else None
    persons = []
    for idx in range(xy.shape[0]):
        kp_xy = xy[idx]
        kp_conf = conf[idx] if conf is not None else np.ones((kp_xy.shape[0],), dtype=np.float32)
        persons.append(PersonKeypoints(xy=kp_xy, conf=kp_conf))
    return persons


def run_mediapipe(landmarker, image_bgr: np.ndarray) -> List[PersonKeypoints]:
    import mediapipe as mp

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)

    persons: List[PersonKeypoints] = []
    for pose in result.pose_landmarks:
        xy = np.zeros((17, 2), dtype=np.float32)
        conf = np.zeros((17,), dtype=np.float32)
        for mp_idx, coco_idx in MP_TO_COCO.items():
            lm = pose[mp_idx]
            xy[coco_idx, 0] = lm.x * image_bgr.shape[1]
            xy[coco_idx, 1] = lm.y * image_bgr.shape[0]
            conf[coco_idx] = float(lm.visibility)
        persons.append(PersonKeypoints(xy=xy, conf=conf))
    return persons


def evaluate(
    dataset_json: str,
    images_dir: str,
    yolo_model: str,
    mp_model: str,
    device: str,
    max_images: int,
    pck_alpha: float,
    oks_threshold: float,
    out_report: str,
) -> None:
    images, annotations = load_coco_annotations(dataset_json)

    image_ids = list(annotations.keys())
    if max_images > 0:
        image_ids = image_ids[:max_images]

    yolo_pcks = []
    yolo_oks = []
    mp_pcks = []
    mp_oks = []

    from ultralytics import YOLO
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    yolo_model = YOLO(yolo_model)
    base = python.BaseOptions(model_asset_path=mp_model)
    mp_options = vision.PoseLandmarkerOptions(
        base_options=base,
        output_segmentation_masks=False,
        num_poses=6,
        running_mode=vision.RunningMode.IMAGE,
    )

    t0 = time.time()
    mp_landmarker = vision.PoseLandmarker.create_from_options(mp_options)
    for idx, image_id in enumerate(image_ids, start=1):
        file_name = images.get(image_id)
        if not file_name:
            continue
        img_path = os.path.join(images_dir, file_name)
        image = cv2.imread(img_path)
        if image is None:
            continue

        gts = annotations.get(image_id, [])

        yolo_preds = run_yolov8(yolo_model, image, device)
        mp_preds = run_mediapipe(mp_landmarker, image)

        for gt, pred in _match_predictions(gts, yolo_preds, oks_threshold):
            if pred is None:
                yolo_pcks.append(0.0)
                yolo_oks.append(0.0)
            else:
                yolo_pcks.append(_pck(gt, pred, pck_alpha))
                yolo_oks.append(_oks(gt, pred))

        for gt, pred in _match_predictions(gts, mp_preds, oks_threshold):
            if pred is None:
                mp_pcks.append(0.0)
                mp_oks.append(0.0)
            else:
                mp_pcks.append(_pck(gt, pred, pck_alpha))
                mp_oks.append(_oks(gt, pred))

        if idx % 10 == 0:
            print(f"Processed {idx}/{len(image_ids)} images...")

    mp_landmarker.close()
    elapsed = time.time() - t0

    def _mean(values: List[float]) -> float:
        return float(np.mean(values)) if values else 0.0

    summary = {
        "images_evaluated": len(image_ids),
        "time_elapsed_sec": round(elapsed, 4),
        "pck_alpha": pck_alpha,
        "oks_match_threshold": oks_threshold,
        "yolov8_pck": round(_mean(yolo_pcks), 6),
        "yolov8_oks": round(_mean(yolo_oks), 6),
        "mediapipe_pck": round(_mean(mp_pcks), 6),
        "mediapipe_oks": round(_mean(mp_oks), 6),
        "yolo_model": yolo_model,
        "mediapipe_model": mp_model,
        "device": device,
    }

    print("\nAccuracy Summary")
    print("=" * 70)
    print(f"Images evaluated: {summary['images_evaluated']}")
    print(f"Time elapsed: {summary['time_elapsed_sec']:.2f}s")
    print(f"YOLOv8 PCK@{pck_alpha:.2f}: {summary['yolov8_pck']:.4f}")
    print(f"YOLOv8 OKS: {summary['yolov8_oks']:.4f}")
    print(f"MediaPipe PCK@{pck_alpha:.2f}: {summary['mediapipe_pck']:.4f}")
    print(f"MediaPipe OKS: {summary['mediapipe_oks']:.4f}")

    if out_report:
        report = [
            "Accuracy Summary",
            "=" * 70,
            f"Images evaluated: {summary['images_evaluated']}",
            f"Time elapsed: {summary['time_elapsed_sec']:.2f}s",
            f"PCK alpha: {summary['pck_alpha']}",
            f"OKS match threshold: {summary['oks_match_threshold']}",
            f"YOLOv8 model: {summary['yolo_model']}",
            f"MediaPipe model: {summary['mediapipe_model']}",
            f"Device: {summary['device']}",
            f"YOLOv8 PCK@{pck_alpha:.2f}: {summary['yolov8_pck']:.4f}",
            f"YOLOv8 OKS: {summary['yolov8_oks']:.4f}",
            f"MediaPipe PCK@{pck_alpha:.2f}: {summary['mediapipe_pck']:.4f}",
            f"MediaPipe OKS: {summary['mediapipe_oks']:.4f}",
        ]
        with open(out_report, "w", encoding="utf-8") as f:
            f.write("\n".join(report) + "\n")
        print(f"Saved report: {out_report}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate pose accuracy (PCK/OKS).")
    parser.add_argument("--dataset-json", required=True, help="Path to COCO keypoints JSON.")
    parser.add_argument("--images-dir", required=True, help="Directory with dataset images.")
    parser.add_argument("--yolo-model", default="yolov8m-pose.pt", help="YOLOv8 pose model.")
    parser.add_argument(
        "--mediapipe-model",
        default="pose_landmarker_full.task",
        help="MediaPipe Pose Landmarker model.",
    )
    parser.add_argument("--device", default="cuda", help="YOLOv8 device: cuda or cpu.")
    parser.add_argument("--max-images", type=int, default=0, help="Limit images (0 = all).")
    parser.add_argument("--pck-alpha", type=float, default=0.2, help="PCK threshold fraction.")
    parser.add_argument("--oks-threshold", type=float, default=0.3, help="OKS match threshold.")
    parser.add_argument("--out-report", default="", help="Optional path to save a text summary.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(
        dataset_json=args.dataset_json,
        images_dir=args.images_dir,
        yolo_model=args.yolo_model,
        mp_model=args.mediapipe_model,
        device=args.device,
        max_images=args.max_images,
        pck_alpha=args.pck_alpha,
        oks_threshold=args.oks_threshold,
        out_report=args.out_report,
    )
