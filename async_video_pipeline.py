"""Asynchronous video pipeline: capture and inference in separate threads."""
from __future__ import annotations

import argparse
import os
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np
import pandas as pd


# COCO keypoint indices used by YOLO pose models.
NOSE = 0
LEFT_EYE = 1
RIGHT_EYE = 2
LEFT_EAR = 3
RIGHT_EAR = 4
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_ELBOW = 7
RIGHT_ELBOW = 8
LEFT_WRIST = 9
RIGHT_WRIST = 10
LEFT_HIP = 11
RIGHT_HIP = 12


@dataclass
class PoseLabel:
    label: str
    score: float
    box: Tuple[int, int, int, int]
    color: Tuple[int, int, int]
    reason: str


class SleepPoseClassifier:
    """Rule-based Normal/Sleeping classifier built from YOLO pose keypoints."""

    def __init__(
        self,
        sleep_threshold: float,
        persist_seconds: float,
        min_box_area_ratio: float,
        min_keypoints: int,
    ) -> None:
        self.sleep_threshold = sleep_threshold
        self.persist_seconds = persist_seconds
        self.min_box_area_ratio = min_box_area_ratio
        self.min_keypoints = min_keypoints
        self._track_sleep_started_at: Dict[int, float] = {}
        self._track_labels: Dict[int, str] = {}

    def classify(self, result, now: float, frame_shape: Tuple[int, int, int]) -> List[PoseLabel]:
        if result.boxes is None or result.keypoints is None:
            return []

        boxes = result.boxes.xyxy.cpu().numpy()
        keypoints = result.keypoints.xy.cpu().numpy()
        confidences = self._keypoint_confidences(result)
        labels: List[PoseLabel] = []
        frame_area = float(frame_shape[0] * frame_shape[1])

        valid_indices = self._valid_person_indices(boxes, confidences, frame_area)

        for person_index in valid_indices:
            box = boxes[person_index]
            if person_index >= len(keypoints):
                continue

            score, reason = self._sleep_score(keypoints[person_index], confidences[person_index], box)
            track_id = self._track_id(box, person_index)

            previous_label = self._track_labels.get(track_id, "Normal")
            if score >= self.sleep_threshold:
                self._track_sleep_started_at.setdefault(track_id, now)
                sleeping_long_enough = now - self._track_sleep_started_at[track_id] >= self.persist_seconds
                label = "Sleeping" if sleeping_long_enough or previous_label == "Sleeping" else "Normal"
            else:
                self._track_sleep_started_at.pop(track_id, None)
                label = "Normal"

            self._track_labels[track_id] = label
            color = (0, 0, 255) if label == "Sleeping" else (0, 255, 0)
            x1, y1, x2, y2 = [int(v) for v in box]
            labels.append(PoseLabel(label, score, (x1, y1, x2, y2), color, reason))

        return labels

    def _keypoint_confidences(self, result) -> np.ndarray:
        if result.keypoints.conf is not None:
            return result.keypoints.conf.cpu().numpy()
        keypoints = result.keypoints.xy.cpu().numpy()
        return np.ones((keypoints.shape[0], keypoints.shape[1]), dtype=np.float32)

    def _track_id(self, box: np.ndarray, fallback: int) -> int:
        x1, _, x2, _ = box
        center_x = int((x1 + x2) / 2.0)
        return int(center_x / 80) if center_x >= 0 else fallback

    def _valid_person_indices(
        self,
        boxes: np.ndarray,
        confidences: np.ndarray,
        frame_area: float,
    ) -> List[int]:
        valid_indices: List[int] = []
        for person_index, box in enumerate(boxes):
            if self._box_area_ratio(box, frame_area) < self.min_box_area_ratio:
                continue
            if self._visible_keypoint_count(confidences[person_index]) < self.min_keypoints:
                continue
            valid_indices.append(person_index)
        return valid_indices

    def _box_area_ratio(self, box: np.ndarray, frame_area: float) -> float:
        x1, y1, x2, y2 = box
        box_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        return box_area / max(1.0, frame_area)

    def _visible_keypoint_count(self, conf: np.ndarray) -> int:
        return int(np.count_nonzero(conf >= 0.25))

    def _sleep_score(self, kp: np.ndarray, conf: np.ndarray, box: np.ndarray) -> Tuple[float, str]:
        x1, y1, x2, y2 = box
        box_w = max(1.0, x2 - x1)
        box_h = max(1.0, y2 - y1)

        head = self._mean_point(kp, conf, [NOSE, LEFT_EYE, RIGHT_EYE, LEFT_EAR, RIGHT_EAR])
        shoulders = self._mean_point(kp, conf, [LEFT_SHOULDER, RIGHT_SHOULDER])
        hips = self._mean_point(kp, conf, [LEFT_HIP, RIGHT_HIP])
        wrists = self._mean_point(kp, conf, [LEFT_WRIST, RIGHT_WRIST])
        elbows = self._mean_point(kp, conf, [LEFT_ELBOW, RIGHT_ELBOW])

        score = 0.0
        reasons: List[str] = []

        if head is not None and shoulders is not None:
            head_low_ratio = (head[1] - shoulders[1]) / box_h
            if head_low_ratio > -0.06:
                score += 0.25
                reasons.append("head low")

        if shoulders is not None and hips is not None:
            torso_vec = hips - shoulders
            torso_angle = abs(float(np.degrees(np.arctan2(torso_vec[1], torso_vec[0]))))
            torso_vertical_span = abs(float(hips[1] - shoulders[1])) / box_h

            if torso_angle < 35.0 or torso_angle > 145.0:
                score += 0.30
                reasons.append("horizontal torso")
            if torso_vertical_span < 0.30:
                score += 0.15
                reasons.append("folded body")

        if head is not None and wrists is not None:
            head_to_wrists = float(np.linalg.norm(head - wrists)) / max(box_w, box_h)
            if head_to_wrists < 0.36:
                score += 0.18
                reasons.append("head near hands")

        if head is not None and elbows is not None:
            head_to_elbows = float(np.linalg.norm(head - elbows)) / max(box_w, box_h)
            if head_to_elbows < 0.34:
                score += 0.12
                reasons.append("head near arms")

        if box_w / box_h > 0.85:
            score += 0.15
            reasons.append("wide posture")

        return min(score, 1.0), ", ".join(reasons) or "upright posture"

    def _mean_point(self, kp: np.ndarray, conf: np.ndarray, indices: Iterable[int]) -> np.ndarray | None:
        points = []
        for index in indices:
            if index < len(kp) and conf[index] >= 0.25 and kp[index][0] > 0 and kp[index][1] > 0:
                points.append(kp[index])
        if not points:
            return None
        return np.mean(np.asarray(points, dtype=np.float32), axis=0)


class TrainedPostureClassifier(SleepPoseClassifier):
    """Normal/Sleeping classifier trained from YOLO pose geometry features."""

    def __init__(
        self,
        model_path: str,
        sleep_threshold: float,
        persist_seconds: float,
        min_box_area_ratio: float,
        min_keypoints: int,
    ) -> None:
        super().__init__(sleep_threshold, persist_seconds, min_box_area_ratio, min_keypoints)
        import joblib

        artifact = joblib.load(model_path)
        self.model = artifact["model"]
        self.feature_columns = artifact["feature_columns"]
        self.labels = artifact.get("labels", ["normal", "sleeping"])
        self.model_path = model_path

    def classify(self, result, now: float, frame_shape: Tuple[int, int, int]) -> List[PoseLabel]:
        if result.boxes is None or result.keypoints is None:
            return []

        boxes = result.boxes.xyxy.cpu().numpy()
        keypoints = result.keypoints.xy.cpu().numpy()
        confidences = self._keypoint_confidences(result)
        labels: List[PoseLabel] = []
        frame_area = float(frame_shape[0] * frame_shape[1])

        valid_indices = self._valid_person_indices(boxes, confidences, frame_area)
        for person_index in valid_indices:
            box = boxes[person_index]
            if person_index >= len(keypoints):
                continue

            features = self._feature_dict(keypoints[person_index], confidences[person_index], box, frame_area)
            vector = pd.DataFrame([[features[column] for column in self.feature_columns]], columns=self.feature_columns)
            predicted = str(self.model.predict(vector)[0])
            sleeping_score = self._sleeping_probability(vector, predicted)
            track_id = self._track_id(box, person_index)

            previous_label = self._track_labels.get(track_id, "Normal")
            sleeping_candidate = predicted == "sleeping" and sleeping_score >= self.sleep_threshold
            if sleeping_candidate:
                self._track_sleep_started_at.setdefault(track_id, now)
                sleeping_long_enough = now - self._track_sleep_started_at[track_id] >= self.persist_seconds
                label = "Sleeping" if sleeping_long_enough or previous_label == "Sleeping" else "Normal"
            else:
                self._track_sleep_started_at.pop(track_id, None)
                label = "Normal"

            self._track_labels[track_id] = label
            color = (0, 0, 255) if label == "Sleeping" else (0, 255, 0)
            x1, y1, x2, y2 = [int(v) for v in box]
            labels.append(PoseLabel(label, sleeping_score, (x1, y1, x2, y2), color, "trained classifier"))

        return labels

    def _sleeping_probability(self, vector: pd.DataFrame, predicted: str) -> float:
        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(vector)[0]
            classes = [str(value) for value in self.model.classes_]
            if "sleeping" in classes:
                return float(probabilities[classes.index("sleeping")])
        return 1.0 if predicted == "sleeping" else 0.0

    def _feature_dict(
        self,
        kp: np.ndarray,
        conf: np.ndarray,
        box: np.ndarray,
        frame_area: float,
    ) -> Dict[str, float]:
        x1, y1, x2, y2 = box
        box_w = max(1.0, float(x2 - x1))
        box_h = max(1.0, float(y2 - y1))
        box_area = box_w * box_h

        head = self._mean_point(kp, conf, [NOSE, LEFT_EYE, RIGHT_EYE])
        shoulders = self._mean_point(kp, conf, [LEFT_SHOULDER, RIGHT_SHOULDER])
        hips = self._mean_point(kp, conf, [LEFT_HIP, RIGHT_HIP])
        wrists = self._mean_point(kp, conf, [LEFT_WRIST, RIGHT_WRIST])
        elbows = self._mean_point(kp, conf, [LEFT_ELBOW, RIGHT_ELBOW])

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
            "head_to_wrists": self._normalized_distance(head, wrists, max(box_w, box_h)),
            "head_to_elbows": self._normalized_distance(head, elbows, max(box_w, box_h)),
            "shoulder_slope": self._pair_slope(kp, conf, LEFT_SHOULDER, RIGHT_SHOULDER, box_h),
            "hip_slope": self._pair_slope(kp, conf, LEFT_HIP, RIGHT_HIP, box_h),
            "shoulder_width_ratio": self._pair_width(kp, conf, LEFT_SHOULDER, RIGHT_SHOULDER, box_w),
            "hip_width_ratio": self._pair_width(kp, conf, LEFT_HIP, RIGHT_HIP, box_w),
            "head_to_hips_y": head_to_hips_y,
        }

    def _normalized_distance(self, a: np.ndarray | None, b: np.ndarray | None, scale: float) -> float:
        if a is None or b is None:
            return -1.0
        return float(np.linalg.norm(a - b) / max(1.0, scale))

    def _pair_slope(self, kp: np.ndarray, conf: np.ndarray, left: int, right: int, box_h: float) -> float:
        if conf[left] < 0.25 or conf[right] < 0.25:
            return -1.0
        return float((kp[right][1] - kp[left][1]) / max(1.0, box_h))

    def _pair_width(self, kp: np.ndarray, conf: np.ndarray, left: int, right: int, box_w: float) -> float:
        if conf[left] < 0.25 or conf[right] < 0.25:
            return -1.0
        return float(abs(kp[right][0] - kp[left][0]) / max(1.0, box_w))


def draw_pose_labels(
    frame: np.ndarray,
    labels: List[PoseLabel],
    show_score: bool,
    show_reason: bool,
) -> np.ndarray:
    """Draw one label and bounding box for every classified person."""
    if not labels:
        return frame

    for pose_label in labels:
        x1, y1, x2, y2 = pose_label.box
        label_bg = (255, 0, 255)
        label_text = pose_label.label
        if show_score:
            label_text = f"{pose_label.label} {pose_label.score:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), pose_label.color, 4)

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.8
        thickness = 2
        pad_x = 10
        pad_y = 8
        (text_w, text_h), baseline = cv2.getTextSize(label_text, font, scale, thickness)

        label_x1 = max(0, x1)
        label_y2 = max(text_h + baseline + pad_y * 2, y1)
        label_x2 = min(frame.shape[1] - 1, label_x1 + text_w + pad_x * 2)
        label_y1 = max(0, label_y2 - text_h - baseline - pad_y * 2)

        cv2.rectangle(frame, (label_x1, label_y1), (label_x2, label_y2), label_bg, -1)
        cv2.putText(
            frame,
            label_text,
            (label_x1 + pad_x, label_y2 - baseline - pad_y),
            font,
            scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
        if show_reason:
            cv2.putText(
                frame,
                pose_label.reason,
                (label_x1, min(frame.shape[0] - 8, label_y2 + 22)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                pose_label.color,
                1,
                cv2.LINE_AA,
            )

    return frame


def require_cuda_device(requested: str) -> str:
    """Return a CUDA device string or fail fast; CPU fallback is not allowed."""
    normalized = requested.strip().lower()
    if not normalized.startswith("cuda"):
        raise RuntimeError(
            f"GPU-only mode is enabled, but device '{requested}' was requested. "
            "Use --device cuda:0 or another CUDA device."
        )

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required to verify CUDA GPU availability.") from exc

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is not available. This application is configured to run on GPU only, "
            "so it will not fall back to CPU."
        )

    if ":" in normalized:
        index_text = normalized.split(":", 1)[1]
        if index_text:
            try:
                device_index = int(index_text)
            except ValueError as exc:
                raise RuntimeError(f"Invalid CUDA device '{requested}'. Expected format like cuda:0.") from exc
            if device_index < 0 or device_index >= torch.cuda.device_count():
                raise RuntimeError(
                    f"CUDA device '{requested}' is not available. "
                    f"Detected {torch.cuda.device_count()} CUDA device(s)."
                )

    return requested


def run_async(
    source: str,
    model_path: str,
    device: str,
    conf: float,
    classifier_mode: str,
    posture_classifier_path: str,
    sleep_threshold: float,
    sleep_persist: float,
    min_box_area_ratio: float,
    min_keypoints: int,
    show_score: bool,
    show_reason: bool,
    window_scale: float,
    loop_video: bool,
    playback_fps: float,
) -> None:
    from ultralytics import YOLO

    print(
        f"[INFO] Starting async pose demo | source={source} | model={model_path} | "
        f"device={device} | conf={conf}"
    )
    is_webcam = source.isdigit()
    cap = cv2.VideoCapture(0 if is_webcam else source)
    if not cap.isOpened():
        raise RuntimeError("Failed to open video source.")
    video_frame_delay = 1.0 / max(1.0, playback_fps) if not is_webcam else 0.0

    model = YOLO(model_path)
    if classifier_mode == "trained":
        classifier_file = Path(posture_classifier_path)
        if not classifier_file.exists():
            raise RuntimeError(
                f"Trained posture classifier not found: {posture_classifier_path}. "
                "Use --classifier-mode rule or train the classifier first."
            )
        pose_classifier = TrainedPostureClassifier(
            model_path=str(classifier_file),
            sleep_threshold=sleep_threshold,
            persist_seconds=sleep_persist,
            min_box_area_ratio=min_box_area_ratio,
            min_keypoints=min_keypoints,
        )
        print(f"[INFO] Loaded trained posture classifier: {classifier_file}")
    else:
        pose_classifier = SleepPoseClassifier(
            sleep_threshold=sleep_threshold,
            persist_seconds=sleep_persist,
            min_box_area_ratio=min_box_area_ratio,
            min_keypoints=min_keypoints,
        )
        print("[INFO] Using rule-based posture classifier")
    frame_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=1)
    result_queue: queue.Queue[Tuple[np.ndarray, float]] = queue.Queue(maxsize=1)
    stop_event = threading.Event()
    capture_done = threading.Event()

    infer_done = threading.Event()

    def capture_loop() -> None:
        while not stop_event.is_set():
            ok, frame = cap.read()
            if not ok:
                if loop_video and not is_webcam:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                capture_done.set()
                break
            if frame_queue.full():
                try:
                    frame_queue.get_nowait()
                except queue.Empty:
                    pass
            frame_queue.put(frame)
            if video_frame_delay > 0:
                time.sleep(video_frame_delay)

    def infer_loop() -> None:
        while not stop_event.is_set():
            if capture_done.is_set() and frame_queue.empty():
                break
            try:
                frame = frame_queue.get(timeout=0.1)
            except queue.Empty:
                if capture_done.is_set():
                    break
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
                pose_labels = pose_classifier.classify(results[0], time.time(), annotated_frame.shape)
                annotated_frame = draw_pose_labels(annotated_frame, pose_labels, show_score, show_reason)

            if result_queue.full():
                try:
                    result_queue.get_nowait()
                except queue.Empty:
                    pass
            result_queue.put((annotated_frame, inf_ms))
        infer_done.set()

    capture_thread = threading.Thread(target=capture_loop, daemon=True)
    infer_thread = threading.Thread(target=infer_loop, daemon=True)
    capture_thread.start()
    infer_thread.start()
    window_name = "YOLO Pose Normal/Sleeping Pipeline"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    fps = 0.0
    last = time.time()
    frame_count = 0
    while not stop_event.is_set():
        try:
            frame, inf_ms = result_queue.get(timeout=0.1)
        except queue.Empty:
            if capture_done.is_set() and infer_done.is_set():
                break
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
        if frame_count == 1:
            height, width = frame.shape[:2]
            cv2.resizeWindow(
                window_name,
                max(640, int(width * window_scale)),
                max(480, int(height * window_scale)),
            )
        cv2.imshow(window_name, frame)
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
    parser.add_argument("--model", default="yolo11s-pose.pt")
    parser.add_argument("--device", default="cuda:0", help="CUDA device only, for example cuda or cuda:0")
    parser.add_argument("--conf", type=float, default=0.35, help="Detection confidence threshold.")
    parser.add_argument(
        "--classifier-mode",
        choices=["trained", "rule"],
        default="trained",
        help="Use the trained Normal/Sleeping classifier or the older rule-based classifier.",
    )
    parser.add_argument(
        "--posture-classifier",
        default="posture_models/yolo11s_balanced_drop_confusing/best_posture_classifier.joblib",
        help="Path to the trained posture classifier artifact.",
    )
    parser.add_argument(
        "--sleep-threshold",
        type=float,
        default=0.55,
        help="Pose score threshold for considering a person sleeping.",
    )
    parser.add_argument(
        "--sleep-persist",
        type=float,
        default=1.0,
        help="Seconds the sleeping posture must persist before labeling Sleeping.",
    )
    parser.add_argument(
        "--min-box-area-ratio",
        type=float,
        default=0.02,
        help="Ignore detections smaller than this fraction of the frame area.",
    )
    parser.add_argument(
        "--min-keypoints",
        type=int,
        default=5,
        help="Ignore detections with fewer visible pose keypoints.",
    )
    parser.add_argument(
        "--show-score",
        action="store_true",
        help="Show the rule-based sleeping score next to each label.",
    )
    parser.add_argument(
        "--show-reason",
        action="store_true",
        help="Show which posture rules contributed to the sleeping score.",
    )
    parser.add_argument(
        "--window-scale",
        type=float,
        default=1.25,
        help="Scale factor for the popup display window.",
    )
    parser.add_argument(
        "--loop-video",
        action="store_true",
        help="Loop video file sources instead of closing when the file ends.",
    )
    parser.add_argument(
        "--playback-fps",
        type=float,
        default=15.0,
        help="Display pacing for video file sources. Webcam sources ignore this.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    selected_device = require_cuda_device(args.device)
    run_async(
        args.source,
        args.model,
        selected_device,
        args.conf,
        args.classifier_mode,
        args.posture_classifier,
        args.sleep_threshold,
        args.sleep_persist,
        args.min_box_area_ratio,
        args.min_keypoints,
        args.show_score,
        args.show_reason,
        args.window_scale,
        args.loop_video,
        args.playback_fps,
    )

