"""Collect labeled posture frames from a webcam or video source.

This creates the project dataset for the trained Normal/Sleeping classifier.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2


VALID_LABELS = {"normal", "sleeping", "confusing"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect labeled posture frames.")
    parser.add_argument("--source", default="0", help="Webcam index or video path.")
    parser.add_argument("--label", required=True, choices=sorted(VALID_LABELS), help="Label for captured frames.")
    parser.add_argument("--output-dir", default="posture_dataset/raw", help="Dataset output directory.")
    parser.add_argument("--interval", type=float, default=0.5, help="Seconds between auto-saved frames.")
    parser.add_argument("--max-frames", type=int, default=100, help="Stop after this many saved frames.")
    parser.add_argument("--manual", action="store_true", help="Press SPACE to save each frame instead of auto-saving.")
    parser.add_argument("--prefix", default="", help="Optional filename prefix.")
    return parser.parse_args()


def next_index(label_dir: Path, prefix: str, label: str) -> int:
    stem = f"{prefix}_{label}" if prefix else label
    existing = sorted(label_dir.glob(f"{stem}_*.jpg"))
    if not existing:
        return 1
    numbers = []
    for path in existing:
        try:
            numbers.append(int(path.stem.rsplit("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    return max(numbers, default=0) + 1


def main() -> int:
    args = parse_args()
    label_dir = Path(args.output_dir) / args.label
    label_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(0 if args.source.isdigit() else args.source)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open source: {args.source}")

    index = next_index(label_dir, args.prefix, args.label)
    saved = 0
    last_save = 0.0
    stem = f"{args.prefix}_{args.label}" if args.prefix else args.label
    window_name = f"Collect Posture Dataset - {args.label}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("[INFO] Controls: SPACE=save in manual mode, q/ESC=quit")
    print(f"[INFO] Saving to: {label_dir.resolve()}")

    while saved < args.max_frames:
        ok, frame = cap.read()
        if not ok:
            break

        preview = frame.copy()
        cv2.putText(
            preview,
            f"label={args.label} saved={saved}/{args.max_frames}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow(window_name, preview)

        key = cv2.waitKey(1) & 0xFF
        if key in {ord("q"), 27}:
            break

        should_save = False
        now = time.time()
        if args.manual:
            should_save = key == 32
        elif now - last_save >= args.interval:
            should_save = True

        if should_save:
            output_path = label_dir / f"{stem}_{index:04d}.jpg"
            cv2.imwrite(str(output_path), frame)
            saved += 1
            index += 1
            last_save = now
            print(f"[SAVE] {output_path}")

    cap.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Finished. Saved {saved} frame(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
