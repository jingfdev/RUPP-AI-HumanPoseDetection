"""Import a Roboflow YOLO-format dataset into the posture_dataset/raw folders.

This helper copies images into:

- posture_dataset/raw/normal
- posture_dataset/raw/sleeping
- posture_dataset/raw/confusing

It uses YOLO label files plus data.yaml class names when available.
"""
from __future__ import annotations

import argparse
import shutil
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_SLEEPING_CLASSES = {"sleep", "sleeping", "Sleep", "Sleeping"}
DEFAULT_NORMAL_CLASSES = {
    "normal",
    "not sleeping",
    "not_sleeping",
    "working",
    "upright",
    "raise_head",
    "standing",
    "sit",
    "sitting",
}
DEFAULT_CONFUSING_CLASSES = {
    "phone",
    "using_phone",
    "Using_phone",
    "book",
    "reading",
    "read",
    "writing",
    "write",
    "bend",
    "bow_head",
    "using_laptop",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import a Roboflow YOLO dataset into posture_dataset/raw.")
    parser.add_argument("--source-dir", required=True, help="Extracted Roboflow dataset folder.")
    parser.add_argument("--output-dir", default="posture_dataset/raw", help="Output raw posture dataset folder.")
    parser.add_argument("--prefix", default="public", help="Filename prefix for copied images.")
    parser.add_argument("--copy-unlabeled-as", default="", choices=["", "normal", "sleeping", "confusing"], help="Optional label for images with no label file.")
    return parser.parse_args()


def parse_data_yaml(path: Path) -> Dict[int, str]:
    if not path.exists():
        return {}

    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    names: Dict[int, str] = {}
    in_names_block = False
    inline_names = ""

    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("names:"):
            value = line.split(":", 1)[1].strip()
            if value.startswith("[") and value.endswith("]"):
                inline_names = value.strip("[]")
                break
            in_names_block = True
            continue
        if in_names_block:
            if ":" in line and not line.startswith("-"):
                key, value = line.split(":", 1)
                if key.strip().isdigit():
                    names[int(key.strip())] = value.strip().strip("'\"")
                else:
                    in_names_block = False
            elif line.startswith("-"):
                names[len(names)] = line[1:].strip().strip("'\"")

    if inline_names:
        for index, name in enumerate(inline_names.split(",")):
            names[index] = name.strip().strip("'\"")

    return names


def image_files(source_dir: Path) -> Iterable[Path]:
    for path in sorted(source_dir.rglob("*")):
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def label_file_for_image(image_path: Path) -> Path:
    parts = list(image_path.parts)
    if "images" in parts:
        index = parts.index("images")
        parts[index] = "labels"
        return Path(*parts).with_suffix(".txt")
    return image_path.with_suffix(".txt")


def class_ids_from_label(label_path: Path) -> List[int]:
    if not label_path.exists():
        return []
    class_ids: List[int] = []
    for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        fields = line.strip().split()
        if not fields:
            continue
        try:
            class_ids.append(int(float(fields[0])))
        except ValueError:
            continue
    return class_ids


def target_label(class_names: List[str], copy_unlabeled_as: str) -> str | None:
    normalized = {name.strip() for name in class_names}
    lower = {name.lower() for name in normalized}

    if lower & {name.lower() for name in DEFAULT_SLEEPING_CLASSES}:
        return "sleeping"
    if lower & {name.lower() for name in DEFAULT_CONFUSING_CLASSES}:
        return "confusing"
    if lower & {name.lower() for name in DEFAULT_NORMAL_CLASSES}:
        return "normal"
    return copy_unlabeled_as or None


def main() -> int:
    args = parse_args()
    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    names = parse_data_yaml(source_dir / "data.yaml")
    if not names:
        print("[WARN] No data.yaml class names found. Only --copy-unlabeled-as can import unlabeled images.")

    for label in ["normal", "sleeping", "confusing"]:
        (output_dir / label).mkdir(parents=True, exist_ok=True)

    counts: Counter[str] = Counter()
    skipped = 0

    for image_path in image_files(source_dir):
        class_ids = class_ids_from_label(label_file_for_image(image_path))
        class_names = [names[class_id] for class_id in class_ids if class_id in names]
        label = target_label(class_names, args.copy_unlabeled_as)
        if label is None:
            skipped += 1
            continue

        counts[label] += 1
        output_path = output_dir / label / f"{args.prefix}_{label}_{counts[label]:05d}{image_path.suffix.lower()}"
        shutil.copy2(image_path, output_path)

    print("[INFO] Imported images:")
    for label in ["normal", "sleeping", "confusing"]:
        print(f"  {label}: {counts[label]}")
    print(f"[INFO] Skipped: {skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
