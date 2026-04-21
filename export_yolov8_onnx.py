"""Export YOLOv8 pose model to ONNX."""
from __future__ import annotations

import argparse
import os


def export_onnx(model_path: str, imgsz: int, opset: int) -> str:
    from ultralytics import YOLO

    model = YOLO(model_path)
    result = model.export(format="onnx", imgsz=imgsz, opset=opset)
    return str(result)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export YOLOv8 pose to ONNX.")
    parser.add_argument("--model", default="yolov8m-pose.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    onnx_path = os.path.splitext(args.model)[0] + ".onnx"
    if os.path.exists(onnx_path) and not args.force:
        size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
        print(f"ONNX already exists: {onnx_path} ({size_mb:.1f} MB)")
    else:
        print("Exporting ONNX...")
        out = export_onnx(args.model, args.imgsz, args.opset)
        print(f"Export complete: {out}")
