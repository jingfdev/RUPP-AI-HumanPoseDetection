
## 1. Work Complete
- Added accuracy report output support for PCK/OKS runs and a PyTorch vs ONNX Runtime benchmark script.
- Wrote a concise “how to run” guide covering accuracy, benchmarking, and OpenPose setup.
- Prepared the project to run end-to-end evaluations once the COCO val2017 dataset is available.

## 2. In Progress
- Awaiting COCO val2017 images and annotations to run accuracy evaluation and generate a final summary.
- Stabilizing OpenPose runtime on Windows with reliable DLL discovery and path configuration.

## 3. Plan for Next Week
- Run accuracy evaluation on COCO val2017 and summarize YOLOv8 vs MediaPipe results.
- Run PyTorch vs ONNX Runtime benchmarks with consistent inputs and report results.
- Test with real video capture to validate end-to-end throughput.
- Test with COCO val2017 images dataset for accuracy verification.
- Finalize OpenPose enablement steps and commit updates.

## 4. Challenge
- OpenPose Python bindings require a custom Windows build and correct DLL discovery.
- Keeping timing fair across CPU/GPU (sync points, warmup, provider availability) without VRAM bottlenecks.
- Keypoint format differences across YOLO/MediaPipe/OpenPose require careful mapping for accuracy metrics.
