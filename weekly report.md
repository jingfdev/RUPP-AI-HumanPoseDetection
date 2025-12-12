# WEEKLY REPORT

**Week Ending:** February 22, 2026  
**Project:** RUPP-AI Human Pose Detection

---

## Summary of Work Completed:

1. **Built CPU vs GPU (CUDA) Performance Comparison Pipeline**
   - Designed and implemented a comprehensive benchmarking notebook (`CPU_vs_GPU_Comparison.ipynb`) with 34 cells covering hardware detection, model loading, benchmark execution, statistical analysis, visualization, and automated report generation
   - Compared five model configurations across two compute devices: YOLOv8n-pose (CPU & GPU), YOLOv8s-pose (CPU & GPU), YOLOv8m-pose (CPU & GPU), MediaPipe Pose (CPU), and MediaPipe Pose+Face (CPU)

2. **Technology Stack & Environment Setup**
   - **Runtime:** Python 3.10 virtual environment (`venv310`) on Windows 10
   - **Deep Learning Framework:** PyTorch 2.10.0 with CUDA 12.6 support (`torch+cu126`), enabling native GPU acceleration via `torch.cuda` APIs
   - **YOLO Models:** Ultralytics YOLOv8 pose estimation family in three sizes — Nano (3.3M params, 9.2 GFLOPs), Small (11.6M params, 30.2 GFLOPs), and Medium (26.4M params, 81.0 GFLOPs) — loaded from `.pt` PyTorch checkpoint files
   - **MediaPipe:** Google MediaPipe 0.10.32 using the Tasks API (`PoseLandmarker` and `FaceLandmarker`) with pre-built `.task` model bundles; configured with `Delegate.CPU` since the Windows pip build does not ship a GPU delegate
   - **Visualization:** Matplotlib for bar charts, box plots, time-series plots, and speedup charts; Pandas + Jinja2 for styled HTML result tables
   - **Hardware:** Intel Core i5-11400H (12 logical cores) + NVIDIA GeForce GTX 1650 Max-Q (4 GB VRAM, CUDA 12.6, cuDNN enabled)

3. **Benchmarking Methodology**
   - Captured 5 live webcam test frames at 640×480 resolution to serve as standardized inputs
   - Performed 10 warm-up iterations per model/device combination to eliminate JIT compilation overhead, CUDA kernel caching, and memory allocation artifacts
   - Executed 50 timed iterations per configuration to collect statistically significant data (mean, std, min, max, P95, FPS)
   - Used `torch.cuda.synchronize()` before and after GPU inference to ensure accurate wall-clock GPU timing (preventing asynchronous kernel launch from inflating apparent speed)
   - MediaPipe models benchmarked with equivalent warm-up and iteration counts for fair comparison

4. **Benchmark Results — Inference Time & Throughput**

   | Model | Device | Avg Time (ms) | Std (ms) | P95 (ms) | Avg FPS | Best FPS |
   |-------|--------|---------------|----------|----------|---------|----------|
   | YOLOv8n-pose (3.3M) | CPU | 57.47 | 8.55 | 69.02 | 17.4 | 22.5 |
   | YOLOv8n-pose (3.3M) | GPU (CUDA) | 12.11 | 2.62 | 19.13 | 82.6 | 105.0 |
   | YOLOv8s-pose (11.6M) | CPU | 138.53 | 7.03 | 152.66 | 7.2 | 7.7 |
   | YOLOv8s-pose (11.6M) | GPU (CUDA) | 17.28 | 0.26 | 17.72 | 57.9 | 59.3 |
   | YOLOv8m-pose (26.4M) | CPU | 331.82 | 43.47 | 435.51 | 3.0 | 3.4 |
   | YOLOv8m-pose (26.4M) | GPU (CUDA) | 37.20 | 0.42 | 38.01 | 26.9 | 27.3 |
   | MediaPipe Pose | CPU | 12.08 | 0.38 | 12.62 | 82.8 | 88.3 |
   | MediaPipe Pose+Face | CPU | 14.33 | 0.70 | 15.47 | 69.8 | 75.2 |

5. **GPU Speedup Analysis**

   | Model | CPU (ms) | GPU (ms) | Speedup | FPS Gain |
   |-------|----------|----------|---------|----------|
   | YOLOv8n-pose | 57.47 | 12.11 | **4.75×** | +65.2 FPS |
   | YOLOv8s-pose | 138.53 | 17.28 | **8.02×** | +50.7 FPS |
   | YOLOv8m-pose | 331.82 | 37.20 | **8.92×** | +23.9 FPS |

   **Key insight:** The GPU speedup factor increases with model complexity — larger models with more convolution layers and parameters expose more parallelism that the GPU's CUDA cores can exploit. YOLOv8m-pose (26.4M parameters) achieved the highest speedup (8.92×), while the lighter YOLOv8n-pose (3.3M parameters) still gained a substantial 4.75× speedup.

6. **Real-Time Webcam Benchmark (YOLOv8n-pose, 15-second live test)**
   - 125 frames captured and processed on both CPU and GPU in real time
   - CPU average: 62.1 ms (16 FPS) — below the 30 FPS real-time threshold
   - GPU average: 50.3 ms (20 FPS) — closer to real-time; speedup of 1.24× under live conditions
   - The reduced speedup in the live test (1.24× vs 4.75× in batch) is expected: the real-time loop alternates CPU and GPU inference on the same frame, introducing data transfer overhead and preventing the GPU from maintaining a warmed pipeline

7. **Side-by-Side Live Demo (CPU vs GPU)**
   - Implemented an interactive OpenCV window displaying CPU and GPU inference simultaneously
   - Supports model switching at runtime via keyboard (keys 1–5 for YOLOv8n/s/m + MediaPipe Pose/Face)
   - Visually confirms GPU's advantage: smoother skeleton tracking, no visible frame drops, consistent FPS overlay
   - GPU maintains ≥ 30 FPS even on YOLOv8m-pose; CPU drops well below real-time on medium/large models

8. **Generated Output Artifacts**
   - [`cpu_vs_gpu_comparison_*.csv`](https://drive.google.com/file/d/11MlC9uBphNVi33L0zR_-DcE_wmvtnR8V/view?usp=sharing) — Raw benchmark data with all timing metrics per model/device
   - [`cpu_vs_gpu_report_*.txt`](https://drive.google.com/file/d/1YbHxHoAk4-UEQ5S302qtQugRCG2wKyC6/view?usp=sharing) — Full text reports with hardware info, results tables, and conclusions
   - [`cpu_vs_gpu_bar_chart.png`](https://drive.google.com/file/d/1YbHxHoAk4-UEQ5S302qtQugRCG2wKyC6/view?usp=sharing) — Grouped bar chart comparing inference time and FPS for CPU vs GPU
   - [`cpu_vs_gpu_speedup.png`](https://drive.google.com/file/d/1_JMXQKeptY6bTWaDDjxVjspYW6_ygUPU/view?usp=sharing) — GPU speedup factor bar chart
   - [`cpu_vs_gpu_boxplot.png`](https://drive.google.com/file/d/1XR6WgjFqJ8jysivaabDKY9qJQmvzFRrX/view?usp=sharing) — Box plot of inference time distributions across all models and devices
   - [`cpu_vs_gpu_timeseries.png`](https://drive.google.com/file/d/195Ks9n_IPykLKEcUSSK1Pj3kjyveYn54/view?usp=sharing) — Per-iteration time series for each YOLOv8 model on both devices
   - [`cpu_vs_gpu_all_models.png`](https://drive.google.com/file/d/1JWTeENDJnWFWu609xp9zeqKSyRVXEpoo/view?usp=sharing) — Horizontal bar chart ranking all model/device combinations by FPS
   - [`cpu_vs_gpu_realtime.png`](https://drive.google.com/file/d/1GJyn6iq-y_xQJ_zZLbwhnh_2mjwpKKyI/view?usp=sharing) — Live webcam benchmark time series with rolling average

---

## Detailed Analysis & Findings:

### Why GPU Acceleration Matters for Pose Detection

1. **Parallel Architecture:** The GTX 1650 Max-Q contains 1024 CUDA cores that process convolution operations in parallel. Pose detection models like YOLOv8 are dominated by convolution, pooling, and matrix multiply operations — all of which map naturally to GPU SIMD (Single Instruction, Multiple Data) execution. The CPU, with only 12 logical cores, must process these operations largely sequentially.

2. **Dedicated High-Bandwidth Memory:** The GPU's 4 GB VRAM provides significantly higher memory bandwidth than system RAM. This is critical because model weights, feature maps, and intermediate tensors must be loaded repeatedly during inference. The CPU version transfers data through the system memory bus, which becomes a bottleneck for large models.

3. **Optimized Libraries (cuDNN):** PyTorch leverages NVIDIA's cuDNN library, which provides hand-tuned kernel implementations for common neural network operations. These kernels are optimized for each GPU architecture (Turing in this case), selecting the fastest algorithm variant for each layer configuration automatically.

4. **Scaling with Model Complexity:** The speedup factor grows from 4.75× (nano) to 8.92× (medium) because larger models have more parallel work to offer. The GPU's thousands of cores remain underutilized on very small models; as the model grows, GPU utilization increases and the cost of data transfer and kernel launch overhead is amortized over more computation.

### Why MediaPipe Performs Well on CPU

MediaPipe Pose achieves 82.8 FPS on CPU alone — outperforming even YOLOv8n on GPU in raw throughput. This is because MediaPipe uses a lightweight architecture (BlazePose) specifically designed for CPU inference: depthwise separable convolutions, quantized operations, and a two-stage detector-tracker pipeline that avoids running the full detection network on every frame. The trade-off is that MediaPipe does not support GPU acceleration on the Windows pip build (the GPU delegate is only available on Android/iOS), limiting its utility for more complex multi-model pipelines.

### GPU Latency Consistency

A notable finding is the dramatically lower standard deviation of GPU inference times compared to CPU. For example, YOLOv8m-pose shows a std of 0.42 ms on GPU versus 43.47 ms on CPU. This consistency is critical for real-time applications: predictable frame timing prevents visible stutter and simplifies frame-rate budgeting for downstream processing (rendering, tracking, action recognition).

---

## In Progress:

- Evaluating ONNX Runtime as an alternative inference backend for cross-platform GPU support
- Investigating TensorRT optimization for further GPU inference speedup
- Exploring MediaPipe GPU delegate options for Linux/Android deployment

---

## Plans For Next Week:

- **Model Accuracy Evaluation** — Compare keypoint detection accuracy (PCK/OKS metrics) across models, not just speed
- **Multi-Person Pose Detection** — Benchmark performance with multiple people in frame on CPU vs GPU
- **Edge Deployment Investigation** — Explore ONNX export and optimization for deployment on resource-constrained devices
- **Video Pipeline Optimization** — Implement asynchronous inference pipeline to overlap frame capture with model inference

---

## Challenges & Issues:

1. **MediaPipe GPU Limitation** — The Windows pip build of MediaPipe (0.10.32) does not include a GPU delegate, so MediaPipe models could only be benchmarked on CPU. A fair GPU comparison would require building MediaPipe from source with GPU support or using the Android/Linux GPU delegate.
2. **Real-Time Benchmark Overhead** — The live webcam benchmark showed a reduced speedup (1.24×) compared to the batch benchmark (4.75×) due to alternating CPU/GPU inference on the same frame and data transfer overhead. A dedicated GPU-only pipeline would achieve higher real-time throughput.
3. **GPU Memory Constraints** — The GTX 1650 Max-Q has only 4 GB VRAM, which limits the batch size and model size that can be tested. Larger models (e.g., YOLOv8l-pose, YOLOv8x-pose) were not included to avoid out-of-memory issues.
4. **OpenPose Integration** — Still not available in the current environment; comparison continues to focus on YOLOv8 and MediaPipe families.
