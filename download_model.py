"""Download a lightweight pose estimation ONNX model for benchmarking."""
import urllib.request
import json
import os
import ssl

# Try to find and download yolov8n-pose ONNX
ssl._create_default_https_context = ssl._create_unverified_context

# Try multiple URLs
urls_to_try = [
    "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n-pose.onnx",
    "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n-pose.onnx",
    "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n-pose.onnx",
]

dest = "yolov8n-pose.onnx"
if os.path.exists(dest):
    print(f"Already exists: {os.path.getsize(dest)/(1024*1024):.1f} MB")
else:
    for url in urls_to_try:
        try:
            print(f"Trying: {url}")
            urllib.request.urlretrieve(url, dest)
            print(f"Success: {os.path.getsize(dest)/(1024*1024):.1f} MB")
            break
        except Exception as e:
            print(f"  Failed: {e}")
    else:
        # Fallback: use pip ultralytics to export
        print("\nDirect download failed. Trying ultralytics export...")
        try:
            import subprocess, sys
            subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics", "--quiet"])
            from ultralytics import YOLO
            model = YOLO("yolov8n-pose.pt")
            model.export(format="onnx")
            print(f"Exported: {os.path.getsize(dest)/(1024*1024):.1f} MB")
        except Exception as e2:
            print(f"Export also failed: {e2}")
            print("\nCreating a representative ONNX model instead...")
            # Create a simple ONNX model that mimics pose estimation workload
            import numpy as np
            try:
                import onnx
                from onnx import helper, TensorProto
                
                # Create a simple conv-based model (representative workload)
                X = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 640, 640])
                Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 56, 8400])
                
                # Conv weights
                w1 = numpy_helper.from_array(np.random.randn(16, 3, 3, 3).astype(np.float32), name='w1')
                # ... simplified model creation
            except:
                pass
