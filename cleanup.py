import os
import shutil
import urllib.request

# Files and directories to keep for demo
KEEP = {
    ".venv", ".venv311", ".gitignore", ".git", ".idea", "__pycache__", 
    "async_video_pipeline.py", "demo_progress_report.py",
    "requirements_python.txt", "START_LIVE_DEMO.bat", 
    "yolov8m-pose.pt", "yolo11s-pose.pt", "sample_video.mp4",
    ".vscode", ".gitignore"
}

print("[INFO] Cleaning up unnecessary files...")
cleaned_count = 0
for item in os.listdir("."):
    if item not in KEEP and item != "cleanup.py":
        try:
            if os.path.isdir(item):
                shutil.rmtree(item, ignore_errors=True)
                print(f"  ✓ Deleted folder: {item}")
                cleaned_count += 1
            else:
                os.remove(item)
                print(f"  ✓ Deleted file: {item}")
                cleaned_count += 1
        except Exception as e:
            print(f"  ✗ Failed to delete {item}: {e}")

if not os.path.exists("sample_video.mp4"):
    print("\n[INFO] Downloading sample video for demo...")
    video_url = "https://github.com/intel-iot-devkit/sample-videos/raw/master/person-bicycle-car-detection.mp4"
    try:
        urllib.request.urlretrieve(video_url, "sample_video.mp4")
        print("  ✓ Sample video downloaded successfully.")
    except Exception as e:
        print(f"  ✗ Failed to download sample video: {e}")
else:
    print("\n[INFO] sample_video.mp4 already exists - skipping download.")

print(f"\n[SUCCESS] Cleanup complete! Removed {cleaned_count} items.")
print("[INFO] Project is now ready for demo.")

