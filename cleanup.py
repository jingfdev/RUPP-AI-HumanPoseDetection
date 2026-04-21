import os
import shutil
import urllib.request

# Files and directories to keep
KEEP = {
    ".venv", ".gitignore", ".git", ".idea", "__pycache__", 
    "async_video_pipeline.py", "requirements_python.txt", 
    "START_LIVE_DEMO.bat", "yolov8m-pose.pt"
}

print("Cleaning up unnecessary files...")
for item in os.listdir("."):
    if item not in KEEP and not item.endswith(".mp4") and item != "cleanup.py":
        try:
            if os.path.isdir(item):
                shutil.rmtree(item, ignore_errors=True)
                print(f"Deleted folder: {item}")
            else:
                os.remove(item)
                print(f"Deleted file: {item}")
        except Exception as e:
            print(f"Failed to delete {item}: {e}")

print("\nDownloading a sample video for the demo...")
video_url = "https://github.com/intel-iot-devkit/sample-videos/raw/master/person-bicycle-car-detection.mp4"
if not os.path.exists("sample_video.mp4"):
    try:
        urllib.request.urlretrieve(video_url, "sample_video.mp4")
        print("Sample video downloaded successfully.")
    except Exception as e:
        print(f"Failed to download sample video: {e}")

print("\nCleanup Complete!")
