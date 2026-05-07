@echo off
setlocal enabledelayedexpansion

echo ============================================================
echo   RUPP-AI: Human Pose Detection Demo
echo   Using: YOLO11s-Pose Detection Model
echo ============================================================
echo.
echo Starting Live Pose Detection Demo...
echo Source: sample_video.mp4 (with real-time webcam option)
echo Press "ESC" on the video window to close the demo.
echo.

:: Activate the virtual environment - try .venv first, then .venv311
if exist ".\.venv\Scripts\activate.bat" (
    call .\.venv\Scripts\activate.bat
) else if exist ".\.venv311\Scripts\activate.bat" (
    call .\.venv311\Scripts\activate.bat
) else (
    echo ERROR: Virtual environment not found!
    echo Please ensure .venv or .venv311 exists.
    pause
    exit /b 1
)

:: Determine which model to use
set MODEL=yolo11s-pose.pt
if not exist "!MODEL!" (
    set MODEL=yolov8m-pose.pt
    echo NOTE: Using yolov8m-pose.pt (yolo11s-pose.pt not found)
)

echo Using model: !MODEL!
echo.

:: Run the asynchronous video pipeline with pose model on sample video in GPU-only mode
python async_video_pipeline.py --source sample_video.mp4 --model !MODEL! --device cuda:0 --conf 0.35

pause
