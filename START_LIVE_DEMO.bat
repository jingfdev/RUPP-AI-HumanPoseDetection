@echo off
echo Starting Live Webcam Pose Detection Demo...
echo Press "ESC" on the video window to close the demo.
echo.

:: Activate the virtual environment
call .\.venv\Scripts\Activate.ps1 || call .\.venv\Scripts\activate.bat

:: Run the asynchronous video pipeline with the sample video
python async_video_pipeline.py --source sample_video.mp4

pause
