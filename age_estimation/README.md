# Age Estimation App

This small Python application uses the [DeepFace](https://github.com/serengil/deepface) library to estimate the age from a webcam stream. It opens a window, reads frames from the default camera and overlays the estimated age. No images are stored on disk; they are processed directly in memory.

The app tries to detect glasses, masks or exaggerated facial expressions. If one of these is detected, the age is **not** estimated and a red error message is shown in the window.

## Requirements

Install the dependencies into a virtual environment (recommended):

```bash
pip install -r requirements.txt
```

## Running

```bash
python main.py
```

Press `q` to quit the application.
