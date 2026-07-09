"""
Fridge detection — standalone webcam test.

Same model and settings as the live server (fridge_detection_service.py):
  server/fridge/fridge/best.pt  →  classes: open_fridge, close_fridge
"""

from pathlib import Path

from ultralytics import YOLO

_SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH = _SCRIPT_DIR / "best.pt"
CONFIDENCE = 0.80
IMGSZ = 640

# VIDEO_PATH = _SCRIPT_DIR / "vid" / "5.mp4"
# IMG_PATH = _SCRIPT_DIR / "img" / "3.jpeg"

if not MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"Missing model weights: {MODEL_PATH}\n"
        f"Expected best.pt in {_SCRIPT_DIR}"
    )

model = YOLO(str(MODEL_PATH))

model.predict(
    source=0,
    conf=CONFIDENCE,
    imgsz=IMGSZ,
    save=True,
    show=True,
)
