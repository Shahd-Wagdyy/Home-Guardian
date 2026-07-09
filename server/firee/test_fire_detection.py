"""
Test a YOLO fire-detection model on a single image.

Usage:
    python test_fire_detection.py --image test.jpg
    python test_fire_detection.py --model "New folder/best_2.pt" --image test.jpg --conf 0.4 --save
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

_SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_MODEL = _SCRIPT_DIR / "New folder" / "best_2.pt"


def _resolve_model(path: str | None) -> Path:
    if path:
        candidate = Path(path)
        if not candidate.is_file():
            candidate = _SCRIPT_DIR / path
        if not candidate.is_file():
            raise FileNotFoundError(f"Model not found: {path}")
        return candidate

    if _DEFAULT_MODEL.is_file():
        return _DEFAULT_MODEL

    matches = sorted(_SCRIPT_DIR.rglob("best*.pt"))
    if matches:
        return matches[0]

    raise FileNotFoundError(
        f"No fire model found under {_SCRIPT_DIR}. "
        f"Expected {_DEFAULT_MODEL} or pass --model."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Test a YOLO fire-detection model on an image.")
    parser.add_argument(
        "--model",
        default=None,
        help=f"Path to the .pt model file (default: {_DEFAULT_MODEL.name} in firee/)",
    )
    parser.add_argument("--image", required=True, help="Path to the image to test")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (0-1)")
    parser.add_argument("--save", action="store_true", help="Save the annotated image to disk")
    parser.add_argument("--show", action="store_true", help="Open a window showing the result")
    args = parser.parse_args()

    model_path = _resolve_model(args.model)
    image_path = Path(args.image)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {args.image}")

    model = YOLO(str(model_path))
    print(f"Model: {model_path}")
    print(f"Image: {image_path}")
    print(f"Classes: {model.names}")
    results = model.predict(
        source=str(image_path),
        conf=args.conf,
        save=args.save,
        show=args.show,
    )

    result = results[0]
    if result.boxes is None or len(result.boxes) == 0:
        print("No detections.")
    else:
        print(f"Found {len(result.boxes)} detection(s):")
        for box in result.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            print(
                f"  {label}: {confidence:.2%}  "
                f"box=[{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]"
            )

    if args.save:
        print(f"\nAnnotated image saved to: {result.save_dir}")


if __name__ == "__main__":
    main()
