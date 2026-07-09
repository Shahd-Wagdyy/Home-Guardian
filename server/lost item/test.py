"""
Home Guardian — Lost Item Detection Test Script
Runs the trained model on webcam feed and displays
class names + confidence on bounding boxes.
"""

import cv2
import time
from pathlib import Path
from ultralytics import YOLO

_SCRIPT_DIR = Path(__file__).resolve().parent

# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────
CONFIG = {
    "MODEL_PATH": str(_SCRIPT_DIR / "best(2).pt"),
    "SOURCE": 0,               # 0 = laptop cam, 1 = USB cam (auto-fallback)
    "CONFIDENCE": 0.3,        # minimum confidence threshold
    "IMGSZ": 640,
    "PROCESS_EVERY_N": 2,      # process every 2nd frame for speed
}

# ──────────────────────────────────────────────
# COLORS — one per class (auto-assigned)
# ──────────────────────────────────────────────
COLORS = [
    (0, 255, 0),     (0, 0, 255),    (255, 0, 0),
    (0, 255, 255),   (255, 0, 255),  (255, 165, 0),
    (128, 0, 128),   (0, 128, 255),  (0, 255, 128),
    (255, 128, 0),   (128, 255, 0),  (255, 255, 0),
]

def get_color(class_id):
    return COLORS[class_id % len(COLORS)]


# ──────────────────────────────────────────────
# DRAW DETECTIONS
# ──────────────────────────────────────────────
def draw_detections(frame, boxes, class_names):
    for box in boxes:
        class_id = int(box.cls)
        conf     = float(box.conf)
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        color     = get_color(class_id)
        cls_name  = class_names.get(class_id, f"class_{class_id}")
        label     = f"{cls_name} {conf:.0%}"

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Label background
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x1, y1 - lh - 10), (x1 + lw + 6, y1), color, -1)

        # Label text
        cv2.putText(frame, label, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


# ──────────────────────────────────────────────
# INFO PANEL
# ──────────────────────────────────────────────
def draw_info(frame, fps, detection_count, class_names):
    h, w = frame.shape[:2]

    cv2.rectangle(frame, (0, 0), (260, 55 + len(class_names) * 18), (0, 0, 0), -1)
    cv2.rectangle(frame, (0, 0), (260, 55 + len(class_names) * 18), (50, 50, 50), 1)

    def put(text, y, color=(200, 200, 200)):
        cv2.putText(frame, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 1, cv2.LINE_AA)

    put("Home Guardian — Lost Item Test", 18)
    put(f"FPS: {fps:.1f}  |  Detections: {detection_count}", 38,
        (0, 255, 0) if detection_count > 0 else (200, 200, 200))

    # Class legend
    y = 58
    for cid, cname in class_names.items():
        color = get_color(cid)
        cv2.rectangle(frame, (8, y - 10), (20, y + 2), color, -1)
        put(f"  {cname}", y, (200, 200, 200))
        y += 18


def open_camera(preferred_source):
    """Open webcam; try preferred index then fall back to 0, 1, 2."""
    candidates = []
    for idx in (preferred_source, 0, 1, 2):
        if idx not in candidates:
            candidates.append(idx)

    for idx in candidates:
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            ret, _ = cap.read()
            if ret:
                print(f"[Info] Using camera index {idx}")
                return cap
        cap.release()

    return None


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def run():
    print("[Home Guardian] Lost Item Detection Test")
    print(f"[Info] Loading model from: {CONFIG['MODEL_PATH']}")

    model = YOLO(CONFIG["MODEL_PATH"], task="detect")
    class_names = model.names
    print(f"[Info] Classes detected: {list(class_names.values())}")
    print("[Info] Press Q to quit | Press S to save screenshot")

    cap = open_camera(CONFIG["SOURCE"])
    if cap is None:
        print(
            f"[ERROR] Cannot open any camera (tried index {CONFIG['SOURCE']}, 0, 1, 2). "
            "Close other apps using the webcam or change CONFIG['SOURCE']."
        )
        return

    # Warmup
    print("[Info] Warming up camera...")
    time.sleep(1)
    for _ in range(15):
        cap.read()

    frame_count = 0
    fps = 0
    prev_time = time.time()
    last_boxes = []
    screenshot_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[Info] End of stream")
            break

        frame_count += 1

        # ── FPS calculation ──
        now = time.time()
        fps = 1.0 / (now - prev_time + 1e-6)
        prev_time = now

        # ── Run inference every N frames ──
        if frame_count % CONFIG["PROCESS_EVERY_N"] == 0:
            results = model(frame, verbose=False,
                            conf=CONFIG["CONFIDENCE"],
                            imgsz=CONFIG["IMGSZ"])[0]
            last_boxes = results.boxes

        # ── Draw ──
        draw_detections(frame, last_boxes, class_names)
        draw_info(frame, fps, len(last_boxes), class_names)

        cv2.imshow("Home Guardian — Lost Item Detection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            screenshot_count += 1
            filename = f"screenshot_{screenshot_count}.jpg"
            cv2.imwrite(filename, frame)
            print(f"[Saved] Screenshot saved as {filename}")

    cap.release()
    cv2.destroyAllWindows()
    print("[Info] Done!")


if __name__ == "__main__":
    run()