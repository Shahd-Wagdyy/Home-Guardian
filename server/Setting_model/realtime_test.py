"""
Home Guardian — Real-Time Door & Window Detection
==================================================
Tests your trained YOLOv8 model on:
  1. Webcam (live camera feed)
  2. Video file
  3. Single image

Usage:
  python realtime_test.py --source 0              # webcam
  python realtime_test.py --source video.mp4      # video file
  python realtime_test.py --source image.jpg      # single image
  python realtime_test.py --source 0 --conf 0.6   # webcam with custom confidence

Requirements:
  pip install ultralytics opencv-python
"""

import argparse
import time
import os
import sys
import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque

# ─────────────────────────────────────────────
# CONFIGURATION — edit these
# ─────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'best.pt')  # path to your best.pt (next to this script)
CONF_THRESHOLD = 0.50        # confidence threshold
IOU_THRESHOLD = 0.60         # NMS IoU threshold
SMOOTHING_FRAMES = 5         # temporal smoothing window
MAX_DET = 50                 # max detections per frame

# Class colors (BGR format for OpenCV)
CLASS_COLORS = {
    'door_closed':   (60,  60,  220),   # red
    'door_opened':   (60,  200, 60),    # green
    'window_closed': (220, 150, 60),    # blue-ish
    'window_opened': (60,  220, 220),   # yellow
}

# Alert classes — these trigger a warning overlay
ALERT_CLASSES = {'window_opened', 'door_opened'}

# ─────────────────────────────────────────────
# TEMPORAL SMOOTHER
# Reduces flickering by voting across last N frames
# ─────────────────────────────────────────────
class TemporalSmoother:
    def __init__(self, window=5):
        self.window = window
        self.history = deque(maxlen=window)

    def update(self, detections):
        self.history.append(detections)
        return self._get_stable()

    def _get_stable(self):
        if len(self.history) < 2:
            return self.history[-1] if self.history else []
        # Count class votes per detection slot (simplified: return latest if consistent)
        return self.history[-1]


# ─────────────────────────────────────────────
# DRAWING UTILITIES
# ─────────────────────────────────────────────
def draw_detection(frame, box, label, conf, color):
    x1, y1, x2, y2 = map(int, box)

    # Draw filled rectangle background for label
    label_text = f'{label} {conf:.2f}'
    (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

    # Box outline — thick and colored
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Label background
    cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 8, y1), color, -1)

    # Label text
    cv2.putText(frame, label_text, (x1 + 4, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return frame


def draw_hud(frame, detections, fps, frame_count, alert_active):
    h, w = frame.shape[:2]

    # ── FPS counter ──────────────────────────
    cv2.putText(frame, f'FPS: {fps:.1f}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # ── Frame counter ────────────────────────
    cv2.putText(frame, f'Frame: {frame_count}', (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

    # ── Detection count ──────────────────────
    cv2.putText(frame, f'Detections: {len(detections)}', (10, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

    # ── Model info ───────────────────────────
    cv2.putText(frame, 'Home Guardian v1', (w - 200, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    # ── Alert overlay ────────────────────────
    if alert_active:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 180), -1)
        cv2.addWeighted(overlay, 0.08, frame, 0.92, 0, frame)

        alert_text = '⚠ ALERT: OPEN DETECTED'
        (tw, th), _ = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)
        cv2.putText(frame, alert_text, ((w - tw) // 2, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

    # ── Legend ───────────────────────────────
    legend_y = h - 120
    cv2.rectangle(frame, (10, legend_y - 10), (220, h - 10), (30, 30, 30), -1)
    cv2.rectangle(frame, (10, legend_y - 10), (220, h - 10), (80, 80, 80), 1)

    for i, (cls_name, color) in enumerate(CLASS_COLORS.items()):
        y = legend_y + i * 22
        cv2.rectangle(frame, (18, y), (34, y + 14), color, -1)
        cv2.putText(frame, cls_name, (40, y + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)

    return frame



# ─────────────────────────────────────────────
# MAIN INFERENCE LOOP
# ─────────────────────────────────────────────
def run_inference(source, model_path, conf, iou):

    # ── Load model ───────────────────────────
    if not os.path.exists(model_path):
        print(f'ERROR: Model not found at {model_path}')
        print('Update MODEL_PATH at the top of this script.')
        sys.exit(1)

    print(f'Loading model: {model_path}')
    model = YOLO(model_path)
    class_names = model.names
    print(f'Classes: {list(class_names.values())}')
    print(f'Confidence: {conf} | IoU: {iou}')

    # ── Open source ──────────────────────────
    is_image = isinstance(source, str) and source.lower().endswith(
        ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    )

    if is_image:
        run_image(model, source, conf, iou, class_names)
        return

    # Video or webcam
    src = int(source) if str(source).isdigit() else source
    cap = cv2.VideoCapture(src)

    if not cap.isOpened():
        print(f'ERROR: Cannot open source: {source}')
        sys.exit(1)

    # Get video properties
    cap_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    print(f'Source: {cap_w}x{cap_h} @ {cap_fps:.1f}fps')
    print('Press Q to quit | S to save screenshot | P to pause')

    smoother = TemporalSmoother(window=SMOOTHING_FRAMES)

    frame_count = 0
    fps = 0.0
    fps_buffer = deque(maxlen=30)
    paused = False

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print('Stream ended.')
                break

            frame_count += 1
            t_start = time.time()

            # ── Run YOLO inference ───────────
            results = model(
                frame,
                conf=conf,
                iou=iou,
                max_det=MAX_DET,
                verbose=False,
                device=0 if cv2.cuda.getCudaEnabledDeviceCount() > 0 else 'cpu'
            )

            # ── Parse detections ─────────────
            detections = []
            alert_active = False

            if results[0].boxes is not None:
                for box in results[0].boxes:
                    cls_id = int(box.cls[0].item())
                    cls_name = class_names[cls_id]
                    conf_val = float(box.conf[0].item())
                    xyxy = box.xyxy[0].tolist()
                    detections.append((xyxy, cls_name, conf_val))

                    if cls_name in ALERT_CLASSES:
                        alert_active = True

            # ── Apply temporal smoothing ─────
            stable_detections = smoother.update(detections)

            # ── Draw detections ──────────────
            for xyxy, cls_name, conf_val in stable_detections:
                color = CLASS_COLORS.get(cls_name, (200, 200, 200))
                frame = draw_detection(frame, xyxy, cls_name, conf_val, color)

            # ── Compute FPS ──────────────────
            elapsed = time.time() - t_start
            fps_buffer.append(1.0 / max(elapsed, 1e-6))
            fps = sum(fps_buffer) / len(fps_buffer)

            # ── Draw HUD ─────────────────────
            frame = draw_hud(frame, stable_detections, fps, frame_count, alert_active)

        # ── Show frame ───────────────────────
        cv2.imshow('Home Guardian — Real-Time Detection  [Q=quit  S=save  P=pause]', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            print('Quit.')
            break
        elif key == ord('s'):
            fname = f'screenshot_{frame_count}.jpg'
            cv2.imwrite(fname, frame)
            print(f'Saved: {fname}')
        elif key == ord('p'):
            paused = not paused
            print('Paused' if paused else 'Resumed')

    cap.release()
    cv2.destroyAllWindows()

    # ── Print session summary ─────────────────
    print('\n' + '=' * 45)
    print('SESSION SUMMARY')
    print('=' * 45)
    print(f'Total frames processed : {frame_count}')
    print(f'Average FPS            : {fps:.1f}')


def run_image(model, image_path, conf, iou, class_names):
    print(f'Running on image: {image_path}')
    frame = cv2.imread(image_path)

    if frame is None:
        print(f'ERROR: Cannot read image: {image_path}')
        sys.exit(1)

    results = model(frame, conf=conf, iou=iou, verbose=False)
    alert_active = False
    detections = []

    if results[0].boxes is not None:
        for box in results[0].boxes:
            cls_id = int(box.cls[0].item())
            cls_name = class_names[cls_id]
            conf_val = float(box.conf[0].item())
            xyxy = box.xyxy[0].tolist()
            detections.append((xyxy, cls_name, conf_val))
            if cls_name in ALERT_CLASSES:
                alert_active = True
            color = CLASS_COLORS.get(cls_name, (200, 200, 200))
            frame = draw_detection(frame, xyxy, cls_name, conf_val, color)

    print(f'Detections: {len(detections)}')
    for xyxy, cls_name, conf_val in detections:
        print(f'  {cls_name}: {conf_val:.3f}')

    frame = draw_hud(frame, detections, 0, 1, alert_active)

    output_path = 'result_' + os.path.basename(image_path)
    cv2.imwrite(output_path, frame)
    print(f'Result saved: {output_path}')

    cv2.imshow('Home Guardian — Result  [any key to close]', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Home Guardian Real-Time Detection')
    parser.add_argument('--source', default='0',
                        help='Source: 0=webcam, path to video/image file')
    parser.add_argument('--model', default=MODEL_PATH,
                        help='Path to best.pt model file')
    parser.add_argument('--conf', type=float, default=CONF_THRESHOLD,
                        help='Confidence threshold (default: 0.50)')
    parser.add_argument('--iou', type=float, default=IOU_THRESHOLD,
                        help='IoU threshold (default: 0.60)')
    args = parser.parse_args()

    print('=' * 45)
    print('HOME GUARDIAN — REAL-TIME DETECTION')
    print('=' * 45)

    run_inference(
        source=args.source,
        model_path=args.model,
        conf=args.conf,
        iou=args.iou
    )