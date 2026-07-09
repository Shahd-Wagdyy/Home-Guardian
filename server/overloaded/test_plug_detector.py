"""
realtime_plug_detector.py
Real-time overloaded-plug detection from a webcam (runs LOCALLY, not in Colab).

USAGE
-----
python realtime_plug_detector.py --weights best.pt
python realtime_plug_detector.py --weights best.pt --conf 0.6 --cam 0

LIVE CONTROLS (focus the video window)
--------------------------------------
q       quit
+ / -   raise / lower confidence threshold on the fly
s       save a snapshot of the current frame
"""

import argparse
import time
import os
import cv2
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=r"D:\flutter_application_1\flutter_application_1\server\overloaded\best.pt", help="path to trained best.pt")
    ap.add_argument("--cam", default="0", help="webcam index (0 is usually the built-in cam)")
    ap.add_argument("--conf", type=float, default=0.65, help="starting confidence threshold")
    ap.add_argument("--width", type=int, default=1280, help="capture width")
    ap.add_argument("--height", type=int, default=720, help="capture height")
    args = ap.parse_args()

    if not os.path.exists(args.weights):
        print(f"Weights not found: {args.weights}")
        return

    model = YOLO(args.weights)
    print(f"Loaded {args.weights} | classes: {model.names}")

    cap = cv2.VideoCapture(int(args.cam))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        print(f"Could not open camera index {args.cam}")
        return

    conf = args.conf
    prev = time.time()
    fps = 0.0
    snap_count = 0
    print("Running. Press 'q' to quit, '+'/'-' to adjust confidence, 's' to snapshot.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read frame from camera.")
            break

        # inference on the current frame
        results = model.predict(frame, conf=conf, verbose=False)
        annotated = results[0].plot()
        n = len(results[0].boxes)

        # smoothed FPS
        now = time.time()
        inst_fps = 1.0 / max(now - prev, 1e-6)
        fps = 0.9 * fps + 0.1 * inst_fps if fps else inst_fps
        prev = now

        # overlay info
        overlay = f"FPS: {fps:4.1f}   conf>={conf:.2f}   detections: {n}"
        cv2.putText(annotated, overlay, (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("Overloaded Plug Detector - real time", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key in (ord("+"), ord("=")):
            conf = min(0.95, round(conf + 0.05, 2))
            print(f"conf -> {conf}")
        elif key in (ord("-"), ord("_")):
            conf = max(0.05, round(conf - 0.05, 2))
            print(f"conf -> {conf}")
        elif key == ord("s"):
            fname = f"snapshot_{snap_count:03d}.jpg"
            cv2.imwrite(fname, annotated)
            print(f"saved {fname}")
            snap_count += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()