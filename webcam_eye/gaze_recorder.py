import cv2
import csv
import time
import signal
import argparse
import os
from eyetrax import GazeEstimator


def write_row(writer, timestamp, frame_idx, x, y, blink):
    writer.writerow({
        "timestamp": f"{timestamp:.6f}",
        "frame": frame_idx,
        "x": "" if x is None else int(x),
        "y": "" if y is None else int(y),
        "blink": int(bool(blink)),
    })


def record(output, camera_index=0, model_path="webcam_eye/gaze_model.pkl", fps=30):
    # Ensure directory exists
    out_dir = os.path.dirname(output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    estimator = GazeEstimator()
    estimator.load_model(model_path)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open webcam {camera_index}")

    # Try to set a frame rate (may be ignored by some cameras)
    cap.set(cv2.CAP_PROP_FPS, fps)

    running = True

    def _signal_handler(signum, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    with open(output, "a", newline="", encoding="utf-8") as f:
        fieldnames = ["timestamp", "frame", "x", "y", "blink"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        # If file is empty write header
        if f.tell() == 0:
            writer.writeheader()

        frame_idx = 0
        while running:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            features, blink = estimator.extract_features(frame)
            x = y = None
            if features is not None and not blink:
                try:
                    x, y = estimator.predict([features])[0]
                except Exception:
                    x = y = None

            # Remove/clamp out-of-bounds values based on configured screen size and buffer
            try:
                screen_w = int(record.screen_w)
                screen_h = int(record.screen_h)
                buffer = int(record.buffer)
            except Exception:
                # Fallback to defaults if attributes are not set
                screen_w, screen_h, buffer = 1920, 1080, 100

            if x is not None and y is not None:
                # If point is farther than buffer outside the screen, drop it
                if (x < -buffer) or (x > screen_w + buffer) or (y < -buffer) or (y > screen_h + buffer):
                    x = None
                    y = None
                else:
                    # Clamp points within [0, screen_w] and [0, screen_h]
                    x = max(0, min(x, screen_w))
                    y = max(0, min(y, screen_h))

            write_row(writer, time.time(), frame_idx, x, y, blink)
            f.flush()
            frame_idx += 1

    cap.release()


def cli():
    parser = argparse.ArgumentParser(description="Headless gaze recorder — saves timestamp,x,y,blink to CSV")
    parser.add_argument("--output", "-o", default="webcam_eye/gaze_log.csv", help="CSV output path")
    parser.add_argument("--camera", "-c", type=int, default=0, help="Camera index for cv2.VideoCapture")
    parser.add_argument("--model", "-m", default="webcam_eye/gaze_model.pkl", help="Path to gaze model pickle")
    parser.add_argument("--fps", type=int, default=30, help="Target camera FPS")
    parser.add_argument("--screen-w", type=int, default=1920, help="Screen width in pixels (default 1920)")
    parser.add_argument("--screen-h", type=int, default=1080, help="Screen height in pixels (default 1080)")
    parser.add_argument("--buffer", type=int, default=100, help="Buffer in pixels around screen to accept/ clamp points (default 100)")
    args = parser.parse_args()

    print(f"Starting headless gaze recorder -> {args.output}")
    print("Stop with Ctrl+C")
    # Attach screen params to function so the record loop can read them quickly
    record.screen_w = args.screen_w
    record.screen_h = args.screen_h
    record.buffer = args.buffer
    record(args.output, camera_index=args.camera, model_path=args.model, fps=args.fps)


if __name__ == "__main__":
    cli()
