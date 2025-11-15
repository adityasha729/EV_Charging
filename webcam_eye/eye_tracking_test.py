import cv2
import numpy as np
import pyautogui
from eyetrax import GazeEstimator, run_9_point_calibration
import time

def main():
    # Create gaze estimator
    estimator = GazeEstimator()

    
    estimator.load_model("webcam_eye/gaze_model.pkl")

    screen_w, screen_h = pyautogui.size()
    window_name = "Gaze Tracker"
    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam.")

    
    alpha = 0.4
    prev_x, prev_y = None, None

    print("🟢 Tracking started — press ESC to exit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        features, blink = estimator.extract_features(frame)
        if features is not None and not blink:
            x, y = estimator.predict([features])[0]

            # Smooth gaze movement
            if prev_x is not None:
                x = int(alpha * x + (1 - alpha) * prev_x)
                y = int(alpha * y + (1 - alpha) * prev_y)
            prev_x, prev_y = x, y
        else:
            
            pass

        
        canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
        if prev_x and prev_y:
            cv2.circle(canvas, (int(prev_x), int(prev_y)), 20, (0, 0, 255), -1)

        cv2.imshow(window_name, canvas)

        # Exit on ESC
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("🔴 Tracking stopped.")

if __name__ == "__main__":
    main()
