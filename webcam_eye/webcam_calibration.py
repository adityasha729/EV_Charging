from eyetrax import GazeEstimator, run_9_point_calibration
import time

def main():
    # Create gaze estimator
    estimator = GazeEstimator()

    print("🟡 Starting 9-point calibration. Follow the on-screen dots...")

    run_9_point_calibration(estimator)
    print("✅ Calibration complete!")

    estimator.save_model("webcam_eye/gaze_model.pkl")

if __name__ == "__main__":
    main()