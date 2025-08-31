import cv2
import mediapipe as mp
import time
from collections import deque
import numpy as np
import os
import csv
from datetime import datetime

from features import extract_features_from_landmarks, normalize_landmarks
from gestures import GestureClassifier
from utils import FPSMeter, draw_landmarks_and_bbox

# Configuration
CAM_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
MAX_HANDS = 1
DATA_OUTPUT = "captured_data.csv"  
OUTPUT_DIR = "D:\\hand-gestures\\demo"

# Ensure demo folder exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Create classifier
classifier = GestureClassifier(window_size=10, debounce_count=5)

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=MAX_HANDS,
    model_complexity=0,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
)

# Video capture setup
cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW if os.name == 'nt' else 0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

#Video Writer setup
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
video_path = os.path.join(OUTPUT_DIR, f"recording_{timestamp}.avi")
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # codec
out = cv2.VideoWriter(video_path, fourcc, 20.0, (FRAME_WIDTH, FRAME_HEIGHT))
print(f"Recording video to {video_path}")

fps = FPSMeter()
data_capture = False
print("Press 'q' to quit, 'c' to calibrate (show open palm), 'd' to toggle data capture.")

# Prepare data file header if needed
if not os.path.exists(DATA_OUTPUT):
    with open(DATA_OUTPUT, 'w', newline='') as f:
        writer = csv.writer(f)
        header = [f'f{i}' for i in range(0, 30)] + ['label']
        writer.writerow(header)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read from camera. Exiting.")
            break

        start = time.time()
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        annotation_frame = frame.copy()

        gesture_label = None
        confidence = 0.0
        hand_present = False

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            hand_present = True

            lm = np.array([[p.x, p.y, p.z] for p in hand_landmarks.landmark], dtype=float)
            lm_norm = normalize_landmarks(lm)
            feat = extract_features_from_landmarks(lm_norm)

            gesture_label, confidence = classifier.predict(feat, lm_norm)

            if gesture_label:
                draw_landmarks_and_bbox(annotation_frame, hand_landmarks, gesture_label, confidence)
            else:
                mp_drawing.draw_landmarks(annotation_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        fps.update()
        cv2.putText(annotation_frame, f"FPS: {fps.fps:.1f}", (annotation_frame.shape[1]-160, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        # Show and record
        cv2.imshow("Hand Gesture Recognition", annotation_frame)
        out.write(annotation_frame)  

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            print("Calibration: Please show OPEN PALM for 2 seconds...")
            calib_samples = []
            calib_start = time.time()
            while time.time() - calib_start < 2.0:
                ret2, frame2 = cap.read()
                if not ret2:
                    break
                img_rgb2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
                res2 = hands.process(img_rgb2)
                if res2.multi_hand_landmarks:
                    lm2 = np.array([[p.x, p.y, p.z] for p in res2.multi_hand_landmarks[0].landmark])
                    lm2_norm = normalize_landmarks(lm2)
                    feat2 = extract_features_from_landmarks(lm2_norm)
                    calib_samples.append(feat2)
                cv2.imshow("Hand Gesture Recognition", frame2)
                cv2.waitKey(1)
            if len(calib_samples) > 0:
                classifier.calibrate_open_palm(np.mean(calib_samples, axis=0))
                print("Calibration complete.")
            else:
                print("Calibration failed — no hand seen.")
        elif key == ord('d'):
            data_capture = not data_capture
            print("Data capture:", data_capture)

finally:
    cap.release()
    out.release()  
    cv2.destroyAllWindows()
    hands.close()
