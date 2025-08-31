import time
import cv2
import mediapipe as mp
import numpy as np

class FPSMeter:
    def __init__(self, smoothing=0.9):
        self.last = time.time()
        self.fps = 0.0
        self.smoothing = smoothing

    def update(self):
        now = time.time()
        dt = now - self.last
        if dt > 0:
            inst = 1.0 / dt
            self.fps = (self.fps * self.smoothing) + (inst * (1.0 - self.smoothing))
        self.last = now

def draw_landmarks_and_bbox(image, hand_landmarks, label=None, confidence=0.0):
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands

    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # compute bbox
    h, w, _ = image.shape
    xs = [p.x for p in hand_landmarks.landmark]
    ys = [p.y for p in hand_landmarks.landmark]
    min_x = int(max(min(xs) * w - 10, 0))
    max_x = int(min(max(xs) * w + 10, w))
    min_y = int(max(min(ys) * h - 10, 0))
    max_y = int(min(max(ys) * h + 10, h))
    cv2.rectangle(image, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)

    # label
    if label:
        text = f"{label} ({confidence:.2f})"
    else:
        text = "Detecting..."
    cv2.putText(image, text, (min_x, min_y - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
