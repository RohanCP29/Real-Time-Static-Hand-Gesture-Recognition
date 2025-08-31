# Real-Time Static Hand Gesture Recognition

**Your Full Name:** `Rohhan Patil`

---

## Overview
This project implements a **real-time Python application** that uses a webcam to detect and recognize a set of static hand gestures. It captures a live video feed, detects and tracks a single hand, extracts landmark-based features, and classifies the current hand pose into one of the supported gestures — all with on-screen feedback and optional recording.

---

## Gesture Vocabulary
The application recognizes **four** static gestures:

1. **Open Palm**  
2. **Fist**  
3. **Peace Sign (V-sign)**  
4. **Thumbs Up**

A detected gesture’s name is displayed on the output video window in real time.

---

## Technology Choice & Justification
- **MediaPipe Hands** — for hand detection and 21 landmark extraction. Fast and CPU-friendly.
- **OpenCV** — for webcam capture, drawing overlays, and recording.
- **NumPy** — for numeric operations (angles, distances, normalization).
- **scikit-learn + joblib** — optional, for training lightweight classifiers.

**Why this stack?**  
MediaPipe + OpenCV ensures robust, low-latency detection suitable for real-time use while keeping the app lightweight.

---

## How the System Works
1. Capture video frames with OpenCV.  
2. Detect hand and extract landmarks with MediaPipe.  
3. Normalize landmarks to scale/position.  
4. Extract features (angles, distances, orientation).  
5. Apply rules-based classifier:
   - **Open Palm**: all fingers extended.  
   - **Fist**: all fingers folded.  
   - **Peace**: index & middle extended, others folded.  
   - **Thumbs Up**: four fingers folded, thumb extended upwards.  
6. Smooth predictions with a sliding window + debounce.  
7. Display gesture name and FPS on screen.  
8. Optionally record annotated video.

---

## Installation
Run each installation command separately:

```bash
pip install opencv-python
pip install mediapipe
pip install numpy
pip install scikit-learn
pip install joblib
```

⚠️ If mediapipe fails:

Make sure Python version is 3.8–3.11

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

---

## Running the App
```bash
python app.py
```

### Controls
- `q` → quit
- `c` → calibrate (show open palm for 2s)
- `d` → toggle data capture

Recordings are saved in the `demo/` folder.

---

## Project Structure
```
hand-gestures/
├─ src/
│  ├─ app.py                 # main loop, calibration, recording
│  ├─ features.py            # normalization & feature extraction
│  ├─ gestures.py            # rules-based classifier with smoothing & calibration
│  └─ utils.py               # drawing helpers and FPS meter
├─ demo/
│  └─ recording_YYYYMMDD_HHMMSS.avi
├─ captured_data.csv         # optional: features + labels
└─ README.md
```

---

## Demo
A demonstration video is included in the `demo/` folder:

```bash
demo/demo.mp4
```
If you run the app, an annotated `.avi` file will also be saved automatically.

---

## Performance
- Runs in real-time (>25 FPS on CPU).
- Adjustable smoothing to reduce flicker.
- Works on standard laptop webcam input.

---

## Troubleshooting
- mediapipe not installing → check Python version (3.8–3.11 only).
- Low FPS → reduce webcam resolution or set `model_complexity=0`.
- Flickering gestures → increase smoothing window/debounce values.

---

## License
Add your license (e.g., MIT, Apache 2.0).

---

## Contact
For improvements or questions:  
rohanpatil4002@gmail.com

## Code Quality

- Code is modular and organized by functionality.
- Inline comments explain complex or non-obvious logic.
- Functions are named descriptively for readability.

---

## Requirements

- Python 3.7+
- OpenCV
- MediaPipe

Install dependencies with:
```bash
pip install -r requirements.txt
```

