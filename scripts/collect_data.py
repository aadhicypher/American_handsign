import cv2
import mediapipe as mp
import numpy as np
import os
import random
import time

# =============================
# CONFIG
# =============================
LABELS = ['A', 'B', 'C']
SAMPLES_PER_LABEL = 50
TRAIN_SPLIT = 0.8

BASE_DIR = "dataset"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR = os.path.join(BASE_DIR, "test")

# =============================
# CREATE DIRECTORIES
# =============================
print("Creating directories...")
for split in [TRAIN_DIR, TEST_DIR]:
    for label in LABELS:
        path = os.path.join(split, label)
        os.makedirs(path, exist_ok=True)
        print(f"Created: {path}")

# =============================
# GLOBAL RESULT STORAGE
# =============================
latest_result = None

# =============================
# CALLBACK FUNCTION
# =============================
def result_callback(result, output_image, timestamp_ms):
    global latest_result
    latest_result = result

# =============================
# MEDIAPIPE SETUP
# =============================
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=r"C:\Users\Acer\OneDrive\Desktop\Projects\handsign_ml\hand_landmarker.task"
    ),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands=1,
    result_callback=result_callback
)

# =============================
# DATA COUNTERS
# =============================
counts = {label: 0 for label in LABELS}

# =============================
# HAND CONNECTIONS (for drawing)
# =============================
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (0, 9), (9,10), (10,11), (11,12),      # Middle
    (0,13), (13,14), (14,15), (15,16),     # Ring
    (0,17), (17,18), (18,19), (19,20)      # Pinky
]

# =============================
# START CAMERA
# =============================
print("Starting camera...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot open camera")
    exit()

print("Camera started. Press A, B, or C to capture. ESC to quit.")

with HandLandmarker.create_from_options(options) as landmarker:
    frame_id = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Cannot read frame")
            break

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # Send frame for async processing
        frame_id += 1
        landmarker.detect_async(mp_image, frame_id)

        # -------------------------
        # DRAW LANDMARKS
        # -------------------------
        if latest_result and latest_result.hand_landmarks:
            for hand_landmarks in latest_result.hand_landmarks:
                # Draw points
                points = []
                for lm in hand_landmarks:
                    x = int(lm.x * w)
                    y = int(lm.y * h)
                    points.append((x, y))
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

                # Draw connections
                for start, end in HAND_CONNECTIONS:
                    cv2.line(frame, points[start], points[end], (255, 0, 0), 2)
            
            # Draw hand status
            cv2.putText(frame, "HAND DETECTED", (10, h - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "NO HAND DETECTED", (10, h - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # -------------------------
        # UI TEXT
        # -------------------------
        cv2.putText(frame, "Press A/B/C to capture | ESC to quit",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        y_offset = 60
        for label in LABELS:
            color = (0, 255, 0) if counts[label] >= SAMPLES_PER_LABEL else (0, 255, 255)
            cv2.putText(frame, f"{label}: {counts[label]}/{SAMPLES_PER_LABEL}",
                        (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            y_offset += 30

        cv2.imshow("Hand Data Collection", frame)

        # Wait for key press
        key = cv2.waitKey(1) & 0xFF

        # -------------------------
        # SAVE DATA
        # -------------------------
        if key == ord('a') or key == ord('A'):
            label = 'A'
            if latest_result and latest_result.hand_landmarks and counts[label] < SAMPLES_PER_LABEL:
                landmarks = []
                for lm in latest_result.hand_landmarks[0]:
                    landmarks.extend([lm.x, lm.y, lm.z])

                landmarks = np.array(landmarks, dtype=np.float32)

                split = TRAIN_DIR if random.random() < TRAIN_SPLIT else TEST_DIR
                filename = f"{label}_{counts[label]}.npy"
                save_path = os.path.join(split, label, filename)

                np.save(save_path, landmarks)
                counts[label] += 1
                print(f"Saved {save_path}")
            elif not latest_result or not latest_result.hand_landmarks:
                print("No hand detected! Cannot save.")
            else:
                print(f"Already collected {SAMPLES_PER_LABEL} samples for {label}")

        elif key == ord('b') or key == ord('B'):
            label = 'B'
            if latest_result and latest_result.hand_landmarks and counts[label] < SAMPLES_PER_LABEL:
                landmarks = []
                for lm in latest_result.hand_landmarks[0]:
                    landmarks.extend([lm.x, lm.y, lm.z])

                landmarks = np.array(landmarks, dtype=np.float32)

                split = TRAIN_DIR if random.random() < TRAIN_SPLIT else TEST_DIR
                filename = f"{label}_{counts[label]}.npy"
                save_path = os.path.join(split, label, filename)

                np.save(save_path, landmarks)
                counts[label] += 1
                print(f"Saved {save_path}")
            elif not latest_result or not latest_result.hand_landmarks:
                print("No hand detected! Cannot save.")
            else:
                print(f"Already collected {SAMPLES_PER_LABEL} samples for {label}")

        elif key == ord('c') or key == ord('C'):
            label = 'C'
            if latest_result and latest_result.hand_landmarks and counts[label] < SAMPLES_PER_LABEL:
                landmarks = []
                for lm in latest_result.hand_landmarks[0]:
                    landmarks.extend([lm.x, lm.y, lm.z])

                landmarks = np.array(landmarks, dtype=np.float32)

                split = TRAIN_DIR if random.random() < TRAIN_SPLIT else TEST_DIR
                filename = f"{label}_{counts[label]}.npy"
                save_path = os.path.join(split, label, filename)

                np.save(save_path, landmarks)
                counts[label] += 1
                print(f"Saved {save_path}")
            elif not latest_result or not latest_result.hand_landmarks:
                print("No hand detected! Cannot save.")
            else:
                print(f"Already collected {SAMPLES_PER_LABEL} samples for {label}")

        elif key == 27:  # ESC
            print("ESC pressed. Exiting...")
            break

        # Check if all samples collected
        if all(counts[l] >= SAMPLES_PER_LABEL for l in LABELS):
            print("Data collection complete!")
            break

cap.release()
cv2.destroyAllWindows()

print("\nFinal counts:")
for label in LABELS:
    print(f"{label}: {counts[label]}/{SAMPLES_PER_LABEL}")