import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
import numpy as np

print(f"MediaPipe version: {mp.__version__}")

# -----------------------------
# GLOBAL RESULT STORAGE
# -----------------------------
latest_result = None

# -----------------------------
# CALLBACK FUNCTION
# -----------------------------
def print_result(result, output_image, timestamp_ms):
    global latest_result
    latest_result = result
    if result.hand_landmarks:
        print(f"Hands detected at timestamp {timestamp_ms}: {len(result.hand_landmarks)} hand(s)")

# -----------------------------
# HAND CONNECTIONS (for manual drawing)
# -----------------------------
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (0, 9), (9,10), (10,11), (11,12),      # Middle
    (0,13), (13,14), (14,15), (15,16),     # Ring
    (0,17), (17,18), (18,19), (19,20)      # Pinky
]

# -----------------------------
# MEDIAPIPE SETUP
# -----------------------------
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=r"C:\Users\Acer\OneDrive\Desktop\Projects\handsign_ml\hand_landmarker.task"
    ),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands=2,
    result_callback=print_result
)

# -----------------------------
# WEBCAM LOOP
# -----------------------------
with HandLandmarker.create_from_options(options) as landmarker:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Webcam not accessible")
        exit()

    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        frame_id += 1
        landmarker.detect_async(mp_image, frame_id)

        # -----------------------------
        # DRAW LANDMARKS MANUALLY (no external imports needed)
        # -----------------------------
        if latest_result and latest_result.hand_landmarks:
            for idx, hand_landmarks in enumerate(latest_result.hand_landmarks):
                # Draw points
                points = []
                for lm in hand_landmarks:
                    x = int(lm.x * w)
                    y = int(lm.y * h)
                    points.append((x, y))
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)  # Green dots

                # Draw connections
                for start, end in HAND_CONNECTIONS:
                    cv2.line(frame, points[start], points[end], (255, 0, 0), 2)  # Blue lines

                # Draw handedness text (if available)
                if latest_result.handedness and idx < len(latest_result.handedness):
                    handedness = latest_result.handedness[idx]
                    x_min = min([lm.x for lm in hand_landmarks]) * w
                    y_min = min([lm.y for lm in hand_landmarks]) * h
                    cv2.putText(frame, f"{handedness[0].category_name}",
                                (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_DUPLEX,
                                1, (88, 205, 54), 1, cv2.LINE_AA)

        cv2.imshow("LIVE Hand Landmarks", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()