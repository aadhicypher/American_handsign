import cv2
import mediapipe as mp
import numpy as np
import joblib
import os

# =============================
# LOAD TRAINED MODEL
# =============================
import joblib

MODEL_PATH = "hand_sign_model.pkl"

if not os.path.exists(MODEL_PATH):
    print(f"Error: Model file '{MODEL_PATH}' not found!")
    print("Please train the model first using train_model.py")
    exit()

print("Loading model...")
model = joblib.load(MODEL_PATH)
print("Model loaded successfully!")

# =============================
# GLOBAL RESULT STORAGE
# =============================
latest_result = None
predicted_label = None
confidence = None

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
print("Show hand signs A, B, or C to the camera")
print("Press ESC to quit")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot open camera")
    exit()

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
        # PREDICT IF HAND DETECTED
        # -------------------------
        if latest_result and latest_result.hand_landmarks:
            # Extract landmarks
            landmarks = []
            for lm in latest_result.hand_landmarks[0]:
                landmarks.extend([lm.x, lm.y, lm.z])
            
            landmarks = np.array(landmarks, dtype=np.float32).reshape(1, -1)
            
            # Make prediction
            predicted_label = model.predict(landmarks)[0]
            
            # Get prediction probabilities for confidence
            try:
                probabilities = model.predict_proba(landmarks)[0]
                confidence = max(probabilities) * 100
            except:
                confidence = 100.0  # If model doesn't support predict_proba
            
            # -------------------------
            # DRAW LANDMARKS
            # -------------------------
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
            
            # -------------------------
            # DISPLAY PREDICTION
            # -------------------------
            # Large prediction text
            text = f"Sign: {predicted_label}"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 2, 3)[0]
            text_x = (w - text_size[0]) // 2
            text_y = 80
            
            # Draw background rectangle for text
            cv2.rectangle(frame, 
                         (text_x - 10, text_y - text_size[1] - 10),
                         (text_x + text_size[0] + 10, text_y + 10),
                         (0, 0, 0), -1)
            
            # Draw prediction text
            cv2.putText(frame, text, (text_x, text_y),
                       cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 0), 3)
            
            # Draw confidence
            conf_text = f"Confidence: {confidence:.1f}%"
            cv2.putText(frame, conf_text, (text_x, text_y + 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Status text
            cv2.putText(frame, "HAND DETECTED", (10, h - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            # No hand detected
            cv2.putText(frame, "NO HAND DETECTED", (10, h - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "Show your hand sign to the camera", 
                       (w//2 - 250, h//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # -------------------------
        # UI TEXT
        # -------------------------
        cv2.putText(frame, "Hand Sign Recognition | ESC to quit",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Hand Sign Recognition", frame)

        # ESC to quit
        if cv2.waitKey(1) & 0xFF == 27:
            print("Exiting...")
            break

cap.release()
cv2.destroyAllWindows()
print("Program terminated.")