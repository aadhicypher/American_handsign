import os
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

# =============================
# CONFIG
# =============================
DATASET_DIR = "dataset"
LABELS = {"A": 0, "B": 1, "C": 2}

# =============================
# LOAD DATA
# =============================
def load_data(split):
    X, y = [], []
    split_path = os.path.join(DATASET_DIR, split)

    for label_name, label_id in LABELS.items():
        label_dir = os.path.join(split_path, label_name)
        for file in os.listdir(label_dir):
            if file.endswith(".npy"):
                data = np.load(os.path.join(label_dir, file))
                X.append(data)
                y.append(label_id)

    return np.array(X), np.array(y)

print("Loading data...")
X_train, y_train = load_data("train")
X_test, y_test = load_data("test")

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

# =============================
# MODEL PIPELINE
# =============================
model = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="rbf", C=10, gamma="scale"))
])

# =============================
# TRAIN
# =============================
print("Training model...")
model.fit(X_train, y_train)

# =============================
# EVALUATE
# =============================
y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=LABELS.keys()))

# =============================
# SAVE MODEL
# =============================
joblib.dump(model, "hand_sign_model.pkl")
print("\nModel saved as hand_sign_model.pkl")
