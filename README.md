 🤟 HandSign_ML
**Hand Sign Recognition using Support Vector Machine (SVM)**

---

## 📌 Introduction
HandSign_ML is a machine learning–based project designed to recognize **static hand sign gestures** using **hand landmark features** and a **Support Vector Machine (SVM)** classifier.  
The system extracts key hand landmarks from images, converts them into numerical features, and classifies them into predefined hand sign categories.

This project demonstrates the practical use of:
- Computer Vision
- Feature Extraction
- Classical Machine Learning algorithms

---

## 🎯 Objectives
- Detect hand landmarks from images
- Extract meaningful features from landmarks
- Train an SVM classifier for hand sign recognition
- Test the trained model on unseen data

---

## 🧠 Methodology
1. **Hand Detection & Landmark Extraction**
   - Hand landmarks are detected using MediaPipe.
   - Each hand is represented using multiple landmark points.

2. **Feature Engineering**
   - Landmark coordinates (x, y) are normalized.
   - The coordinates are flattened into a feature vector.

3. **Model Training**
   - A Support Vector Machine (SVM) classifier is trained using the extracted features.
   - Hyperparameters are tuned for better performance.

4. **Prediction**
   - The trained model predicts the hand sign for new input images.

---

## 📂 Project Structure
handsign_ml
|_dataset
|     |_test
|     |_train
|_scripts
|     |_run_inference.py
|     |_handpipe.py
|     |_collect_data.py
|     |_train_model.py
|_handlandmarker.task
|_hand_sign_model.pkl



---

## 🛠️ Tools & Technologies
- **Programming Language:** Python  
- **Libraries:**
  - NumPy
  - OpenCV
  - scikit-learn
  - MediaPipe
- **Algorithm:** Support Vector Machine (SVM)

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/HandSign_ML.git
cd HandSign_ML
---
### Install requirements
pip install -r requirements.txt

---
## To train the model
python train_model.py

### To run the model
python run_inference.py


