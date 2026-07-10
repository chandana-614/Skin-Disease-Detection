import os
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
import joblib

IMG_SIZE = 128

train_path = "dataset/train"
test_path = "dataset/test"

def load_data(path):
    data = []
    labels = []
    class_names = sorted(os.listdir(path))

    for label, folder in enumerate(class_names):
        folder_path = os.path.join(path, folder)

        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

            features = hog(img,
                           orientations=9,
                           pixels_per_cell=(8, 8),
                           cells_per_block=(2, 2),
                           visualize=False)

            data.append(features)
            labels.append(label)

    return np.array(data), np.array(labels), class_names


print("📂 Loading data...")
X_train, y_train, class_names = load_data(train_path)
X_test, y_test, _ = load_data(test_path)

# 🔥 SCALE FEATURES (VERY IMPORTANT)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("🧠 Training model...")

# 🔥 BALANCE CLASSES
model = SVC(kernel='rbf', probability=True, class_weight='balanced')

model.fit(X_train, y_train)

print("📊 Evaluating...")
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred) * 100
print(f"🔥 Accuracy: {accuracy:.2f}")

# Save model + scaler
joblib.dump((model, scaler, class_names, accuracy), "models/model.pkl")

print("✅ Model saved!")