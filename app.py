import cv2
import numpy as np
import joblib
from tkinter import Tk, filedialog
from skimage.feature import hog
import time

print("🚀 Loading model...")

# Load model, scaler, class names, accuracy
model, scaler, class_names, accuracy = joblib.load("models/model.pkl")

IMG_SIZE = 128


def detect_and_classify(img_path):
    img = cv2.imread(img_path)

    if img is None:
        print("❌ Could not read image")
        return

    original = img.copy()

    # =========================
    # 🔴 DETECTION (highlight)
    # =========================
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h > 1000:
            cv2.rectangle(original, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # =========================
    # 🧠 CLASSIFICATION
    # =========================
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Resize
    resized = cv2.resize(gray_img, (IMG_SIZE, IMG_SIZE))

    # Extract HOG features
    features = hog(
        resized,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        visualize=False
    ).reshape(1, -1)

    # Apply scaler
    features = scaler.transform(features)

    # Prediction
    pred = model.predict(features)[0]
    prob = model.predict_proba(features)[0]

    label = class_names[pred]
    confidence = np.max(prob) * 100

    # =========================
    # 🟢 PRINT RESULT
    # =========================
    print("\n✅ RESULT")
    print(f"Disease: {label}")
    print(f"Confidence: {confidence:.2f}")
    print(f"Model Accuracy: {accuracy:.2f}")

    # =========================
    # 🟢 DRAW TEXT
    # =========================
    text = f"{label} ({confidence:.2f})"

    cv2.putText(
        original,
        text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # =========================
    # 💾 SAVE UNIQUE OUTPUT
    # =========================
    timestamp = int(time.time())
    output_path = f"outputs/result_{timestamp}.jpg"

    cv2.imwrite(output_path, original)
    print(f"📁 Saved as: {output_path}")

    # =========================
    # 🖼️ DISPLAY LARGE IMAGE
    # =========================
    display_img = cv2.resize(original, (900, 700))
    cv2.imshow("Result", display_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def select_image():
    root = Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename(
        title="Select Skin Image",
        filetypes=[("Images", "*.jpg *.png *.jpeg")]
    )

    return file_path


# =========================
# 🚀 MAIN
# =========================
if __name__ == "__main__":
    path = select_image()

    if path:
        detect_and_classify(path)
    else:
        print("❌ No image selected")