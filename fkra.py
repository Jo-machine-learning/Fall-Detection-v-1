import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# ============================
# 1) تحميل الداتا و تجهيزها
# ============================

DATASET_PATH = "dataset/"   # جواها فولدرين: fall / no_fall
IMG_SIZE = 128

def load_data():
    X, y = [], []
    classes = ["no_fall", "fall"]

    for label, folder in enumerate(classes):
        folder_path = os.path.join(DATASET_PATH, folder)
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            img = cv2.imread(img_path)

            if img is None:
                continue

            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img / 255.0

            X.append(img)
            y.append(label)

    X = np.array(X)
    y = to_categorical(y, num_classes=2)
    return X, y

print("[INFO] Loading dataset...")
X, y = load_data()

# ============================
# 2) تقسيم الداتا Train/Val/Test
# ============================

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.25, shuffle=True)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, shuffle=True)

print(f"Train: {len(X_train)},  Val: {len(X_val)},  Test: {len(X_test)}")

# ============================
# 3) بناء المودل
# ============================

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# ============================
# 4) تدريب المودل
# ============================

print("[INFO] Training model...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=12,
    batch_size=32
)

model.save("fall_model.h5")
print("[INFO] Model saved!")

# ============================
# 5) اختبار المودل Test
# ============================

loss, acc = model.evaluate(X_test, y_test)
print(f"[TEST] Accuracy = {acc*100:.2f}%")

# ============================
# 6) تشغيل الفيديو و عمل Fall Detection
# ============================

print("[INFO] Starting video detection...")

cap = cv2.VideoCapture("vid4.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)[0]
    cls = np.argmax(pred)

    label = "FALL" if cls == 1 else "NO FALL"

    cv2.putText(frame, label, (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0,0,255) if label=="FALL" else (0,255,0), 3)

    cv2.imshow("Fall Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()