import os  # للتعامل مع مسارات الملفات والمجلدات
import cv2  # لمعالجة الصور وقراءة الفيديوهات
import numpy as np  # للتعامل مع المصفوفات (Data Arrays)
import tensorflow as tf  # المكتبة الأساسية لبناء النماذج
# استيراد طبقات الشبكة العصبية من Keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical  # لتحويل التصنيفات إلى ترميز One-Hot
from sklearn.model_selection import train_test_split  # لتقسيم البيانات

# ============================
# 1) تحميل الداتا و تجهيزها
# ============================

DATASET_PATH = "dataset/"   # مسار المجلد الذي يحتوي على مجلدات fall و no_fall
IMG_SIZE = 128  # تحديد حجم الصورة (128x128)

def load_data():
    """تحميل الصور وتصنيفها من مجلد مجموعة البيانات."""
    X, y = [], []  # X للصور، y للتصنيفات
    classes = ["no_fall", "fall"] # تحديد أسماء الفئات

    for label, folder in enumerate(classes):
        folder_path = os.path.join(DATASET_PATH, folder) # بناء مسار المجلد (مثل dataset/fall)
        for img_name in os.listdir(folder_path): # المرور على كل ملف في المجلد
            img_path = os.path.join(folder_path, img_name) # بناء المسار الكامل للصورة
            img = cv2.imread(img_path) # قراءة الصورة باستخدام OpenCV

            if img is None: # تخطي الملفات غير الصالحة أو التي لا يمكن قراءتها
                continue

            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) # تغيير حجم الصورة ليناسب إدخال الموديل
            img = img / 255.0  # تطبيع قيم البكسل (Normalization) لتكون بين 0 و 1

            X.append(img) # إضافة الصورة إلى قائمة البيانات
            y.append(label) # إضافة التصنيف الرقمي (0 أو 1) إلى قائمة التصنيفات

    X = np.array(X) # تحويل قائمة الصور إلى مصفوفة NumPy
    y = to_categorical(y, num_classes=2) # تحويل التصنيفات إلى ترميز One-Hot (مثل [1, 0] و [0, 1])
    return X, y

print("[INFO] Loading dataset...")
X, y = load_data()

# ============================
# 2) تقسيم الداتا Train/Val/Test
# ============================

# تقسيم البيانات إلى تدريب (75%) ومؤقت (25%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.25, shuffle=True, random_state=42)
# تقسيم البيانات المؤقتة بالتساوي إلى التحقق (12.5%) والاختبار (12.5%)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, shuffle=True, random_state=42)

print(f"Train: {len(X_train)},  Val: {len(X_val)},  Test: {len(X_test)}")

# ============================
# 3) بناء المودل (النسخة المطورة)
# ============================

model = Sequential([
    # طبقة التلافيف الأولى: 32 مرشح، حجم نواة 3x3، تنشيط ReLU
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    BatchNormalization(), # إضافة Batch Normalization لتحسين التدريب
    MaxPooling2D(2,2), # تجميع لتقليل الأبعاد بمقدار النصف

    # طبقة التلافيف الثانية: 64 مرشح
    Conv2D(64, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),

    # طبقة التلافيف الثالثة: 128 مرشح
    Conv2D(128, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Flatten(), # تسوية المخرجات إلى متجه أحادي الأبعاد
    
    # الطبقة الكثيفة الأولى: 128 وحدة عصبية
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.5), # إسقاط 50% من الوحدات لمنع التكيف المفرط (Overfitting)
    
    # طبقة الإخراج: وحدتان (لأن لدينا تصنيفين: fall/no_fall) وتنشيط Softmax للتصنيف
    Dense(2, activation='softmax')
])

# تجميع الموديل: استخدام مُحسِّن Adam، دالة خسارة Crossentropy، ومقياس الدقة (Accuracy)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary() # طباعة ملخص هيكل الموديل

# ============================
# 4) تدريب المودل
# ============================

print("[INFO] Training model...")
# تدريب الموديل باستخدام بيانات التدريب والتحقق
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val), # استخدام بيانات التحقق لتقييم الأداء أثناء التدريب
    epochs=12, # عدد مرات المرور الكاملة على بيانات التدريب
    batch_size=32 # عدد العينات التي تتم معالجتها في كل تكرار
)

model.save("fall_model_improved.h5") # حفظ الموديل المدرب
print("[INFO] Model saved!")

# ============================
# 5) اختبار المودل Test
# ============================

# تقييم أداء الموديل على بيانات الاختبار غير المرئية مسبقاً
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"[TEST] Accuracy = {acc*100:.2f}%")

# ============================
# 6) تشغيل الفيديو و عمل Fall Detection
# ============================

print("[INFO] Starting video detection...")

cap = cv2.VideoCapture("vid4.mp4") # فتح ملف الفيديو

while True:
    ret, frame = cap.read() # قراءة إطار تلو الآخر من الفيديو
    if not ret: # إذا لم يتم قراءة الإطار (نهاية الفيديو)
        break

    # تجهيز الإطار للإدخال في الموديل:
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE)) # تغيير الحجم
    img = img / 255.0 # تطبيع القيم
    img = np.expand_dims(img, axis=0) # إضافة بُعد الدفعة (Batch dimension) لتصبح (1, 128, 128, 3)

    pred = model.predict(img, verbose=0)[0] # التنبؤ بالتصنيف
    cls = np.argmax(pred) # الحصول على فهرس الفئة الأعلى احتمالية (0 أو 1)

    label = "FALL" if cls == 1 else "NO FALL" # تحديد التسمية كنص

    # عرض النتيجة على الإطار
    cv2.putText(frame, label, (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0,0,255) if label=="FALL" else (0,255,0), 3) # وضع النص باللون الأحمر للسقوط والأخضر لغير السقوط

    cv2.imshow("Fall Detection", frame) # عرض الإطار

    if cv2.waitKey(1) & 0xFF == ord('q'): # إيقاف التشغيل عند الضغط على 'q'
        break

cap.release() # تحرير كائن الفيديو
cv2.destroyAllWindows() # إغلاق جميع النوافذ المفتوحة
