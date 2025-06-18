import os
import joblib
import time
import numpy as np
from datetime import datetime
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from preprocess import load_dataset, MODEL_DIR, LOG_DIR
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------- CONFIGURATION ----------------------
MODEL_PATH = os.path.join(MODEL_DIR, "svm_best_model.pkl")

# ---------------------- LOGGING ----------------------
log_file = os.path.join(LOG_DIR, f"svm_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
def log(message):
    print(message)
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"{message}\n")

# ---------------------- LOAD DATA ----------------------
log("🔄 Loading preprocessed dataset...")
start_time = time.time()
X, y, label_encoder = load_dataset()
y_class = np.argmax(y, axis=1)
log(f"⏱️ Data loading time: {time.time() - start_time:.2f} seconds")

# ---------------------- TRAIN/VALIDATION SPLIT ----------------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y_class, test_size=0.2, stratify=y_class)

# ---------------------- DEFINE & TRAIN FINAL MODEL ----------------------
log("🚀 Training final SVM model with best hyperparameters...")
start_time = time.time()
final_model = SVC(
    C=10,
    gamma=0.01,
    kernel='rbf',
    decision_function_shape='ovo',
    probability=True
)
final_model.fit(X_train, y_train)
log(f"⏱️ Model training time: {time.time() - start_time:.2f} seconds")

# ---------------------- FINAL EVALUATION ----------------------
start_time = time.time()
train_acc = accuracy_score(y_train, final_model.predict(X_train))
val_acc = accuracy_score(y_val, final_model.predict(X_val))
y_pred = final_model.predict(X_val)

log("\n📊 Final Accuracy Summary:")
log(f"✅ Training Accuracy: {train_acc * 100:.2f}%")
log(f"✅ Testing Accuracy: {val_acc * 100:.2f}%")

log("\n📋 Classification Report:")
report = classification_report(y_val, y_pred, target_names=label_encoder.classes_)
log(report)
log(f"⏱️ Evaluation time: {time.time() - start_time:.2f} seconds")

# ---------------------- CONFUSION MATRIX ----------------------
cm = confusion_matrix(y_val, y_pred, normalize='true')
plt.figure(figsize=(10, 8))
sns.heatmap(cm, 
            annot=True, 
            fmt=".2f", 
            xticklabels=label_encoder.classes_, 
            yticklabels=label_encoder.classes_,
            cmap='Blues')
plt.title("SVM Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.xticks(rotation=45)
plt.tight_layout()
conf_matrix_path = os.path.join(LOG_DIR, "svm_confusion_matrix.png")
plt.savefig(conf_matrix_path)
plt.show()
log(f"🖼️ Confusion matrix saved to: {conf_matrix_path}")

# ---------------------- SAVE MODEL ----------------------
joblib.dump(final_model, MODEL_PATH)
log(f"💾 SVM model saved to: {MODEL_PATH}")
