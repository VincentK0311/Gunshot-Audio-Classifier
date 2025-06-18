import os
import joblib
import numpy as np
import time
from datetime import datetime
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from preprocess import load_dataset, MODEL_DIR, LOG_DIR
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------- CONFIGURATION ----------------------
MODEL_PATH = os.path.join(MODEL_DIR, "tuner_svm_best_model.pkl")

# ---------------------- LOGGING ----------------------
log_file = os.path.join(LOG_DIR, f"svm_tunner_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
def log(message):
    print(message)
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"{message}\n")

# ---------------------- LOAD DATA ----------------------
log("🔄 Loading preprocessed dataset...")
X, y, label_encoder = load_dataset()
y_class = np.argmax(y, axis=1)

# ---------------------- TRAIN/VALIDATION SPLIT ----------------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y_class, test_size=0.2, random_state=42, stratify=y_class)

# ---------------------- HYPERPARAMETER TUNING ----------------------
log("🔍 Starting SVM hyperparameter tuning with 5-fold cross-validation...")
start_time = time.time()
param_grid = {
    'C': [1, 10, 100],
    'gamma': [0.1, 0.01, 0.001],
    'kernel': ['rbf', 'linear', 'poly'],
    'decision_function_shape': ['ovo', 'ovr'] 
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid = GridSearchCV(SVC(), param_grid, cv=cv, scoring='accuracy', verbose=2, n_jobs=-1)
grid.fit(X_train, y_train)
elapsed_time = time.time() - start_time
log(f"⏱️ Total search time: {elapsed_time:.2f} seconds")
log(f"\n✅ Best Parameters: {grid.best_params_}")
log(f"📈 Cross-Validation Accuracy (on training data): {grid.best_score_ * 100:.2f}%")

# ---------------------- RETRAIN FINAL MODEL ON FULL TRAINING SET ----------------------
best_params = grid.best_params_

final_model = SVC(
    C=best_params['C'],
    gamma=best_params['gamma'],
    kernel=best_params['kernel'],
    decision_function_shape=best_params['decision_function_shape']
)
final_model.fit(X_train, y_train)

# ---------------------- FINAL EVALUATION ----------------------
train_acc = accuracy_score(y_train, final_model.predict(X_train))
val_acc = accuracy_score(y_val, final_model.predict(X_val))
y_pred = final_model.predict(X_val)

log("\n📊 Final Accuracy Summary:")
log(f"✅ Cross-Validation Accuracy (on training data): {grid.best_score_ * 100:.2f}%")
log(f"✅ Training Accuracy (on full training set): {train_acc * 100:.2f}%")
log(f"✅ Validation Accuracy (on hold-out set): {val_acc * 100:.2f}%")

log("\n📋 Classification Report:")
report = classification_report(y_val, y_pred, target_names=label_encoder.classes_)
log(report)

# ---------------------- SAVE MODEL ----------------------
joblib.dump(final_model, MODEL_PATH)
log(f"💾 Best SVM model saved to: {MODEL_PATH}")
# ---------------------- END ----------------------