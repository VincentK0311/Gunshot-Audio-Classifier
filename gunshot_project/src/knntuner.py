import os
import joblib
import time
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns

from preprocess import load_dataset, MODEL_DIR, LOG_DIR

# ---------------------- CONFIGURATION ----------------------
MODEL_PATH = os.path.join(MODEL_DIR, "knn_best_tuned_model.pkl")
log_file = os.path.join(LOG_DIR, f"knn_grid_search_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

def log(message):
    print(message)
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"{message}\n")

# ---------------------- LOAD DATA ----------------------
log("🔄 Loading preprocessed dataset...")
X, y, label_encoder = load_dataset()
y_class = np.argmax(y, axis=1)

# ---------------------- TRAIN/VALIDATION SPLIT ----------------------
seed = 42
X_train, X_val, y_train, y_val = train_test_split(
    X, y_class, test_size=0.2, random_state=seed, stratify=y_class
)

# ---------------------- HYPERPARAMETER TUNING ----------------------
param_grid = {
    'n_neighbors': [1, 3, 5, 7],
    'weights': ['uniform', 'distance'],
    'metric': ['manhattan', 'euclidean', 'cosine']
}

log("🚀 Performing GridSearchCV on training data...")
start_time = time.time()
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
grid = GridSearchCV(
    KNeighborsClassifier(),
    param_grid,
    cv=cv,
    scoring='accuracy',
    verbose=2,
    n_jobs=-1
)
grid.fit(X_train, y_train)
elapsed_time = time.time() - start_time
log(f"⏱️ Total search time: {elapsed_time:.2f} seconds")

# ---------------------- EVALUATION ----------------------
best_model = grid.best_estimator_
val_preds = best_model.predict(X_val)
train_acc = accuracy_score(y_train, best_model.predict(X_train))
val_acc = accuracy_score(y_val, val_preds)

log(f"\n✅ Best Parameters: {grid.best_params_}")
log(f"📈 Training Accuracy: {train_acc * 100:.2f}%")
log(f"📈 Testing Accuracy: {val_acc * 100:.2f}%")

log("\n📋 Classification Report:")
report = classification_report(y_val, val_preds, target_names=label_encoder.classes_)
log(report)

# ---------------------- SAVE MODEL ----------------------
joblib.dump(best_model, MODEL_PATH)
log(f"💾 Best tuned model saved to: {MODEL_PATH}")
log("✅ Hyperparameter tuning and evaluation complete.")
