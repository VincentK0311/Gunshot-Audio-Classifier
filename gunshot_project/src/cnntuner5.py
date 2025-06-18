import os
import numpy as np
import time
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from scikeras.wrappers import KerasClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from preprocess import load_dataset, MODEL_DIR, LOG_DIR

# ---------------------- CONFIGURATION ----------------------
MODEL_PATH = os.path.join(MODEL_DIR, "cnn_best_model.keras")

# ---------------------- LOGGING ----------------------
log_file = os.path.join(LOG_DIR, f"cnntuner_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
def log(message):
    print(message)
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"{message}\n")

# ---------------------- LOAD DATA ----------------------
log("🔄 Loading preprocessed dataset...")
start_time = time.time()
X, y, label_encoder = load_dataset()
X = X.reshape(X.shape[0], X.shape[1], 1)

# Extract integer class labels for StratifiedKFold
y_labels = np.argmax(y, axis=1)  # integer class labels (0, 1, 2, ..., n_classes-1)

# ---------------------- TRAIN/VALIDATION SPLIT ----------------------
log("🔀 Splitting data into 80% train and 20% validation sets (stratified)...")
X_train, X_val, y_train, y_val = train_test_split(
    X, y_labels, test_size=0.2, random_state=42, stratify=y_labels  # <-- stratify to preserve class distribution
)

log(f"⏱️ Data loading time: {time.time() - start_time:.2f} seconds")

# ---------------------- BUILD DYNAMIC CNN MODEL ----------------------
def build_cnn_model(optimizer='rmsprop', dropout_rate=0.3, num_filters=32, num_conv_layers=3, **kwargs):
    model = Sequential()
    model.add(tf.keras.Input(shape=(X.shape[1], 1)))

    max_filters = 512  # Cap to avoid memory explosion
    
    for i in range(num_conv_layers):
        filters = min(num_filters * (2 ** i), max_filters)
        model.add(Conv1D(filters, 3, activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(2))
        model.add(Dropout(dropout_rate))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(len(label_encoder.classes_), activation='softmax'))

    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# ---------------------- WRAP WITH SCIKERAS WRAPPER ----------------------
wrapped_model = KerasClassifier(
    model=build_cnn_model,
    optimizer='rmsprop',
    dropout_rate=0.3,
    num_filters=32,
    num_conv_layers=3,  # Default, GridSearch will override
    verbose=0,
    target_type="int"
)

# ---------------------- GRID SEARCH SETUP ----------------------
param_grid = {
    'optimizer': ['rmsprop', 'adam'],
    'dropout_rate': [0.3, 0.5],
    'num_filters': [32, 64, 128],
    'num_conv_layers': [2, 3, 4, 5, 6],  # <-- Dynamic 2 to 6 layers
    'fit__batch_size': [16, 32],
    'fit__epochs': [50]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid = GridSearchCV(estimator=wrapped_model,
                    param_grid=param_grid,
                    cv=cv,
                    scoring='accuracy',
                    verbose=2,
                    n_jobs=1)

# ---------------------- START GRID SEARCH ----------------------
log("🚀 Starting GridSearchCV with 5-Fold Cross-Validation...")
start_time = time.time()
grid_result = grid.fit(X_train, y_train)
log(f"⏱️ Total search time: {time.time() - start_time:.2f} seconds")

# ---------------------- GRID SEARCH RESULTS ----------------------
log(f"Best Parameters: {grid_result.best_params_}")
log(f"Best Cross-Validation Accuracy: {grid_result.best_score_ * 100:.2f}%")

# ---------------------- FINAL EVALUATION ON ALL DATA ----------------------
log("\n📊 Final Evaluation on Validation Set with Best Model:")
best_model = grid_result.best_estimator_.model_

y_pred = np.argmax(best_model.predict(X_val), axis=1)  # <-- Predict on validation set
y_true = y_val  # <-- Validation true labels

accuracy = accuracy_score(y_true, y_pred)
log(f"✅ Accuracy on entire dataset: {accuracy * 100:.2f}%")

log("\n📋 Classification Report:")
report = classification_report(y_true, y_pred, target_names=label_encoder.classes_)
log(report)

# ---------------------- SAVE MODEL ----------------------
grid_result.best_estimator_.model_.save(MODEL_PATH)
log(f"💾 CNN best model saved to: {MODEL_PATH}")
