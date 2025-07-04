
import os
import numpy as np
import time
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf

from preprocess import load_dataset, MODEL_DIR, LOG_DIR

# ---------------------- CONFIGURATION ----------------------
MODEL_PATH = os.path.join(MODEL_DIR, "cnn_best_model.keras")

# ---------------------- LOGGING ----------------------
log_file = os.path.join(LOG_DIR, f"cnn_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
def log(message):
    print(message)
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"{message}\n")

# ---------------------- LOAD DATA ----------------------
log("🔄 Loading preprocessed dataset...")
start_time = time.time()
X, y, label_encoder = load_dataset()
X = X.reshape(X.shape[0], X.shape[1], 1)
y_class = np.argmax(y, axis=1)
log(f"⏱️ Data loading time: {time.time() - start_time:.2f} seconds")

# ---------------------- TRAIN/VALIDATION SPLIT ----------------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y_class, test_size=0.2, random_state=47, stratify=y_class)

# ---------------------- BUILD CNN MODEL ----------------------
def build_cnn_model(input_shape, num_filters=128, dropout_rate=0.3, optimizer='rmsprop'):
    model = Sequential([
        tf.keras.Input(shape=input_shape),
        
        # 1st Convolutional Block
        Conv1D(num_filters, 3, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(2),
        Dropout(dropout_rate),
        
        # 2nd Convolutional Block
        Conv1D(num_filters * 2, 3, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(2),
        Dropout(dropout_rate),
        
        # 3rd Convolutional Block
        Conv1D(num_filters * 4, 3, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(2),
        Dropout(dropout_rate),
        
        # Fully Connected Layers
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(len(label_encoder.classes_), activation='softmax')

    ])
    
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = build_cnn_model(input_shape=(X.shape[1], 1))

# ---------------------- CALLBACKS ----------------------
early_stop = EarlyStopping(monitor='val_accuracy', patience=100, restore_best_weights=True)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=50,
    min_lr=1e-6,
    verbose=1
)

# ---------------------- TRAIN MODEL ----------------------
log("🚀 Training CNN model...")
start_time = time.time()

history = model.fit(X_train, to_categorical(y_train),
                    validation_data=(X_val, to_categorical(y_val)),
                    epochs=500,
                    batch_size=32,
                    verbose=1,
                    callbacks=[early_stop, reduce_lr])

log(f"⏱️ Model training time: {time.time() - start_time:.2f} seconds")

# ---------------------- TRAINING HISTORY PLOTS ----------------------
log("📉 Plotting training history (combined figure)...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Training and Validation Accuracy
axes[0].plot(history.history['accuracy'], label='Training Accuracy', color='blue')
axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy', color='green')
axes[0].axhline(y=0.95, color='gray', linestyle='--', linewidth=1.2, label='Reference Line (95%)')
axes[0].set_title("CNN Training and Validation Accuracy", fontsize=14)
axes[0].set_xlabel("Epoch", fontsize=12)
axes[0].set_ylabel("Accuracy", fontsize=12)
axes[0].tick_params(labelsize=12)
axes[0].legend(fontsize=11)
axes[0].grid(True)

# Plot 2: Training and Validation Loss
axes[1].plot(history.history['loss'], label='Training Loss', color='blue')
axes[1].plot(history.history['val_loss'], label='Validation Loss', color='orange')
axes[1].set_title("CNN Training and Validation Loss", fontsize=14)
axes[1].set_xlabel("Epoch", fontsize=12)
axes[1].set_ylabel("Loss", fontsize=12)
axes[1].tick_params(labelsize=12)
axes[1].legend(fontsize=11)
axes[1].grid(True)

plt.tight_layout()
combined_plot_path = os.path.join(LOG_DIR, "cnn_training_history_combined.png")
plt.savefig(combined_plot_path, dpi=600)  # Save with higher DPI for report quality
plt.show()
plt.close()

log(f"🖼️ Combined training history plot saved to: {combined_plot_path}")

# ---------------------- FINAL EVALUATION ----------------------
log("\n📊 Final Accuracy Summary:")
start_time = time.time()
train_acc = accuracy_score(y_train, np.argmax(model.predict(X_train), axis=1))
val_acc = accuracy_score(y_val, np.argmax(model.predict(X_val), axis=1))
y_pred = np.argmax(model.predict(X_val), axis=1)

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
plt.title("CNN Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.xticks(rotation=45)
plt.tight_layout()
conf_matrix_path = os.path.join(LOG_DIR, "cnn_confusion_matrix.png")
plt.savefig(conf_matrix_path)
plt.show()
plt.close()
log(f"🖼️ Confusion matrix saved to: {conf_matrix_path}")

# ---------------------- SAVE MODEL ----------------------
model.save(MODEL_PATH)
log(f"💾 CNN model saved to: {MODEL_PATH}")
