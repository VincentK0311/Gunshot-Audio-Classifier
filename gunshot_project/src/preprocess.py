import os
import numpy as np
import librosa
import glob 
import joblib
import random
from datetime import datetime
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.utils import to_categorical
from collections import defaultdict
import time

# ---------------------- CONFIGURATION ----------------------
# on PC Set the base directory for your project
DATA_DIR = "C:\\Users\\19990\\Desktop\\FYP Project\\gunshot_project\\data\\Gunshot audio dataset"
PROCESSED_DATA_DIR = "C:\\Users\\19990\\Desktop\\FYP Project\\gunshot_project\\processed_data"
MODEL_DIR = "C:\\Users\\19990\\Desktop\\FYP Project\\gunshot_project\\models"
LOG_DIR = "C:\\Users\\19990\\Desktop\\FYP Project\\gunshot_project\\log"
SRC_DIR = SRC_DIR = "C:\\Users\\19990\\Desktop\\FYP Project\\gunshot_project\\src"#used for inference in Django app (views.py)

# Ensure directories exist
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Augmentation control
AUGMENTATION_ENABLED = True # Turn off if you want only original files

# Audio settings
SAMPLE_RATE = 44100     # Audio sampling rate
SEGMENT_DURATION = 2.0  # Segment length in seconds

log_file_path = os.path.join(LOG_DIR, f"preprocessing_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

def log_message(message):
    """Prints and saves log messages to file."""
    print(message)
    with open(log_file_path, 'a', encoding='utf-8') as f:
        f.write(f"{datetime.now()} - {message}\n")

# ---------------------- AUDIO PROCESSING ----------------------
def pad_to_duration(audio, sr, duration):
    """Pad or truncate the audio to match the specified duration."""
    target_len = int(sr * duration)
    if len(audio) < target_len:
        return np.pad(audio, (0, target_len - len(audio)))
    return audio[:target_len]

def add_gaussian_noise(audio, mean=0, var=0.01):
    """Add Gaussian noise to the audio signal using a fresh random generator each time."""
    rng = np.random.default_rng()  # Independent generator; not affected by np.random.seed()
    noise = rng.normal(mean, np.sqrt(var), len(audio))
    return audio + noise

# ---------------------- FEATURE EXTRACTION (108D) ----------------------
def extract_features(audio, sr):
    """
    Extract a 108-dimensional feature vector using:
    - MFCC (13 x 3)
    - Chroma (12 x 3)
    - Spectral contrast (7 x 3)
    - Spectral centroid, ZCR, bandwidth, energy (each x 3)
    """
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y=audio)
    energy = np.array([np.sum(audio ** 2)])
    bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)

    def summarize(feature):
        """Compute mean, std, median for a given feature array."""
        return [np.mean(feature), np.std(feature), np.median(feature)]

    features = []
    for f in [mfcc, chroma, contrast]:
        for row in f:
            features.extend(summarize(row))

    for f in [centroid, zcr, bandwidth]:
        features.extend(summarize(f[0]))

    features.extend(summarize(energy))  # energy is 1D array

    return np.array(features[:108])  # ensure it’s 108D

# ---------------------- FILE PROCESSING ----------------------
def get_gun_classes(data_dir):
    """Return list of gun class subdirectories."""
    return sorted([f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))])

def process_file(file_path, class_name):
    """
    Load and process an audio file:
    - Pad/truncate to duration
    - Optionally augment
    - Extract features
    - Save .npy feature file
    """
    results = []
    try:
        audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        audio = pad_to_duration(audio, sr, SEGMENT_DURATION)

        base_name = os.path.splitext(os.path.basename(file_path))[0]
        class_output_dir = os.path.join(PROCESSED_DATA_DIR, class_name)
        os.makedirs(class_output_dir, exist_ok=True)

        # Handle original and noisy variants
        variants = [(audio, "")]
        if AUGMENTATION_ENABLED:
            variants.append((add_gaussian_noise(audio), "_noisy"))

        for variant_audio, suffix in variants:
            stat_feat = extract_features(variant_audio, sr)

            # Save feature
            np.save(os.path.join(class_output_dir, f"{base_name}{suffix}_feat.npy"), stat_feat)

            results.append((stat_feat, class_name))

        log_message(f"[✓] {class_name} - {base_name}.wav processed")
        return results

    except Exception as e:
        log_message(f"[!] Failed to process {os.path.basename(file_path)}: {e}")
        return []

# ---------------------- MAIN PIPELINE ----------------------
def run_pipeline():
    """Main preprocessing pipeline that:
    - Reads and processes audio files
    - Extracts 108D features
    - Applies label encoding and normalization
    - Saves encoders and returns processed dataset
    """
    start_time = time.time()
    log_message("🚀 Starting article-compliant preprocessing pipeline...")

    gun_classes = get_gun_classes(DATA_DIR)
    feat_list, label_list = [], []
    total_files = defaultdict(int)

    label_encoder = LabelEncoder()
    label_encoder.fit(gun_classes)

    # Process each file by class
    for class_name in gun_classes:
        class_path = os.path.join(DATA_DIR, class_name)
        for file in os.listdir(class_path):
            if file.endswith(".wav"):
                file_path = os.path.join(class_path, file)
                results = process_file(file_path, class_name)
                total_files[class_name] += len(results)
                for stat_feat, label in results:
                    feat_list.append(stat_feat)
                    label_list.append(label)

     # Label encoding + one-hot encoding
    encoded_labels = label_encoder.transform(label_list)
    categorical_labels = to_categorical(encoded_labels)

    # Convert to numpy arrays
    X_feat = np.array(feat_list, dtype=np.float32)
    y = categorical_labels

    # Normalize 108D features (shared input for all models)
    scaler = StandardScaler()
    X_feat_scaled = scaler.fit_transform(X_feat)

    # Save encoders for later use
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.pkl'))
    joblib.dump(label_encoder, os.path.join(MODEL_DIR, 'label_encoder.pkl'))

    # Logging dataset stats
    log_message("📊 File counts per class:")
    for c, count in total_files.items():
        log_message(f"   {c}: {count} samples")

    log_message(f"📦 Classes: {label_encoder.classes_}")
    log_message(f"📊 Feature shape (ALL models): {X_feat_scaled.shape}")
    log_message(f"🔖 Labels shape: {y.shape}")
    log_message(f"⏱️ Total Time: {time.time() - start_time:.2f} sec")
    log_message("✅ Preprocessing finished.")

    return X_feat_scaled, y, label_encoder

# ---------------------- LOAD DATASET ----------------------
def load_dataset(data_dir=PROCESSED_DATA_DIR):
    """
    Loads existing *_feat.npy or *_mfcc.npy files. If missing, triggers preprocessing.
    """
    # Check for either type of saved feature file
    feat_files = glob.glob(os.path.join(data_dir, '**', '*_feat.npy'), recursive=True)
    mfcc_files = glob.glob(os.path.join(data_dir, '**', '*_mfcc.npy'), recursive=True)

    if not feat_files and not mfcc_files:
        print("⚠️ No .npy feature files found. Running preprocessing...")
        return run_pipeline()

    print("📥 Found existing preprocessed features. Loading...")

    feat_list, label_list = [], []
    gun_classes = get_gun_classes(data_dir)

    label_encoder_path = os.path.join(MODEL_DIR, 'label_encoder.pkl')
    if not os.path.exists(label_encoder_path):
        raise FileNotFoundError("❌ Label encoder not found. Please rerun the preprocessing pipeline.")

    label_encoder = joblib.load(label_encoder_path)

    for class_name in gun_classes:
        class_dir = os.path.join(data_dir, class_name)
        #for file in os.listdir(class_dir):
        for file in sorted(os.listdir(class_dir)):    
            if file.endswith("_feat.npy") or file.endswith("_mfcc.npy"):
                try:
                    feat = np.load(os.path.join(class_dir, file))
                    feat_list.append(feat)
                    label_list.append(class_name)
                except Exception as e:
                    log_message(f"[!] Failed to load {file}: {e}")

    encoded_labels = label_encoder.transform(label_list)
    categorical_labels = to_categorical(encoded_labels)

    scaler_path = os.path.join(MODEL_DIR, 'scaler.pkl')
    scaler = joblib.load(scaler_path)

    X = scaler.transform(np.array(feat_list, dtype=np.float32))
    y = categorical_labels

    print("✅ Dataset loaded successfully.")
    print(f"📊 Features shape: {X.shape}")
    print(f"🔖 Labels shape: {y.shape}")
    return X, y, label_encoder

# ----- FOR INFERENCE (e.g., Django App) -----
def extract_features_from_file(file_path, scaler_path, encoder_path):
    """
    Extracts 108D features from an audio file using the same preprocessing
    and feature extraction steps used during training.
    """
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    audio = pad_to_duration(audio, sr, SEGMENT_DURATION)

    # Extract features
    stat_feat = extract_features(audio, sr)

    # Load scaler and label encoder (if needed for inverse transform or confidence display)
    scaler = joblib.load(scaler_path)
    label_encoder = joblib.load(encoder_path)

    # Normalize the feature
    stat_feat_scaled = scaler.transform([stat_feat])  # Keep batch shape (1, 108)

    return stat_feat_scaled, label_encoder


# ---------------------- MAIN ENTRY ----------------------
if __name__ == "__main__":
    run_pipeline()
