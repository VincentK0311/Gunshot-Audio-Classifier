import os
import sys
import time
import numpy as np
import joblib
from django.conf import settings
from django.shortcuts import render
from .forms import AudioUploadForm
from tensorflow.keras.models import load_model

# ----- Setup src path -----
SRC_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'gunshot_project', 'src'))
sys.path.append(SRC_PATH)

# ----- Import feature extraction -----
# Ensure preprocess.py is in the same directory as this script or adjust the import accordingly
from preprocess import SRC_DIR, extract_features_from_file

# ----- Load artifacts -----
BASE_MODEL_DIR = os.path.abspath(os.path.join(settings.BASE_DIR, '..', 'gunshot_project', 'models'))

label_encoder = joblib.load(os.path.join(BASE_MODEL_DIR, 'label_encoder.pkl'))
scaler = joblib.load(os.path.join(BASE_MODEL_DIR, 'scaler.pkl'))

svm_model = joblib.load(os.path.join(BASE_MODEL_DIR, 'svm_best_model.pkl'))
knn_model = joblib.load(os.path.join(BASE_MODEL_DIR, 'knn_best_model.pkl'))
cnn_model = load_model(os.path.join(BASE_MODEL_DIR, 'cnn_best_model.keras'))

# Display name mapping
model_display_names = {
    'cnn': 'CNN',
    'knn': 'KNN',
    'svm': 'SVM',
}

# ----- Get Probabilities for Confidence Table -----
def get_model_confidences(model, input_data, model_type):
    if model_type == 'cnn':
        input_reshaped = np.reshape(input_data, (1, 108, 1))
        probs = model.predict(input_reshaped, verbose=0)[0]
    else:
        probs = model.predict_proba([input_data])[0]
    return probs  # numpy array of probabilities

# ----- Main View -----
def upload_audio(request):
    if request.method == 'POST':
        form = AudioUploadForm(request.POST, request.FILES)
        if form.is_valid():
            start_time = time.time()
            audio = form.cleaned_data['audio_file']
            model_choice = form.cleaned_data['model_choice']

            try:
                # Save the file
                os.makedirs(settings.MEDIA_ROOT, exist_ok=True)
                save_path = os.path.join(settings.MEDIA_ROOT, audio.name)
                with open(save_path, 'wb+') as f:
                    for chunk in audio.chunks():
                        f.write(chunk)

                print("📁 Audio file saved to:", save_path)
                print("🧠 Model selected:", model_choice)

                # Extract features
                features_scaled, _ = extract_features_from_file(
                    save_path,
                    os.path.join(BASE_MODEL_DIR, 'scaler.pkl'),
                    os.path.join(BASE_MODEL_DIR, 'label_encoder.pkl')
                )
                features_scaled = features_scaled[0]
                print("📊 Feature shape:", features_scaled.shape)

                gun_classes = label_encoder.classes_

                # Prediction based on model choice
                confidence_table = []

                if model_choice in ['svm', 'knn', 'cnn']:
                    models = {
                        'svm': svm_model,
                        'knn': knn_model,
                        'cnn': cnn_model
                    }
                    model = models.get(model_choice)
                    probs = get_model_confidences(model, features_scaled, model_choice)
                    final_index = np.argmax(probs)

                    # Prepare confidence table
                    for idx, gun_type in enumerate(gun_classes):
                        confidence_table.append({
                            'gun_type': gun_type,
                            model_choice: f"{probs[idx]*100:.2f}%"
                        })

                else:
                    raise ValueError("Invalid model choice.")

                # Decode result
                result = label_encoder.inverse_transform([final_index])[0]

                # Map class names to image filenames
                gun_image_map = {
                    "AK-12": "ak_12.jpg",
                    "AK-47": "ak_47.jpg",
                    "IMI Desert Eagle": "imi_desert_eagle.jpg",
                    "M16": "m16.jpg",
                    "M249": "m249.jpg",
                    "MG-42": "mg_42.jpg",
                    "MP5": "mp5.jpg",
                    "Zastava M92": "zastava_m92.jpg",
                }

                # Get image path based on prediction
                image_file = gun_image_map.get(result, "default.jpg")
                gun_image_url = f"gun_images/{image_file}"

                elapsed = f"{time.time() - start_time:.2f} seconds"

                return render(request, 'result.html', {
                    'result': result,
                    'file_name': audio.name,
                    'model_choice': model_choice,
                    'model_display_name': model_display_names.get(model_choice, model_choice),  # 🟢 Pretty name
                    'elapsed': elapsed,
                    'file_url': settings.MEDIA_URL + audio.name,
                    'gun_image_url': gun_image_url,
                    'confidence_table': confidence_table,
                })

            except Exception as e:
                print("❌ Error occurred during processing:", e)
                return render(request, 'main.html', {
                    'form': form,
                    'result': f"⚠️ Error during processing: {str(e)}",
                })

    else:
        form = AudioUploadForm()

    return render(request, 'main.html', {
        'form': form,
    })
