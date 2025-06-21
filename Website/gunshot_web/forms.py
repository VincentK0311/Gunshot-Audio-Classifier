"""from django import forms

class AudioUploadForm(forms.Form):
    audio_file = forms.FileField(label="Upload Gunshot Audio (.wav)")
"""

from django import forms

MODEL_CHOICES = [
    ('cnn', 'CNN'),
    ('knn', 'KNN'),
    ('svm', 'SVM'),
]

class AudioUploadForm(forms.Form):
    audio_file = forms.FileField(label="Upload Gunshot Audio (.wav)")
    model_choice = forms.ChoiceField(choices=MODEL_CHOICES, label="Select Model Type")
