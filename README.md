# Gunshot-Audio-Classifier
FYP project for firearm classification using gunshot audio
Gunshot Audio Classifier

This is a Final Year Project (FYP) that detects and classifies firearm types based on gunshot audio recordings. 
The system uses machine learning models including Support Vector Machine (SVM), k-Nearest Neighbors (kNN), and 
Convolutional Neural Network (CNN). It also includes a Django-based web application for uploading audio and displaying predictions.

**Project Description**
The system processes short (2 seconds) .wav audio files of gunshots and extracts features such as 
MFCC, chroma, spectral contrast, zero crossing rate, spectral centroid, bandwidth, and energy. 
These features are used to train models that classify the firearm type. 
The preprocessing step also includes Gaussian noise augmentation and feature normalization.

The trained models (.pkl and .keras) and preprocessed feature files (.npy) are already included in this project repository. 
You can directly run the Django web application to test the classification functionality without needing to rerun the training or preprocessing steps.

**Folder Structure**
gunshot_project - contains dataset folder, preprocessing script, model training code, and saved models  
gunshot_sound_classification - Python virtual environment folder (not uploaded to GitHub)  
Website - Django web application  
requirements.txt - list of Python dependencies  
README.md - project instructions  
LICENSE, .gitignore, .gitattributes - configuration files  

**Dataset**
The dataset used is from Kaggle:  
Gunshot Audio Dataset by Emrah AYDEMİR  
Link: https://www.kaggle.com/datasets/emrahaydemr/gunshot-audio-dataset

After downloading the dataset, go to the path below and open preprocess.py to update it with your own local directory:

Gunshot-Audio-Classifier-main.zip\Gunshot-Audio-Classifier-main\gunshot_project\src

**How to Update File Paths**
Open preprocess.py and change the following variables to match your folder location:
DATA_DIR  
PROCESSED_DATA_DIR  
MODEL_DIR  
LOG_DIR  
SRC_DIR  

**Setup Instructions (PowerShell on Windows)**
1. Create and activate a virtual environment
cd "your path\FYP Project"  
python -m venv gunshot_sound_classification  
.\gunshot_sound_classification\Scripts\activate  

2. Install required libraries
pip install -r requirements.txt  

**Note:
This project already includes the trained models and preprocessed features.
You do not need to rerun preprocess.py or retrain the models.
You can directly run the web application.**

**Running the Web Application**

cd "your paths\FYP Project"  
.\gunshot_sound_classification\Scripts\activate  
cd Website  
python manage.py migrate  
python manage.py runserver  

**Then open your browser and go to:  **
http://127.0.0.1:8000/

**Running Individual Python Scripts (Optional)**
**If you want to manually run any model script such as svm.py, knn.py, or cnn.py, you can do the following:**
cd "your path\FYP Project"  
.\gunshot_sound_classification\Scripts\activate  
cd gunshot_project\src  
python filename.py  

Replace "filename.py" with the script name you want to run.

**Downloading the Project**
To download this project:
1. Visit the GitHub repository page  
2. Click the green "Code" button  
3. Select "Download ZIP"  
4. Extract the ZIP file to your computer  

**Author**
VincentK0311  
Bachelor of Computer Science (Information Systems)  
Universiti Malaysia Sarawak (UNIMAS)

License
This project uses the MIT License.
