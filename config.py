"""
Proje Genel Yapılandırma ve Sabitler (Project Configuration & Constants)

Bu modül, duygu etiketleri, model parametreleri ve veri seti dizinlerini
merkezi bir şekilde yönetir.
"""

import os

# Duygu Sınıfları (RAVDESS Dataset Emotion Classes)
EMOTIONS = [
    'Nötr',      # 01 = Neutral
    'Sakin',     # 02 = Calm
    'Mutlu',     # 03 = Happy
    'Üzgün',     # 04 = Sad
    'Öfkeli',    # 05 = Angry
    'Korkulu',   # 06 = Fearful
    'İğrenmiş',  # 07 = Disgust
    'Şaşkın'     # 08 = Surprised
]

EMOTIONS_EN = [
    'Neutral',
    'Calm',
    'Happy',
    'Sad',
    'Angry',
    'Fearful',
    'Disgust',
    'Surprised'
]

# Varsayılan Dizinler (Default Directories)
# Ortam değişkeni (DATASET_PATH) tanımlıysa onu kullanır, aksi takdirde yerel ./dataset dizinini arar.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATASET_DIR = os.environ.get("DATASET_PATH", os.path.join(BASE_DIR, "dataset"))
DEFAULT_MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(BASE_DIR, "models"))

# Model Ağırlık Dosya Adları (Model Weight Filenames)
AUDIO_MODEL_FILENAME = "sesmodelson.keras"
VISUAL_MODEL_FILENAME = "gorselson.keras"

# Ses Modeli Parametreleri (Audio Model Hyperparameters)
AUDIO_IMG_SHAPE = (128, 128, 3)
AUDIO_FIXED_SR = 22050
AUDIO_N_MELS = 128

# Görsel Model Parametreleri (Visual Model Hyperparameters)
VISUAL_IMG_SIZE = 64
VISUAL_SEQUENCE_LENGTH = 10

# Rastgelelik Tohumu (Random Seed for Reproducibility)
RANDOM_SEED = 42
