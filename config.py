"""
Project Configuration and Constants (Centralized Settings)

This module defines emotion classes, default directory paths,
and model hyperparameters used across the codebase.
"""

import os

# Emotion Classes (RAVDESS Standard 8 Emotions)
EMOTIONS = [
    'Neutral',   # 01
    'Calm',      # 02
    'Happy',     # 03
    'Sad',       # 04
    'Angry',     # 05
    'Fearful',   # 06
    'Disgust',   # 07
    'Surprised'  # 08
]

# Total Number of Emotion Classes
NUM_CLASSES = len(EMOTIONS)

# Turkish Labels for Localization & Presentation Mapping
EMOTIONS_TR = [
    'Nötr',
    'Sakin',
    'Mutlu',
    'Üzgün',
    'Öfkeli',
    'Korkulu',
    'İğrenmiş',
    'Şaşkın'
]

# Default Paths (Uses environment variable if set, otherwise falls back to local relative paths)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATASET_DIR = os.environ.get("DATASET_PATH", os.path.join(BASE_DIR, "dataset"))
DEFAULT_MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(BASE_DIR, "models"))

# Model Filenames
AUDIO_MODEL_FILENAME = "sesmodelson.keras"
VISUAL_MODEL_FILENAME = "gorselson.keras"

# Audio Model Hyperparameters
AUDIO_IMG_SHAPE = (128, 128, 3)
AUDIO_FIXED_SR = 22050
AUDIO_N_MELS = 128

# Visual Model Hyperparameters
VISUAL_IMG_SIZE = 64
VISUAL_SEQUENCE_LENGTH = 10

# Random Seed for Reproducibility
RANDOM_SEED = 42
