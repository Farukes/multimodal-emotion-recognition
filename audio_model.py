"""
Audio Emotion Recognition Model (2D CNN on 3-Channel Spectrograms)

This script processes speech recordings (.wav) from the RAVDESS dataset,
extracts 3-channel time-frequency representations (Mel-Spectrogram, Delta velocity,
and Delta-Delta acceleration), and trains a 2D Convolutional Neural Network (CNN)
to classify 8 emotional states using an actor-based (subject-independent) evaluation.
"""

import argparse
import os
import random

import cv2
import librosa
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    MaxPooling2D,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from config import (
    AUDIO_FIXED_SR,
    AUDIO_IMG_SHAPE,
    AUDIO_MODEL_FILENAME,
    AUDIO_N_MELS,
    DEFAULT_DATASET_DIR,
    DEFAULT_MODEL_DIR,
    EMOTIONS,
    NUM_CLASSES,
    RANDOM_SEED,
)


def extract_advanced_features(file_path, augment=False):
    """
    Extracts 3-channel spectrogram features from an audio file.
    Channel 1: Log Mel-Spectrogram (128x128)
    Channel 2: Delta (1st derivative / velocity)
    Channel 3: Delta-Delta (2nd derivative / acceleration)
    """
    y, sr = librosa.load(file_path, sr=AUDIO_FIXED_SR, duration=None)

    if augment:
        if random.random() > 0.5:
            y = librosa.effects.pitch_shift(y, sr=sr, n_steps=random.uniform(-2, 2))
        noise = np.random.randn(len(y))
        y = y + 0.005 * noise

    # Channel 1: Mel-Spectrogram
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=AUDIO_N_MELS)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_resized = cv2.resize(mel_db, (AUDIO_IMG_SHAPE[0], AUDIO_IMG_SHAPE[1]), interpolation=cv2.INTER_AREA)

    # Channel 2: Delta (Velocity)
    delta = librosa.feature.delta(mel_resized)

    # Channel 3: Delta-Delta (Acceleration)
    delta2 = librosa.feature.delta(mel_resized, order=2)

    # Stack into a 3-channel tensor (128, 128, 3)
    stacked_features = np.stack([mel_resized, delta, delta2], axis=-1)
    return stacked_features


def load_audio_dataset(dataset_path):
    """
    Scans the dataset directory for RAVDESS speech files (03-*.wav) and extracts labels.
    """
    X, y, actor_ids = [], [], []
    audio_files = []

    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith(".wav") and file.startswith("03"):
                audio_files.append(os.path.join(root, file))

    if not audio_files:
        raise FileNotFoundError(
            f"\n[ERROR] No '03-*.wav' audio files found in '{dataset_path}'!\n"
            f"Please place the RAVDESS dataset in this directory or specify --data_dir.\n"
            f"Example: python audio_model.py --data_dir /path/to/ravdess"
        )

    audio_files.sort()
    print(f"[INFO] Processing {len(audio_files)} audio files with 3-channel spectrogram analysis...")

    for i, a_path in enumerate(audio_files):
        if i % 50 == 0 or i == len(audio_files) - 1:
            print(f"Progress: {i + 1}/{len(audio_files)}", end='\r')
        try:
            parts = os.path.basename(a_path).split('-')
            emotion_idx = int(parts[2]) - 1
            actor_id = int(parts[-1].split('.')[0])

            spec = extract_advanced_features(a_path, augment=False)
            X.append(spec)
            y.append(emotion_idx)
            actor_ids.append(actor_id)
        except (IndexError, ValueError) as err:
            print(f"\n[WARNING] Skipping malformed file '{a_path}': {err}")
            continue

    print(f"\n[INFO] Successfully extracted features for {len(X)} audio samples.")
    return np.array(X), np.array(y), np.array(actor_ids)


def build_audio_model(input_shape=AUDIO_IMG_SHAPE, num_classes=NUM_CLASSES):
    """
    Constructs the 3-Layer 2D Convolutional Neural Network (CNN) architecture.
    """
    model = Sequential([
        # Block 1
        Conv2D(64, (3, 3), padding='same', input_shape=input_shape),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),

        # Block 2
        Conv2D(128, (3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.45),

        # Block 3
        Conv2D(256, (3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.45),

        # Dense Classifier
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.6),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def plot_results(history, model, x_test, y_test, save_path=None):
    """
    Plots training/validation curves and the confusion matrix.
    """
    plt.figure(figsize=(16, 6))

    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation/Test Accuracy')
    plt.title('Audio Model Accuracy Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation/Test Loss')
    plt.title('Audio Model Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    if save_path:
        plt.savefig(os.path.join(save_path, "audio_training_curves.png"), dpi=300)
    plt.show()

    # Confusion Matrix
    y_pred_probs = model.predict(x_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='viridis',
                xticklabels=EMOTIONS, yticklabels=EMOTIONS)
    plt.title('Audio Model - Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    if save_path:
        plt.savefig(os.path.join(save_path, "audio_confusion_matrix.png"), dpi=300)
    plt.show()

    # Classification Report
    print("\n--- DETAILED CLASSIFICATION REPORT (AUDIO MODEL) ---")
    print(classification_report(y_test, y_pred, target_names=EMOTIONS))


def main():
    parser = argparse.ArgumentParser(description="Train Audio Emotion Recognition Model")
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATASET_DIR,
                        help="Path to the RAVDESS dataset directory")
    parser.add_argument("--model_dir", type=str, default=DEFAULT_MODEL_DIR,
                        help="Path to save trained model weights")
    parser.add_argument("--epochs", type=int, default=120, help="Number of training epochs (default: 120)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size (default: 16)")
    args = parser.parse_args()

    # Load audio dataset
    X_raw, y_raw, a_ids = load_audio_dataset(args.data_dir)

    # Subject-Independent / Actor-Based Train/Test Split
    unique_actors = sorted(np.unique(a_ids))
    random.seed(RANDOM_SEED)
    random.shuffle(unique_actors)
    split_point = int(len(unique_actors) * 0.8)

    train_mask = np.isin(a_ids, unique_actors[:split_point])
    test_mask = np.isin(a_ids, unique_actors[split_point:])

    x_train, x_test = X_raw[train_mask], X_raw[test_mask]
    y_train, y_test = y_raw[train_mask], y_raw[test_mask]

    print(f"[INFO] Train Samples: {len(x_train)} (Actors: {unique_actors[:split_point]})")
    print(f"[INFO] Test Samples: {len(x_test)} (Actors: {unique_actors[split_point:]})")

    # Build model
    model = build_audio_model()
    model.summary()

    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, verbose=1),
        EarlyStopping(monitor='val_accuracy', patience=20, restore_best_weights=True)
    ]

    # Train model
    history = model.fit(
        x_train, y_train,
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_data=(x_test, y_test),
        callbacks=callbacks
    )

    # Save model weights
    os.makedirs(args.model_dir, exist_ok=True)
    save_path = os.path.join(args.model_dir, AUDIO_MODEL_FILENAME)
    model.save(save_path)
    # Save a copy in project root for backward compatibility
    model.save(AUDIO_MODEL_FILENAME)
    print(f"\n[SUCCESS] Audio model saved to '{save_path}' and '{AUDIO_MODEL_FILENAME}'.")

    # Evaluate and visualize
    plot_results(history, model, x_test, y_test, save_path=args.model_dir)


if __name__ == "__main__":
    main()
