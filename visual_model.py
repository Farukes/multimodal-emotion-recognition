"""
Visual Emotion Recognition Model (TimeDistributed CNN + GRU + Attention)

This script processes Full-AV video recordings (.mp4, starting with 01) from the
RAVDESS dataset, detects faces using OpenCV Haar Cascades, samples 10-frame sequences,
and trains a hybrid TimeDistributed CNN + GRU + Self-Attention architecture to
classify 8 emotional states using an actor-based (subject-independent) evaluation.
"""

import argparse
import os
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import (
    GRU,
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    Input,
    Layer,
    MaxPooling2D,
    TimeDistributed,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from config import (
    DEFAULT_DATASET_DIR,
    DEFAULT_MODEL_DIR,
    EMOTIONS,
    NUM_CLASSES,
    RANDOM_SEED,
    VISUAL_IMG_SIZE,
    VISUAL_MODEL_FILENAME,
    VISUAL_SEQUENCE_LENGTH,
)


class AttentionBlock(Layer):
    """
    Self-Attention Mechanism that dynamically weights temporal hidden states
    along the sequence length to focus on critical emotion-bearing frames.
    """
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.W = Dense(units, activation='tanh')
        self.V = Dense(1)

    def call(self, inputs):
        score = self.V(self.W(inputs))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * inputs
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config


def collect_visual_dataset(dataset_path, seq_length=VISUAL_SEQUENCE_LENGTH, img_size=VISUAL_IMG_SIZE):
    """
    Extracts face frames from videos, applies horizontal flip augmentation, and pairs with labels.
    """
    X_sequences, y_labels, actor_ids, is_mirrored = [], [], [], []
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    video_files = []
    for root, _, files in os.walk(dataset_path):
        for file in files:
            # Select Full-AV recordings starting with 01
            if file.lower().endswith(".mp4") and file.startswith("01"):
                video_files.append(os.path.join(root, file))

    if not video_files:
        raise FileNotFoundError(
            f"\n[ERROR] No '01-*.mp4' video files found in '{dataset_path}'!\n"
            f"Please place the RAVDESS video dataset in this directory or specify --data_dir.\n"
            f"Example: python visual_model.py --data_dir /path/to/ravdess"
        )

    video_files.sort()
    print(f"[INFO] Processing {len(video_files)} video files in alphabetical order...")

    for i, v_path in enumerate(video_files):
        if i % 20 == 0 or i == len(video_files) - 1:
            print(f"Progress: {i + 1}/{len(video_files)}", end='\r')
        try:
            parts = os.path.basename(v_path).split('-')
            emotion_idx = int(parts[2]) - 1
            actor_id = int(parts[-1].split('.')[0])

            cap = cv2.VideoCapture(v_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames < seq_length:
                cap.release()
                continue

            frame_indices = np.linspace(2, total_frames - 2, seq_length, dtype=int)
            original_seq, mirrored_seq = [], []

            for pos in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
                ret, frame = cap.read()
                if not ret:
                    break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

                for (x, y_p, w, h) in faces:
                    face_roi = cv2.resize(frame[y_p:y_p + h, x:x + w], (img_size, img_size))
                    original_seq.append(face_roi)
                    # Data augmentation: horizontally flipped sequence
                    mirrored_seq.append(cv2.flip(face_roi, 1))
                    break

            if len(original_seq) == seq_length:
                # Original sequence
                X_sequences.append(np.array(original_seq))
                y_labels.append(emotion_idx)
                actor_ids.append(actor_id)
                is_mirrored.append(False)

                # Augmented (mirrored) sequence
                X_sequences.append(np.array(mirrored_seq))
                y_labels.append(emotion_idx)
                actor_ids.append(actor_id)
                is_mirrored.append(True)

            cap.release()
        except (cv2.error, IndexError, ValueError) as err:
            print(f"\n[WARNING] Skipping problematic video '{v_path}': {err}")
            continue

    print(f"\n[INFO] Successfully extracted {len(X_sequences)} video sequences.")
    return np.array(X_sequences), np.array(y_labels), np.array(actor_ids), np.array(is_mirrored)


def build_visual_model(seq_length=VISUAL_SEQUENCE_LENGTH, img_size=VISUAL_IMG_SIZE, num_classes=NUM_CLASSES):
    """
    Constructs the hybrid TimeDistributed CNN + GRU + Attention deep learning model.
    """
    inputs = Input(shape=(seq_length, img_size, img_size, 3))

    # Spatial Feature Extraction (TimeDistributed CNN)
    x = TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding='same'))(inputs)
    x = TimeDistributed(MaxPooling2D(2, 2))(x)
    x = TimeDistributed(BatchNormalization())(x)

    x = TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same'))(x)
    x = TimeDistributed(MaxPooling2D(2, 2))(x)
    x = TimeDistributed(BatchNormalization())(x)

    x = TimeDistributed(Conv2D(128, (3, 3), activation='relu', padding='same'))(x)
    x = TimeDistributed(MaxPooling2D(2, 2))(x)
    x = TimeDistributed(BatchNormalization())(x)

    x = TimeDistributed(Flatten())(x)

    # Temporal Sequence Modeling (Gated Recurrent Unit - GRU)
    x = GRU(256, return_sequences=True, kernel_regularizer=regularizers.l2(0.01))(x)
    x = Dropout(0.55)(x)

    # Self-Attention Layer
    x = AttentionBlock(256)(x)

    # Dense Classifier
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.65)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
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
    plt.plot(history.history['val_accuracy'], label='Test Accuracy (Unseen Subjects)')
    plt.title('Visual Model Accuracy Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation/Test Loss')
    plt.title('Visual Model Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    if save_path:
        plt.savefig(os.path.join(save_path, "visual_training_curves.png"), dpi=300)
    plt.show()

    # Confusion Matrix
    y_pred = np.argmax(model.predict(x_test), axis=1)
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='rocket_r',
                xticklabels=EMOTIONS, yticklabels=EMOTIONS)
    plt.title('Visual Model - Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    if save_path:
        plt.savefig(os.path.join(save_path, "visual_confusion_matrix.png"), dpi=300)
    plt.show()

    # Detailed Classification Report
    print("\n--- DETAILED CLASSIFICATION REPORT (VISUAL MODEL) ---")
    print(classification_report(y_test, y_pred, target_names=EMOTIONS))


def main():
    parser = argparse.ArgumentParser(description="Train Visual Emotion Recognition Model")
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATASET_DIR,
                        help="Path to the RAVDESS dataset directory")
    parser.add_argument("--model_dir", type=str, default=DEFAULT_MODEL_DIR,
                        help="Path to save trained model weights")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs (default: 100)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size (default: 16)")
    args = parser.parse_args()

    # Collect dataset
    X_raw, y_raw, a_ids, is_m = collect_visual_dataset(args.data_dir)

    # Actor-Based (Subject-Independent) Split
    unique_actors = sorted(np.unique(a_ids))
    random.seed(RANDOM_SEED)
    random.shuffle(unique_actors)
    split_point = int(len(unique_actors) * 0.8)

    train_mask = np.isin(a_ids, unique_actors[:split_point])
    # The test set evaluates exclusively on original (unmirrored) recordings from unseen actors
    test_mask = np.isin(a_ids, unique_actors[split_point:]) & (is_m == False)

    x_train = X_raw[train_mask].astype('float32') / 255.0
    x_test = X_raw[test_mask].astype('float32') / 255.0
    y_train = y_raw[train_mask]
    y_test = y_raw[test_mask]

    print(f"[INFO] Train Samples: {len(x_train)} (Actors: {unique_actors[:split_point]})")
    print(f"[INFO] Test Samples (Original only): {len(x_test)} (Actors: {unique_actors[split_point:]})")

    # Build model
    model = build_visual_model()
    model.summary()

    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1),
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
    save_path = os.path.join(args.model_dir, VISUAL_MODEL_FILENAME)
    model.save(save_path)
    # Save a copy in project root for backward compatibility
    model.save(VISUAL_MODEL_FILENAME)
    print(f"\n[SUCCESS] Visual model saved to '{save_path}' and '{VISUAL_MODEL_FILENAME}'.")

    # Evaluate and visualize
    plot_results(history, model, x_test, y_test, save_path=args.model_dir)


if __name__ == "__main__":
    main()
