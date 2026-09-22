"""
Multimodal Emotion Recognition - Late Decision Fusion (Soft Voting)

This script loads the independently trained Audio CNN and Visual CNN-GRU-Attention
models and evaluates a decision-level Late Fusion (Soft Voting) strategy across
synchronized test pairs from unseen actors.
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
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import CustomObjectScope

from config import (
    AUDIO_FIXED_SR,
    AUDIO_IMG_SHAPE,
    AUDIO_MODEL_FILENAME,
    AUDIO_N_MELS,
    DEFAULT_DATASET_DIR,
    DEFAULT_MODEL_DIR,
    EMOTIONS,
    RANDOM_SEED,
    VISUAL_IMG_SIZE,
    VISUAL_MODEL_FILENAME,
    VISUAL_SEQUENCE_LENGTH,
)
from visual_model import AttentionBlock


def resolve_model_path(model_filename, model_dir):
    """
    Searches for the model file in the specified model_dir and the project root directory.
    """
    candidate_1 = os.path.join(model_dir, model_filename)
    if os.path.exists(candidate_1):
        return candidate_1

    candidate_2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), model_filename)
    if os.path.exists(candidate_2):
        return candidate_2

    return None


def extract_audio_features(file_path):
    """
    Extracts 3-channel (Mel, Delta, Delta-Delta) spectrogram representation from speech.
    """
    y, sr = librosa.load(file_path, sr=AUDIO_FIXED_SR)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=AUDIO_N_MELS)
    res_s = cv2.resize(librosa.power_to_db(mel, ref=np.max), (AUDIO_IMG_SHAPE[0], AUDIO_IMG_SHAPE[1]))
    s_feat = np.stack([res_s, librosa.feature.delta(res_s), librosa.feature.delta(res_s, order=2)], axis=-1)
    return s_feat


def extract_video_frames(video_path, face_cascade, seq_length=VISUAL_SEQUENCE_LENGTH, img_size=VISUAL_IMG_SIZE):
    """
    Extracts 10 uniformly spaced, Haar Cascade-detected facial frames from video.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < seq_length:
        cap.release()
        return None

    idx_list = np.linspace(2, total_frames - 2, seq_length, dtype=int)
    for idx in idx_list:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        for (x, y_p, w, h) in faces:
            frames.append(cv2.resize(frame[y_p:y_p + h, x:x + w], (img_size, img_size)))
            break

    cap.release()
    if len(frames) == seq_length:
        return np.array(frames).astype('float32') / 255.0
    return None


def get_synced_test_data(dataset_path):
    """
    Synchronizes audio and video pairs for unseen test actors (20% subject-independent split).
    """
    actors = list(range(1, 25))
    random.seed(RANDOM_SEED)
    random.shuffle(actors)
    test_actors = set(actors[int(len(actors) * 0.8):])

    print(f"[INFO] Test Partition Actors: {sorted(test_actors)}")

    xs_test, xv_test, y_true = [], [], []
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    audio_files = []
    for root, _, files in os.walk(dataset_path):
        for f in files:
            if f.startswith("03") and f.lower().endswith(".wav"):
                audio_files.append(os.path.join(root, f))

    if not audio_files:
        raise FileNotFoundError(
            f"[ERROR] No '03-*.wav' audio files found in '{dataset_path}'. "
            f"Please verify the dataset path."
        )

    audio_files.sort()
    print(f"[INFO] Scanning {len(audio_files)} audio files to construct synchronized multimodal test pairs...")

    for f_path in audio_files:
        f_name = os.path.basename(f_path)
        parts = f_name.split('-')
        a_id = int(parts[-1].split('.')[0])

        if a_id in test_actors:
            try:
                # Find matching Full-AV video (03- -> 01-)
                v_name = "01" + f_name[2:].replace(".wav", ".mp4")
                v_path = None

                target_actor_dir = f"Actor_{a_id:02d}"
                for root, dirs, files in os.walk(dataset_path):
                    if target_actor_dir in dirs:
                        potential_path = os.path.join(root, target_actor_dir, v_name)
                        if os.path.exists(potential_path):
                            v_path = potential_path
                            break
                    if v_name in files:
                        v_path = os.path.join(root, v_name)
                        break

                if v_path and os.path.exists(v_path):
                    v_frames = extract_video_frames(v_path, face_cascade)
                    if v_frames is not None:
                        s_feat = extract_audio_features(f_path)
                        xs_test.append(s_feat)
                        xv_test.append(v_frames)
                        y_true.append(int(parts[2]) - 1)
            except (cv2.error, IndexError, ValueError) as err:
                print(f"[WARNING] Skipping sample '{f_name}': {err}")
                continue

    return np.array(xs_test), np.array(xv_test), np.array(y_true)


def main():
    parser = argparse.ArgumentParser(description="Multimodal Emotion Recognition - Late Decision Fusion")
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATASET_DIR,
                        help="Path to the RAVDESS dataset directory")
    parser.add_argument("--model_dir", type=str, default=DEFAULT_MODEL_DIR,
                        help="Path containing trained model weight files (.keras)")
    parser.add_argument("--audio_weight", type=float, default=0.5,
                        help="Decision weight for Audio Model (default: 0.5)")
    parser.add_argument("--visual_weight", type=float, default=0.5,
                        help="Decision weight for Visual Model (default: 0.5)")
    args = parser.parse_args()

    # Locate trained weights
    audio_path = resolve_model_path(AUDIO_MODEL_FILENAME, args.model_dir)
    visual_path = resolve_model_path(VISUAL_MODEL_FILENAME, args.model_dir)

    if not audio_path or not visual_path:
        print("\n[WARNING] Trained model weight checkpoints were not found:")
        print(f" - Audio Model ({AUDIO_MODEL_FILENAME}): {'FOUND (' + audio_path + ')' if audio_path else 'NOT FOUND'}")
        print(f" - Visual Model ({VISUAL_MODEL_FILENAME}): {'FOUND (' + visual_path + ')' if visual_path else 'NOT FOUND'}")
        print("\nPlease train the models first:")
        print(" 1) python audio_model.py")
        print(" 2) python visual_model.py")
        print("Or place pre-trained weights in the 'models/' directory.")
        return

    print("[1/3] Loading trained neural networks...")
    with CustomObjectScope({'AttentionBlock': AttentionBlock}):
        model_ses = load_model(audio_path)
        model_video = load_model(visual_path)
    print(f"[SUCCESS] Loaded Audio Model: {audio_path}")
    print(f"[SUCCESS] Loaded Visual Model: {visual_path}")

    # Synchronize multimodal test data
    print("\n[2/3] Constructing synchronized test set from unseen subjects...")
    xs, xv, y_true = get_synced_test_data(args.data_dir)

    if len(xs) == 0:
        print("\n[ERROR] No matching audio-video test pairs found!")
        print("Please check the directory structure. Expected folders: Actor_01, Actor_02, etc.")
        return

    # Predictions & Late Fusion
    print(f"\n[3/3] Performing Decision-Level Late Fusion on {len(xs)} test samples...")
    p_s = model_ses.predict(xs)
    p_v = model_video.predict(xv)

    # Weighted Soft Voting
    w_s = args.audio_weight
    w_v = args.visual_weight
    p_final = (w_s * p_s) + (w_v * p_v)
    y_pred = np.argmax(p_final, axis=1)

    # Results reporting
    print("\n" + "=" * 60)
    print("      MULTIMODAL MODEL EVALUATION (LATE DECISION FUSION)      ")
    print("=" * 60)
    print(classification_report(y_true, y_pred, target_names=EMOTIONS))

    # Confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d', cmap='Blues',
                xticklabels=EMOTIONS, yticklabels=EMOTIONS)
    plt.title('Multimodal Model (Late Fusion) - Confusion Matrix')
    plt.xlabel('Predicted Emotion')
    plt.ylabel('Ground Truth Emotion')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
