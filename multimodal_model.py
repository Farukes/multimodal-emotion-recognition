"""
Çok Modlu Duygu Tanıma Modeli - Geç Birleştirme (Multimodal Late Fusion Model)

Bu betik, bağımsız olarak eğitilmiş Ses Modeli (CNN) ve Görsel Modeli (CNN-GRU-Attention)
birlikte kullanarak 'Geç Birleştirme' (Late Fusion / Soft Voting) uygular.
Ses ve yüz verisi senkronize test aktörleri üzerinde değerlendirilir ve nihai
doğruluk ile sınıflandırma performansı raporlanır.
"""

import os
import argparse
import random
import cv2
import librosa
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import CustomObjectScope

from config import (
    EMOTIONS,
    DEFAULT_DATASET_DIR,
    DEFAULT_MODEL_DIR,
    AUDIO_MODEL_FILENAME,
    VISUAL_MODEL_FILENAME,
    AUDIO_FIXED_SR,
    AUDIO_N_MELS,
    AUDIO_IMG_SHAPE,
    VISUAL_IMG_SIZE,
    VISUAL_SEQUENCE_LENGTH,
    RANDOM_SEED
)
from visual_model import AttentionBlock


def resolve_model_path(model_filename, model_dir):
    """
    Model dosyasının sırasıyla model_dir dizininde ve proje ana dizininde varlığını kontrol eder.
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
    Ses dosyasından 3 kanallı (Mel, Delta, Delta-Delta) özellik çıkarır.
    """
    y, sr = librosa.load(file_path, sr=AUDIO_FIXED_SR)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=AUDIO_N_MELS)
    res_s = cv2.resize(librosa.power_to_db(mel, ref=np.max), (AUDIO_IMG_SHAPE[0], AUDIO_IMG_SHAPE[1]))
    s_feat = np.stack([res_s, librosa.feature.delta(res_s), librosa.feature.delta(res_s, order=2)], axis=-1)
    return s_feat


def extract_video_frames(video_path, face_cascade, seq_length=VISUAL_SEQUENCE_LENGTH, img_size=VISUAL_IMG_SIZE):
    """
    Video dosyasından yüz tespiti yaparak 10 adet yüz karesi çıkarır.
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
    Test kümesine ait aktörlerin ses ve video kayıtlarını eşleştirerek hazırlar.
    """
    # 24 Aktör arasından 80/20 aktör bazlı test kümesi seçimi
    actors = list(range(1, 25))
    random.seed(RANDOM_SEED)
    random.shuffle(actors)
    test_actors = set(actors[int(len(actors) * 0.8):])

    print(f"[BİLGİ] Test Kümesi Aktörleri: {sorted(list(test_actors))}")

    xs_test, xv_test, y_true = [], [], []
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    audio_files = []
    for root, _, files in os.walk(dataset_path):
        for f in files:
            if f.startswith("03") and f.lower().endswith(".wav"):
                audio_files.append(os.path.join(root, f))

    if not audio_files:
        raise FileNotFoundError(
            f"[HATA] '{dataset_path}' dizininde '03-*.wav' ses dosyası bulunamadı. "
            f"Lütfen geçerli veri seti yolunu belirtin."
        )

    audio_files.sort()
    print(f"[BİLGİ] {len(audio_files)} ses dosyası taranarak eşleşen test verileri oluşturuluyor...")

    for f_path in audio_files:
        f_name = os.path.basename(f_path)
        parts = f_name.split('-')
        a_id = int(parts[-1].split('.')[0])

        if a_id in test_actors:
            try:
                # Eşleşen video dosya adını belirle (03 ile başlayan ses -> 01 ile başlayan Full-AV video)
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
            except Exception:
                continue

    return np.array(xs_test), np.array(xv_test), np.array(y_true)


def main():
    parser = argparse.ArgumentParser(description="Çok Modlu Duygu Tanıma - Geç Birleştirme (Late Fusion)")
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATASET_DIR,
                        help="RAVDESS veri setinin bulunduğu dizin yolu")
    parser.add_argument("--model_dir", type=str, default=DEFAULT_MODEL_DIR,
                        help="Eğitilen modellerin (.keras) bulunduğu dizin yolu")
    parser.add_argument("--audio_weight", type=float, default=0.5,
                        help="Ses modeli ağırlığı (varsayılan: 0.5)")
    parser.add_argument("--visual_weight", type=float, default=0.5,
                        help="Görsel model ağırlığı (varsayılan: 0.5)")
    args = parser.parse_args()

    # Modelleri Bul ve Yükle
    audio_path = resolve_model_path(AUDIO_MODEL_FILENAME, args.model_dir)
    visual_path = resolve_model_path(VISUAL_MODEL_FILENAME, args.model_dir)

    if not audio_path or not visual_path:
        print("\n[UYARI] Eğitilmiş model dosyaları bulunamadı:")
        print(f" - Ses Modeli ({AUDIO_MODEL_FILENAME}): {'BULUNDU (' + audio_path + ')' if audio_path else 'BULUNAMADI'}")
        print(f" - Görsel Model ({VISUAL_MODEL_FILENAME}): {'BULUNDU (' + visual_path + ')' if visual_path else 'BULUNAMADI'}")
        print("\nLütfen önce modelleri eğitin:")
        print(" 1) python audio_model.py")
        print(" 2) python visual_model.py")
        print("Veya hazır ağırlıkları 'models/' klasörüne yerleştirin.")
        return

    print("[1/3] Modeller yükleniyor...")
    with CustomObjectScope({'AttentionBlock': AttentionBlock}):
        model_ses = load_model(audio_path)
        model_video = load_model(visual_path)
    print(f"[BAŞARILI] Ses Modeli: {audio_path}")
    print(f"[BAŞARILI] Görsel Modeli: {visual_path}")

    # Test Verilerini Eşle ve Hazırla
    print("\n[2/3] Eş zamanlı test verileri toplanıyor...")
    xs, xv, y_true = get_synced_test_data(args.data_dir)

    if len(xs) == 0:
        print("\n[HATA] Eşleşen test verisi bulunamadı!")
        print("Lütfen veri seti klasör yapısını kontrol edin. Beklenen yapı: Actor_01, Actor_02... dizinleri.")
        return

    # Tahmin ve Geç Birleştirme (Late Fusion)
    print(f"\n[3/3] {len(xs)} adet test örneği üzerinde Geç Birleştirme (Late Fusion) gerçekleştiriliyor...")
    p_s = model_ses.predict(xs)
    p_v = model_video.predict(xv)

    # Yumuşak Oylama (Soft Voting) ile olasılık vektörlerinin ağırlıklı ortalaması
    w_s = args.audio_weight
    w_v = args.visual_weight
    p_final = (w_s * p_s) + (w_v * p_v)
    y_pred = np.argmax(p_final, axis=1)

    # Sonuçların Raporlanması
    print("\n" + "=" * 55)
    print("      ÇOK MODLU (MULTIMODAL) MODEL TEST SONUÇLARI      ")
    print("=" * 55)
    print(classification_report(y_true, y_pred, target_names=EMOTIONS))

    # Karmaşıklık Matrisi (Confusion Matrix)
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d', cmap='Blues',
                xticklabels=EMOTIONS, yticklabels=EMOTIONS)
    plt.title('Çok Modlu Model (Geç Birleştirme) - Karmaşıklık Matrisi')
    plt.xlabel('Tahmin Edilen Duygu (Predicted)')
    plt.ylabel('Gerçek Duygu (Actual)')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
