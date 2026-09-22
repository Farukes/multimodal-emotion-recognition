"""
Ses Tabanlı Duygu Tanıma Modeli (Audio Emotion Recognition Model)

Bu betik, RAVDESS veri setindeki ses kayıtlarını (.wav) işleyerek
3 kanallı (Mel-Spektrogram, Delta, Delta-Delta) özellik haritaları çıkarır ve
2D Evrişimli Sinir Ağı (CNN) ile 8 farklı duygu sınıfını sınıflandırır.
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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization, Activation
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam

from config import (
    EMOTIONS,
    DEFAULT_DATASET_DIR,
    DEFAULT_MODEL_DIR,
    AUDIO_MODEL_FILENAME,
    AUDIO_IMG_SHAPE,
    AUDIO_FIXED_SR,
    AUDIO_N_MELS,
    RANDOM_SEED
)


def extract_advanced_features(file_path, augment=False):
    """
    Ses dosyasından 3 kanallı spektrogram özelliklerini çıkarır.
    Kanal 1: Log Mel-Spektrogram (128x128)
    Kanal 2: Delta (1. Derece Türev / Hız Değişimi)
    Kanal 3: Delta-Delta (2. Derece Türev / İvme Değişimi)
    """
    y, sr = librosa.load(file_path, sr=AUDIO_FIXED_SR, duration=None)

    if augment:
        if random.random() > 0.5:
            y = librosa.effects.pitch_shift(y, sr=sr, n_steps=random.uniform(-2, 2))
        noise = np.random.randn(len(y))
        y = y + 0.005 * noise

    # 1. Kanal: Mel-Spektrogram
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=AUDIO_N_MELS)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_resized = cv2.resize(mel_db, (AUDIO_IMG_SHAPE[0], AUDIO_IMG_SHAPE[1]), interpolation=cv2.INTER_AREA)

    # 2. Kanal: Delta (Hız Değişimi)
    delta = librosa.feature.delta(mel_resized)

    # 3. Kanal: Delta-Delta (İvme Değişimi)
    delta2 = librosa.feature.delta(mel_resized, order=2)

    # 3 Kanalı birleştirerek (128, 128, 3) boyutlu tensör oluşturulur
    stacked_features = np.stack([mel_resized, delta, delta2], axis=-1)
    return stacked_features


def load_audio_dataset(dataset_path):
    """
    Belirtilen dizindeki RAVDESS ses dosyalarını (03-*.wav) yükler ve etiketler.
    """
    X, y, actor_ids = [], [], []
    audio_files = []

    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith(".wav") and file.startswith("03"):
                audio_files.append(os.path.join(root, file))

    if not audio_files:
        raise FileNotFoundError(
            f"\n[HATA] '{dataset_path}' dizininde '03-*.wav' ses dosyası bulunamadı!\n"
            f"Lütfen RAVDESS veri setini bu klasöre yerleştirin veya --data_dir ile geçerli yolu belirtin.\n"
            f"Örnek: python audio_model.py --data_dir /path/to/ravdess"
        )

    audio_files.sort()
    print(f"[BİLGİ] {len(audio_files)} adet ses dosyası işleniyor...")

    for i, a_path in enumerate(audio_files):
        if i % 50 == 0 or i == len(audio_files) - 1:
            print(f"İşleniyor: {i + 1}/{len(audio_files)}", end='\r')
        try:
            parts = os.path.basename(a_path).split('-')
            emotion_idx = int(parts[2]) - 1
            actor_id = int(parts[-1].split('.')[0])

            spec = extract_advanced_features(a_path, augment=False)
            X.append(spec)
            y.append(emotion_idx)
            actor_ids.append(actor_id)
        except Exception as e:
            continue

    print(f"\n[BİLGİ] Toplam {len(X)} ses kaydı başarıyla çıkarıldı.")
    return np.array(X), np.array(y), np.array(actor_ids)


def build_audio_model(input_shape=AUDIO_IMG_SHAPE, num_classes=len(EMOTIONS)):
    """
    Güçlendirilmiş 3 Katmanlı Evrişimli Sinir Ağı (CNN) Mimarisi.
    """
    model = Sequential([
        # 1. Blok
        Conv2D(64, (3, 3), padding='same', input_shape=input_shape),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),

        # 2. Blok
        Conv2D(128, (3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.45),

        # 3. Blok
        Conv2D(256, (3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.45),

        # Sınıflandırma Katmanları
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
    Model eğitim geçmişi, doğruluk/kayıp grafikleri ve karmaşıklık matrisini çizer.
    """
    plt.figure(figsize=(16, 6))

    # Doğruluk Grafiği
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Eğitim Başarısı (Train Acc)')
    plt.plot(history.history['val_accuracy'], label='Test Başarısı (Val Acc)')
    plt.title('Ses Modeli Doğruluk Grafiği')
    plt.xlabel('Epok (Epoch)')
    plt.ylabel('Doğruluk (Accuracy)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    # Kayıp Grafiği
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Eğitim Kaybı (Train Loss)')
    plt.plot(history.history['val_loss'], label='Test Kaybı (Val Loss)')
    plt.title('Ses Modeli Kayıp Grafiği')
    plt.xlabel('Epok (Epoch)')
    plt.ylabel('Kayıp (Loss)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    if save_path:
        plt.savefig(os.path.join(save_path, "audio_training_curves.png"), dpi=300)
    plt.show()

    # Karmaşıklık Matrisi (Confusion Matrix)
    y_pred_probs = model.predict(x_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='viridis',
                xticklabels=EMOTIONS, yticklabels=EMOTIONS)
    plt.title('Ses Modeli - Karmaşıklık Matrisi (Confusion Matrix)')
    plt.xlabel('Tahmin Edilen (Predicted)')
    plt.ylabel('Gerçek Değer (Actual)')
    plt.tight_layout()
    if save_path:
        plt.savefig(os.path.join(save_path, "audio_confusion_matrix.png"), dpi=300)
    plt.show()

    # Sınıflandırma Raporu
    print("\n--- DETAYLI SINIFLANDIRMA RAPORU (AUDIO MODEL) ---")
    print(classification_report(y_test, y_pred, target_names=EMOTIONS))


def main():
    parser = argparse.ArgumentParser(description="Ses Tabanlı Duygu Tanıma Modeli Eğitimi")
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATASET_DIR,
                        help="RAVDESS veri setinin bulunduğu dizin yolu")
    parser.add_argument("--model_dir", type=str, default=DEFAULT_MODEL_DIR,
                        help="Eğitilen modelin kaydedileceği dizin yolu")
    parser.add_argument("--epochs", type=int, default=120, help="Eğitim epok sayısı (varsayılan: 120)")
    parser.add_argument("--batch_size", type=int, default=16, help="Yığın boyutu (varsayılan: 16)")
    args = parser.parse_args()

    # Veriyi Yükle
    X_raw, y_raw, a_ids = load_audio_dataset(args.data_dir)

    # Aktör Bazlı Bölme (Subject-Independent Train/Test Split)
    # Modelin kişiye özgü ses karakteristiklerini ezberlemesini önlemek için %80 eğitim / %20 test aktörleri
    unique_actors = sorted(np.unique(a_ids))
    random.seed(RANDOM_SEED)
    random.shuffle(unique_actors)
    split_point = int(len(unique_actors) * 0.8)

    train_mask = np.isin(a_ids, unique_actors[:split_point])
    test_mask = np.isin(a_ids, unique_actors[split_point:])

    x_train, x_test = X_raw[train_mask], X_raw[test_mask]
    y_train, y_test = y_raw[train_mask], y_raw[test_mask]

    print(f"[BİLGİ] Eğitim Örnek Sayısı: {len(x_train)} (Aktörler: {unique_actors[:split_point]})")
    print(f"[BİLGİ] Test Örnek Sayısı: {len(x_test)} (Aktörler: {unique_actors[split_point:]})")

    # Modeli Kur
    model = build_audio_model()
    model.summary()

    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, verbose=1),
        EarlyStopping(monitor='val_accuracy', patience=20, restore_best_weights=True)
    ]

    # Modeli Eğit
    history = model.fit(
        x_train, y_train,
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_data=(x_test, y_test),
        callbacks=callbacks
    )

    # Modeli Kaydet
    os.makedirs(args.model_dir, exist_ok=True)
    save_path = os.path.join(args.model_dir, AUDIO_MODEL_FILENAME)
    model.save(save_path)
    # Proje ana dizinine de geriye uyumluluk için kaydedelim
    model.save(AUDIO_MODEL_FILENAME)
    print(f"\n[BAŞARILI] Ses modeli '{save_path}' ve '{AUDIO_MODEL_FILENAME}' olarak kaydedildi.")

    # Analiz ve Görselleştirme
    plot_results(history, model, x_test, y_test, save_path=args.model_dir)


if __name__ == "__main__":
    main()
