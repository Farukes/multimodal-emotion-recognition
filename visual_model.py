"""
Görsel Tabanlı Duygu Tanıma Modeli (Visual Emotion Recognition Model)

Bu betik, RAVDESS veri setindeki Full-AV (.mp4, 01 ile başlayan) video kayıtlarını işleyerek
yüz tespiti (Haar Cascade) uygular, her videodan 10 karelik sekanslar çıkarır ve
TimeDistributed CNN + GRU + Self-Attention hibrit derin öğrenme mimarisi ile
8 farklı duygu sınıfını sınıflandırır.
"""

import os
import argparse
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense, BatchNormalization, Dropout,
    TimeDistributed, GRU, Conv2D,
    MaxPooling2D, Flatten, Input, Layer
)
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers

from config import (
    EMOTIONS,
    DEFAULT_DATASET_DIR,
    DEFAULT_MODEL_DIR,
    VISUAL_MODEL_FILENAME,
    VISUAL_IMG_SIZE,
    VISUAL_SEQUENCE_LENGTH,
    RANDOM_SEED
)


class AttentionBlock(Layer):
    """
    Zaman boyutu üzerindeki gizli durumları (hidden states) ağırlıklandıran
    Öz-Dikkat (Self-Attention) Mekanizması.
    """
    def __init__(self, units, **kwargs):
        super(AttentionBlock, self).__init__(**kwargs)
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
        config = super(AttentionBlock, self).get_config()
        config.update({"units": self.units})
        return config


def collect_visual_dataset(dataset_path, seq_length=VISUAL_SEQUENCE_LENGTH, img_size=VISUAL_IMG_SIZE):
    """
    Videolardan yüz kareleri çıkarır, veri çoğaltma (yatay çevirme) uygular ve etiketler.
    """
    X_sequences, y_labels, actor_ids, is_mirrored = [], [], [], []
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    video_files = []
    for root, _, files in os.walk(dataset_path):
        for file in files:
            # Sadece Full-AV (01 ile başlayan) video dosyaları seçilir
            if file.lower().endswith(".mp4") and file.startswith("01"):
                video_files.append(os.path.join(root, file))

    if not video_files:
        raise FileNotFoundError(
            f"\n[HATA] '{dataset_path}' dizininde '01-*.mp4' video dosyası bulunamadı!\n"
            f"Lütfen RAVDESS video veri setini bu klasöre yerleştirin veya --data_dir ile yolu belirtin.\n"
            f"Örnek: python visual_model.py --data_dir /path/to/ravdess"
        )

    video_files.sort()
    print(f"[BİLGİ] {len(video_files)} adet video dosyası işleniyor...")

    for i, v_path in enumerate(video_files):
        if i % 20 == 0 or i == len(video_files) - 1:
            print(f"İşleniyor: {i + 1}/{len(video_files)}", end='\r')
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
                    # Veri çoğaltma: Yatay çevrilmiş (mirrored) kare
                    mirrored_seq.append(cv2.flip(face_roi, 1))
                    break

            if len(original_seq) == seq_length:
                # Orijinal sekans
                X_sequences.append(np.array(original_seq))
                y_labels.append(emotion_idx)
                actor_ids.append(actor_id)
                is_mirrored.append(False)

                # Aynalanmış (artırılmış) sekans
                X_sequences.append(np.array(mirrored_seq))
                y_labels.append(emotion_idx)
                actor_ids.append(actor_id)
                is_mirrored.append(True)

            cap.release()
        except Exception:
            continue

    print(f"\n[BİLGİ] Toplam {len(X_sequences)} video sekansı başarıyla çıkarıldı.")
    return np.array(X_sequences), np.array(y_labels), np.array(actor_ids), np.array(is_mirrored)


def build_visual_model(seq_length=VISUAL_SEQUENCE_LENGTH, img_size=VISUAL_IMG_SIZE, num_classes=len(EMOTIONS)):
    """
    TimeDistributed CNN + GRU + Attention Hibrit Derin Öğrenme Mimarisi.
    """
    inputs = Input(shape=(seq_length, img_size, img_size, 3))

    # Uzamsal Özellik Çıkarımı (TimeDistributed CNN)
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

    # Zamansal İlişki Modelleme (Gated Recurrent Unit - GRU)
    x = GRU(256, return_sequences=True, kernel_regularizer=regularizers.l2(0.01))(x)
    x = Dropout(0.55)(x)

    # Öz-Dikkat Katmanı (Self-Attention Layer)
    x = AttentionBlock(256)(x)

    # Yoğun Katmanlar ve Çıkış
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
    Görsel modelin eğitim eğrilerini ve karmaşıklık matrisini çizer.
    """
    plt.figure(figsize=(16, 6))

    # Doğruluk Grafiği
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Eğitim (Train Acc)')
    plt.plot(history.history['val_accuracy'], label='Test - Gerçek (Val Acc)')
    plt.title('Görsel Model Doğruluk Grafiği')
    plt.xlabel('Epok (Epoch)')
    plt.ylabel('Doğruluk (Accuracy)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    # Kayıp Grafiği
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Eğitim Kaybı (Train Loss)')
    plt.plot(history.history['val_loss'], label='Test Kaybı (Val Loss)')
    plt.title('Görsel Model Kayıp Grafiği')
    plt.xlabel('Epok (Epoch)')
    plt.ylabel('Kayıp (Loss)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    if save_path:
        plt.savefig(os.path.join(save_path, "visual_training_curves.png"), dpi=300)
    plt.show()

    # Karmaşıklık Matrisi (Confusion Matrix)
    y_pred = np.argmax(model.predict(x_test), axis=1)
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='rocket_r',
                xticklabels=EMOTIONS, yticklabels=EMOTIONS)
    plt.title('Görsel Model - Karmaşıklık Matrisi (Confusion Matrix)')
    plt.xlabel('Tahmin Edilen (Predicted)')
    plt.ylabel('Gerçek Değer (Actual)')
    plt.tight_layout()
    if save_path:
        plt.savefig(os.path.join(save_path, "visual_confusion_matrix.png"), dpi=300)
    plt.show()

    # Detaylı Sınıflandırma Raporu
    print("\n--- DETAYLI SINIFLANDIRMA RAPORU (VISUAL MODEL) ---")
    print(classification_report(y_test, y_pred, target_names=EMOTIONS))


def main():
    parser = argparse.ArgumentParser(description="Görsel Tabanlı Duygu Tanıma Modeli Eğitimi")
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATASET_DIR,
                        help="RAVDESS veri setinin bulunduğu dizin yolu")
    parser.add_argument("--model_dir", type=str, default=DEFAULT_MODEL_DIR,
                        help="Eğitilen modelin kaydedileceği dizin yolu")
    parser.add_argument("--epochs", type=int, default=100, help="Eğitim epok sayısı (varsayılan: 100)")
    parser.add_argument("--batch_size", type=int, default=16, help="Yığın boyutu (varsayılan: 16)")
    args = parser.parse_args()

    # Veriyi Topla
    X_raw, y_raw, a_ids, is_m = collect_visual_dataset(args.data_dir)

    # Aktör Bazlı Bölme (Subject-Independent Train/Test Split)
    unique_actors = sorted(np.unique(a_ids))
    random.seed(RANDOM_SEED)
    random.shuffle(unique_actors)
    split_point = int(len(unique_actors) * 0.8)

    train_mask = np.isin(a_ids, unique_actors[:split_point])
    # Test setinde yalnızca orijinal (aynalanmamış) veriler değerlendirilir
    test_mask = np.isin(a_ids, unique_actors[split_point:]) & (is_m == False)

    x_train = X_raw[train_mask].astype('float32') / 255.0
    x_test = X_raw[test_mask].astype('float32') / 255.0
    y_train = y_raw[train_mask]
    y_test = y_raw[test_mask]

    print(f"[BİLGİ] Eğitim Örnek Sayısı: {len(x_train)} (Aktörler: {unique_actors[:split_point]})")
    print(f"[BİLGİ] Test Örnek Sayısı (Yalnızca Orijinal): {len(x_test)} (Aktörler: {unique_actors[split_point:]})")

    # Modeli Kur
    model = build_visual_model()
    model.summary()

    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1),
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
    save_path = os.path.join(args.model_dir, VISUAL_MODEL_FILENAME)
    model.save(save_path)
    # Proje ana dizinine de geriye uyumluluk için kaydedelim
    model.save(VISUAL_MODEL_FILENAME)
    print(f"\n[BAŞARILI] Görsel model '{save_path}' ve '{VISUAL_MODEL_FILENAME}' olarak kaydedildi.")

    # Analiz ve Görselleştirme
    plot_results(history, model, x_test, y_test, save_path=args.model_dir)


if __name__ == "__main__":
    main()
