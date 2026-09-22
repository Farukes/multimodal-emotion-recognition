# Multimodal Emotion Recognition from Facial Expressions and Voice
### Yüz ve Ses Dinamiklerinden Çok Modlu Duygu Tanıma (Bitirme Projesi)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13%2B-orange.svg)](https://tensorflow.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green.svg)](https://opencv.org/)
[![Librosa](https://img.shields.io/badge/Librosa-Audio%20Processing-yellow.svg)](https://librosa.org/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

Bu proje, insan konuşmasındaki **ses tonu/prozodi** ve **yüz ifadelerini** eş zamanlı analiz ederek 8 temel duygu durumunu yüksek doğrulukla sınıflandıran derin öğrenme tabanlı **Çok Modlu (Multimodal) Duygu Tanıma Sistemidir**.

---

## 📌 Proje Özeti ve Motivasyon

Tek modlu (yalnızca ses veya yalnızca görüntü) sistemler, ortam gürültüsü, ışık yetersizliği veya bastırılmış mimikler gibi durumlarda yetersiz kalabilmektedir. İnsan iletişiminde ses ve yüz dinamikleri birbirini tamamlayıcı zengin bilgi taşır.

Bu çalışmada:
1. **Ses Kanalı:** Ses kayıtlarından 3 kanallı (Mel-Spektrogram, Delta, Delta-Delta) özellik haritası çıkarılmış ve 2D CNN ile eğitilmiştir.
2. **Görsel Kanal:** Videolardan Haar Cascade ile yüzler tespit edilerek 10 karelik sekanslar oluşturulmuş; uzamsal ve zamansal ilişkileri yakalamak için **TimeDistributed CNN + GRU + Self-Attention** hibrit mimarisi kullanılmıştır.
3. **Çok Modlu Füzyon (Late Fusion):** İki bağımsız modelin Softmax olasılık vektörleri **Yumuşak Oylama (Soft Voting)** ile karar seviyesinde birleştirilmiştir.

---

## 📊 Model Başarı ve Karşılaştırma Tablosu

Tüm modellerde veri sızıntısını önlemek ve ezberlemeyi engellemek amacıyla **Aktör Bazlı (Subject-Independent / Speaker-Independent)** 80/20 train/test ayrımı uygulanmıştır.

| Model | Kullanılan Mimari | Girdi Özellikleri | Test Doğruluğu (Accuracy) |
| :--- | :--- | :--- | :---: |
| **Ses Modeli (Audio)** | 3 Katmanlı 2D CNN + BatchNorm + Dropout | 3 Kanallı Spektrogram (Mel, Delta, Delta-Delta) [128x128x3] | **%71** |
| **Görsel Model (Visual)** | TimeDistributed CNN + GRU + Self-Attention | 10 Kare Yüz Sekansı [10x64x64x3] | **%74** |
| **Birleşik Model (Multimodal)** | Karar Seviyesi Geç Birleştirme (Late Fusion / Soft Voting) | Ses + Görüntü Eş Zamanlı Test Çiftleri | **%81** |

> 💡 **Önemli Not:** Çok modlu geç birleştirme (Late Fusion), tek modlu modellere kıyasla genel doğrulukta **+%7 ile +%10 arasında belirgin bir performans artışı** sağlamıştır. Bu durum, ses ve görsel kanalların birbirinin eksik kaldığı duygu sınıflarını başarıyla tamamladığını doğrulamaktadır.

---

## 🔬 Aktör Bazlı (Subject-Independent) Ayırmanın Önemi

### ❓ "Eğer Aktör Bazlı Ayrım Olmasaydı Ne Olurdu?"

Duygu tanıma çalışmalarında en sık yapılan metodolojik hata, verilerin **rastgele (random split)** veya sadece **duygu sınıflarına göre dengelenerek** bölünmesidir. Bu projede ve bitirme sunumumuzda (Slayt 15-16) özellikle vurgulanan **Aktör Bazlı Bölme (Subject-Independent / Speaker-Independent)** yaklaşımının tercih edilme gerekçesi ve rastgele bölmeyle arasındaki kritik farklar şunlardır:

#### 1. Veri Sızıntısı (Data Leakage)
- **Rastgele / Duygu Bazlı Bölmede:** Bir aktörün ("Örn: Aktör 01") belirli ses ve video kayıtları eğitim setine giderken, aynı aktörün başka kayıtları test setine düşer.
- **Sonuç:** Model, test aşamasında daha önce yüzünü ve ses tonunu eğitimde defalarca gördüğü bir kişiyle karşılaşır. Bu durum makine öğrenmesinde doğrudan bir **Veri Sızıntısı (Data Leakage)** problemidir.

#### 2. Kişi ve Biyometrik Kimliği Ezberleme (Identity Overfitting)
- **Ses Kanalında:** Model, duygunun getirdiği perde (pitch), spektral enerji, tempo veya formant frekanslarını öğrenmek yerine; aktörün kişisel ses rengini (tınısını), konuşma tarzını, temel frekansını (F0) ve vokal imzasını ezberler.
- **Görsel Kanalda:** Model, kaş çatılması, ağız kenarı gerilmesi gibi evrensel mimik hareketlerini öğrenmek yerine; aktörün ten rengini, yüz morfolojisini, sakalını, saç yapısını veya stüdyo ışıklandırmasını ezberler.

#### 3. Sahte / Yanıltıcı Yüksek Başarı (Artificially Inflated Accuracy)
- Rastgele bölme yapıldığında modeller kağıt üzerinde **%90 - %98** gibi olağanüstü yüksek doğruluk oranlarına kolaylıkla ulaşabilir.
- **Ancak bu başarı bir yanılsamadır:** Model gerçekte *duyguları sınıflandırmayı* değil, *kişileri tanımayı (identity classification)* öğrenmiştir.
- Böyle bir model gerçek dünyaya çıkarılıp daha önce veri setinde hiç yer almayan yeni bir kişiyle test edildiğinde performansı **%40-%50 seviyelerine kadar çakılır**.

#### 4. Bu Projede Elde Edilen Sonuçların Bilimsel Değeri
- Bu projede 24 aktör kesin sınırlarla ayrılmıştır (%80 Eğitim / %20 Test).
- Test setindeki 5 aktörün sesi ve yüzü, eğitim esnasında **modele kesinlikle gösterilmemiştir**.
- Elde edilen **%71 (Ses), %74 (Görsel) ve %81 (Çok Modlu / Late Fusion)** doğruluk oranları; modelin aktörleri ezberlemeden, tamamen yabancı yeni insanlarda da genellenebilir saf duygu dinamiklerini başarıyla öğrendiğini kanıtlar.

| Metodoloji | Test Kümesindeki Kişiler | Modelin Gerçekte Öğrendiği | Kağıt Üstü Doğruluk | Gerçek Hayat Genellenebilirliği |
| :--- | :--- | :--- | :---: | :---: |
| **Rastgele / Duygu Bazlı Bölme** | Eğitimde yer alan aynı aktörlerin farklı cümleleri | Biyometrik kimlik, yüz şekli, ses tonu (Ezberleme) | Yapay Olarak Yüksek (~%90 - %98) | ❌ **Çok Zayıf** (Yeni kişide çöker) |
| **Aktör Bazlı Bölme (Bu Proje)** | Modele tamamen yabancı, görülmemiş aktörler (%20) | Saf duygu dinamikleri, yüz kas hareketleri, prozodi | Gerçekçi ve Dürüst (**%81**) | ✅ **Yüksek** (Gerçek dünyaya uyumlu) |

---

## 🎯 Tanınan Duygu Sınıfları (8 Sınıf)

Model, RAVDESS standardındaki 8 duygu sınıfını sınıflandırmaktadır:
1. **Nötr** (Neutral)
2. **Sakin** (Calm)
3. **Mutlu** (Happy)
4. **Üzgün** (Sad)
5. **Öfkeli** (Angry)
6. **Korkulu** (Fearful)
7. **İğrenmiş** (Disgust)
8. **Şaşkın** (Surprised)

---

## 📁 Proje Dosya Yapısı

```
multimodal-emotion-recognition/
│
├── dataset/                    # RAVDESS veri seti klasörü (Actor_01, Actor_02...)
│   └── .gitkeep
├── models/                     # Eğitilen model ağırlıkları (.keras)
│   └── .gitkeep
│
├── config.py                   # Genel yapılandırma, duygu etiketleri ve parametreler
├── audio_model.py              # Ses modeli mimarisi, veri işleme ve eğitim betiği
├── visual_model.py             # Görsel model (CNN+GRU+Attention) ve eğitim betiği
├── multimodal_model.py         # Çok modlu geç birleştirme (Late Fusion) değerlendirme betiği
│
├── ses_model.py                # Ses modeli çalıştırma betiği (audio_model yönlendirmesi)
├── gorsel_model.py             # Görsel model çalıştırma betiği (visual_model yönlendirmesi)
├── birlesik_model.py           # Birleşik model çalıştırma betiği (multimodal_model yönlendirmesi)
├── Sesmodel.py                 # Alternatif çalıştırma betiği
│
├── sunum.pdf                   # Bitirme Projesi detaylı sunum slaytları
├── requirements.txt            # Gerekli Python kütüphaneleri
├── .gitignore                  # Git takip dışı dosyalar
├── LICENSE                     # MIT Lisansı
└── README.md                   # Proje dokümantasyonu
```

---

## 📥 Veri Seti Kurulumu (RAVDESS)

Projede dünya standartlarında kabul gören **RAVDESS (The Ryerson Audio-Visual Database of Emotional Speech and Song)** veri seti kullanılmıştır.

1. **Veri Setini İndirin:**
   - Resmi Zenodo Bağlantısı: [RAVDESS Dataset Zenodo](https://zenodo.org/records/1188976) veya Kaggle üzerinden `Audio_Speech_Actors_01-24` ve `Video_Speech_Actor_01-24` arşivlerini indirin.
2. **Klasörleme:**
   İndirdiğiniz aktör klasörlerini projenin ana dizinindeki `dataset/` içerisine yerleştirin:
   ```
   dataset/
     ├── Actor_01/
     │     ├── 01-01-01-01-01-01-01.mp4
     │     ├── 03-01-01-01-01-01-01.wav
     │     └── ...
     ├── Actor_02/
     └── ...
   ```
   *(Farklı bir dizinde tutmak isterseniz betikleri `--data_dir /dosya/yolu` argümanı ile çalıştırabilirsiniz).*

### RAVDESS Dosya Adlandırma Formatı:
Örnek dosya adı: `03-01-03-01-02-01-12.wav`
- `03`: Modalite (01 = Video ve Ses Full-AV, 02 = Yalnızca Video, 03 = Yalnızca Ses)
- `01`: Vokal Kanal (01 = Konuşma / Speech, 02 = Şarkı / Song)
- `03`: Duygu (01: Nötr, 02: Sakin, 03: Mutlu, 04: Üzgün, 05: Öfkeli, 06: Korkulu, 07: İğrenmiş, 08: Şaşkın)
- `01`: Yoğunluk (01: Normal, 02: Güçlü)
- `02`: Cümle (01 veya 02)
- `01`: Tekrar (01 veya 02)
- `12`: Aktör Numarası (01-24; Tek sayılar erkek, çift sayılar kadın)

---

## 🚀 Kurulum ve Çalıştırma

### 1. Ortamı Hazırlama
```bash
# Depoyu klonlayın
git clone https://github.com/Farukes/multimodal-emotion-recognition.git
cd multimodal-emotion-recognition

# Sanal ortam oluşturup aktif edin
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Gerekli paketleri yükleyin
pip install -r requirements.txt
```

### 2. Modelleri Eğitme

#### Adım 1: Ses Modelini Eğitin
```bash
python audio_model.py --epochs 120 --batch_size 16
```
*(Farklı bir veri yolu için: `python audio_model.py --data_dir "D:/RAVDESS"`)*

Eğitim tamamlandığında model ağırlıkları `models/sesmodelson.keras` dosyasına kaydedilir ve doğruluk/kayıp ile karmaşıklık matrisi grafikleri ekrana gelir.

#### Adım 2: Görsel Modeli Eğitin
```bash
python visual_model.py --epochs 100 --batch_size 16
```
Model `models/gorselson.keras` dosyasına kaydedilir.

### 3. Çok Modlu Modeli (Late Fusion) Test Etme
Ses ve görsel modeller eğitildikten sonra, aktör bazlı ayrılmış test kümesi üzerinde karar füzyonunu çalıştırmak için:
```bash
python multimodal_model.py
```
İsteğe bağlı olarak ses ve görsel modellerin oy ağırlıklarını ayarlayabilirsiniz:
```bash
python multimodal_model.py --audio_weight 0.4 --visual_weight 0.6
```

---

## 🧠 Model Mimarisi Detayları

### Ses Modeli (Audio Model)
- **Girdi:** 128x128 boyutunda 3 kanallı matris (Mel-Spektrogram + Delta Hız + Delta-Delta İvme)
- **Katmanlar:** 3 adet Conv2D bloğu (64, 128, 256 filtre) + Batch Normalization + ReLU + MaxPooling + Dropout
- **Sınıflandırma:** Flatten + Dense(512) + Dropout(0.6) + Softmax(8)

### Görsel Model (Visual Model)
- **Girdi:** Her videodan Haar Cascade ile kırpılıp 64x64 piksele ölçeklenen 10 adet yüz karesi `(10, 64, 64, 3)`
- **Uzamsal Katman:** TimeDistributed CNN blokları (32, 64, 128 filtre) ile her karenin öznitelik haritası çıkarılır
- **Zamansal Katman:** 256 birimli GRU (Gated Recurrent Unit) ile kareler arası mimik geçişleri ve zamansal akış modellenir
- **Öz-Dikkat (Self-Attention):** Duygu ifadesinin en belirgin olduğu kritik anları daha yüksek ağırlıkla öne çıkaran Attention katmanı
- **Sınıflandırma:** Dense(512) + Batch Normalization + Dropout(0.65) + Softmax(8)

### Çok Modlu Birleştirme (Late Fusion / Soft Voting)
$$\hat{y} = \arg\max \left( w_{\text{ses}} \cdot P_{\text{ses}} + w_{\text{görsel}} \cdot P_{\text{görsel}} \right)$$
Eşit ağırlıklandırma ($w_{\text{ses}} = 0.5$, $w_{\text{görsel}} = 0.5$) ile test verilerinde **%81 genel doğruluk** elde edilmiştir.

---

## 📽️ Sunum Dosyası

Bitirme projesine ait ayrıntılı sunum slaytlarına, problem tanımına ve karşılaştırmalı grafiklere proje kök dizinindeki [`sunum.pdf`](sunum.pdf) dosyasından ulaşabilirsiniz.

---

## 👨‍💻 Geliştirici ve Lisans

- **Geliştirici:** Ömer Faruk Eskitürk
- **Lisans:** [MIT Lisansı](LICENSE) - Açık kaynak ve akademik amaçlarla serbestçe kullanılabilir.
