# Multimodal Emotion Recognition from Facial Expressions and Voice Dynamics
### Deep Learning-Based Multimodal Emotion Recognition (Graduation Project)

[🇬🇧 English Documentation](README.md) | [🇹🇷 Türkçe Dokümantasyon için tıklayınız](README_TR.md)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13%2B-orange.svg)](https://tensorflow.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green.svg)](https://opencv.org/)
[![Librosa](https://img.shields.io/badge/Librosa-Audio%20Processing-yellow.svg)](https://librosa.org/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

This project is a deep learning-based **Multimodal Emotion Recognition System** that simultaneously analyzes **vocal prosody/dynamics** and **facial expressions** from human speech to classify 8 fundamental emotional states with high accuracy.

---

## 📌 Project Overview & Motivation

Unimodal emotion recognition systems (relying solely on audio or solely on video) frequently suffer from ambient noise, poor lighting conditions, or subtle/suppressed facial expressions. In natural human communication, acoustic and facial dynamics carry complementary information that resolve ambiguities when interpreted together.

In this work:
1. **Audio Channel:** 3-channel feature maps (Log Mel-Spectrogram, Delta velocity, Delta-Delta acceleration) are extracted from speech recordings and classified using a 2D Convolutional Neural Network (CNN).
2. **Visual Channel:** Faces are detected and cropped from video recordings using OpenCV Haar Cascade, sampled into 10-frame sequences, and processed via a hybrid **TimeDistributed CNN + GRU + Self-Attention** architecture that jointly models spatial facial features and temporal expression transitions.
3. **Multimodal Late Fusion:** Softmax probability vectors from the independently trained unimodal models are combined at the decision level using **Soft Voting (Weighted Late Fusion)**.

---

## 📊 Performance & Comparison Table

To strictly prevent data leakage and avoid identity memorization, an **Actor-Based (Subject-Independent / Speaker-Independent)** 80/20 train/test split was enforced across all models.

| Model | Architecture | Input Features | Test Accuracy |
| :--- | :--- | :--- | :---: |
| **Audio Model** | 3-Layer 2D CNN + BatchNorm + Dropout | 3-Channel Spectrogram (Mel, Delta, Delta2) [128x128x3] | **71%** |
| **Visual Model** | TimeDistributed CNN + GRU + Self-Attention | 10-Frame Facial Sequence [10x64x64x3] | **74%** |
| **Multimodal Model** | Decision-Level Late Fusion (Soft Voting) | Synchronized Audio + Video Test Pairs | **81%** |

> 💡 **Key Takeaway:** Decision-level late fusion achieves a significant **+7% to +10% accuracy improvement** over unimodal models, demonstrating that vocal acoustics and facial expressions effectively compensate for each other's limitations.

---

## 🔬 The Importance of Actor-Based (Subject-Independent) Splitting

### ❓ "What Would Happen Without Actor-Based Splitting?"

In affective computing and emotion recognition research, the most common methodological flaw is using **random split** or **emotion-stratified split** without isolating actors/subjects. In this project (and emphasized in Slides 15–16 of the graduation presentation), the **Actor-Based (Subject-Independent)** split is critically essential for the following reasons:

#### 1. Data Leakage
- **Under Random / Emotion-Based Splitting:** An actor's (e.g., Actor 01) "happy" or "angry" utterances are placed in the training set, while their other utterances end up in the test set.
- **Consequence:** The model encounters a face and voice it has already memorized during training. In machine learning, this constitutes direct **Data Leakage**.

#### 2. Identity & Biometric Overfitting
- **In the Audio Channel:** Rather than learning universal acoustic markers of emotion (pitch variations, energy shifts, tempo), the model memorizes the actor's unique vocal timbre, average fundamental frequency ($F_0$), and individual speaking style.
- **In the Visual Channel:** Rather than learning emotional muscle movements (brow furrows, lip corner pulls), the model memorizes the actor's facial bone structure, skin tone, beard, hairstyle, or studio lighting angle.

#### 3. Artificially Inflated / Deceptive Accuracy
- Models evaluated with random split often achieve deceptively high accuracy on paper (**90% – 98%**).
- **However, this is an illusion:** The model has not learned to *recognize emotions*; it has learned to *identify individuals (identity classification)*.
- When deployed in a real-world scenario with an unseen human subject, the accuracy of such a model instantly **plummets to 40% – 50%**.

#### 4. Scientific Rigor and Generalizability of This Study
- In this repository, the 24 actors are partitioned strictly into 80% Training (~19 actors) and 20% Testing (~5 actors).
- The test actors' voices and faces were **never seen or heard by the models during training**.
- The resulting **71% (Audio), 74% (Visual), and 81% (Multimodal)** scores reflect genuine generalization to new, unseen individuals in real-world conditions.

| Methodology | Test Set Subjects | What the Model Actually Learns | Apparent Accuracy | Real-World Generalizability |
| :--- | :--- | :--- | :---: | :---: |
| **Random / Emotion Split** | Utterances from the *same* actors seen in training | Biometric identity, facial morphology, vocal timbre | Artificially High (~90% - 98%) | ❌ **Extremely Fragile** (Fails on new people) |
| **Actor-Based Split (This Work)** | Utterances from *completely unseen* actors (20%) | Universal emotion dynamics, facial muscle action, prosody | Realistic & Honest (**81%**) | ✅ **Robust** (Generalizes to real world) |

---

## 🎯 Recognized Emotion Classes (8 Classes)

The system classifies all 8 standardized emotion classes from the RAVDESS dataset:
1. **Neutral** (01)
2. **Calm** (02)
3. **Happy** (03)
4. **Sad** (04)
5. **Angry** (05)
6. **Fearful** (06)
7. **Disgust** (07)
8. **Surprised** (08)

---

## 📁 Repository Structure

```
multimodal-emotion-recognition/
│
├── dataset/                    # RAVDESS dataset directory (Actor_01, Actor_02...)
│   └── .gitkeep
├── models/                     # Saved model weight checkpoints (.keras)
│   └── .gitkeep
│
├── config.py                   # Centralized configuration, emotion labels & hyperparams
├── audio_model.py              # Audio feature extraction, 2D CNN architecture & training
├── visual_model.py             # Visual processing (Haar Cascade) & TimeDistributed CNN-GRU-Attention
├── multimodal_model.py         # Multimodal late fusion evaluation & confusion matrix
│
├── sunum.pdf                   # Graduation Project presentation slides
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git untracked patterns
├── LICENSE                     # MIT License
├── README_TR.md                # Turkish documentation
└── README.md                   # English documentation (Main)
```

---

## 📥 Dataset Setup (RAVDESS)

This project utilizes the internationally recognized **RAVDESS (The Ryerson Audio-Visual Database of Emotional Speech and Song)** dataset.

1. **Download the Dataset:**
   - Official Zenodo Repository: [RAVDESS on Zenodo](https://zenodo.org/records/1188976) or via Kaggle (`Audio_Speech_Actors_01-24` and `Video_Speech_Actor_01-24`).
2. **Directory Placement:**
   Extract and place the actor folders inside the `dataset/` directory:
   ```
   dataset/
     ├── Actor_01/
     │     ├── 01-01-01-01-01-01-01.mp4
     │     ├── 03-01-01-01-01-01-01.wav
     │     └── ...
     ├── Actor_02/
     └── ...
   ```
   *(To use an external dataset location, simply pass `--data_dir "/path/to/ravdess"` to any script).*

### RAVDESS Filename Convention:
Example filename: `03-01-03-01-02-01-12.wav`
- `03`: Modality (`01` = Full-AV, `02` = Video-only, `03` = Audio-only)
- `01`: Vocal Channel (`01` = Speech, `02` = Song)
- `03`: Emotion (`01`: Neutral, `02`: Calm, `03`: Happy, `04`: Sad, `05`: Angry, `06`: Fearful, `07`: Disgust, `08`: Surprised)
- `01`: Intensity (`01` = Normal, `02` = Strong)
- `02`: Statement (`01` or `02`)
- `01`: Repetition (`01` or `02`)
- `12`: Actor ID (`01` to `24`; Odd numbers = Male, Even numbers = Female)

---

## 🚀 Installation & Quickstart

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/Farukes/multimodal-emotion-recognition.git
cd multimodal-emotion-recognition

# Create and activate virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

### 2. Training the Models

#### Step 1: Train the Audio Model
```bash
python audio_model.py --epochs 120 --batch_size 16
```
*(Custom dataset path: `python audio_model.py --data_dir "/path/to/dataset"`)*

When training finishes, weights are saved to `models/sesmodelson.keras`, and loss/accuracy curves along with confusion matrices are plotted.

#### Step 2: Train the Visual Model
```bash
python visual_model.py --epochs 100 --batch_size 16
```
Trained weights are saved to `models/gorselson.keras`.

### 3. Evaluate Multimodal Late Fusion
Once both models are trained, evaluate late decision fusion across unseen test actors:
```bash
python multimodal_model.py
```
You can optionally adjust unimodal voting weights:
```bash
python multimodal_model.py --audio_weight 0.4 --visual_weight 0.6
```

---

## 🧠 Model Architectures

### Audio Model (2D CNN)
- **Input:** 3-channel spectrogram matrix `(128, 128, 3)` consisting of Log Mel-Spectrogram, Delta (velocity), and Delta-Delta (acceleration).
- **Backbone:** 3 Conv2D blocks (64, 128, 256 filters) with Batch Normalization, ReLU activation, MaxPooling2D, and Spatial Dropout.
- **Classifier:** Flatten -> Dense(512) -> BatchNorm -> Dropout(0.6) -> Softmax(8).

### Visual Model (TimeDistributed CNN + GRU + Attention)
- **Input:** Uniformly sampled 10 face frames cropped via Haar Cascade and resized to `(10, 64, 64, 3)`.
- **Spatial Feature Extractor:** TimeDistributed Conv2D blocks (32, 64, 128 filters) extracting frame-level visual embeddings.
- **Temporal Sequence Modeling:** 256-unit GRU (Gated Recurrent Unit) capturing transitions in facial expressions across time.
- **Self-Attention Mechanism:** Attention layer dynamically assigning higher weights to pivotal emotion-expressing frames.
- **Classifier:** Dense(512) -> BatchNorm -> Dropout(0.65) -> Softmax(8).

### Multimodal Late Fusion (Soft Voting)
$$\hat{y} = \arg\max \left( w_{\text{audio}} \cdot P_{\text{audio}} + w_{\text{visual}} \cdot P_{\text{visual}} \right)$$
Using equal weighting ($w_{\text{audio}} = 0.5$, $w_{\text{visual}} = 0.5$), the system achieves **81% overall accuracy** on unseen subjects.

---

## 📽️ Presentation Slides

The detailed project presentation slides, theoretical framework, architecture diagrams, and comparative confusion matrices are available in [`sunum.pdf`](sunum.pdf).

---

## 👨‍💻 Author & License

- **Author:** Ömer Faruk Eskitürk
- **License:** [MIT License](LICENSE) - Free for academic and open-source use.
