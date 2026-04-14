# 📷 Camera Fingerprinting — Phone Model Identification

> A computer vision system that identifies the smartphone model that captured an image using camera fingerprinting techniques — including PRNU extraction, CFA artifact analysis, and deep metric learning with ArcFace Loss. No EXIF metadata used.

---

## 🧠 Overview

Every smartphone camera leaves an invisible, device-specific signature in every photo it captures — embedded through the physical properties of its sensor, lens optics, and image processing pipeline.

This project builds a **Camera Fingerprinting System** that identifies the **phone model** that captured a given image purely from pixel-level and frequency-domain analysis — without relying on EXIF metadata.

---

## 🔍 How It Works

The system extracts multiple device-specific fingerprints from each image and fuses them into a deep metric learning classifier:

| Signal | What It Captures |
|---|---|
| **PRNU + Wiener Filter** | Unique sensor noise pattern from pixel-level manufacturing imperfections |
| **CFA / Demosaicing Artifacts** | Periodic interpolation patterns left by Bayer filter reconstruction |
| **ISP Pipeline Fingerprint** | Traces left by sharpening, noise reduction, and tone mapping |
| **Lens Aberration Profile** | Vignetting, distortion, and chromatic aberration unique to each lens-sensor stack |
| **JPEG Quantization Tables** | Manufacturer-specific DCT compression matrices embedded in the file |
| **Noise Statistics Model** | Shot noise and read noise coefficients unique to each sensor |

---

## ⚙️ Pipeline

```
Input Image
    │
    ├── Noise Residual Extraction (Wavelet Denoising)
    │       └── Wiener Filter → PRNU Estimate
    │
    ├── Frequency Analysis (FFT)
    │       └── CFA Artifact Map
    │
    ├── Spatial Analysis
    │       └── Lens Profile + ISP Traces
    │
    ├── JPEG Header Parsing
    │       └── Quantization Table Fingerprint
    │
    └── Feature Fusion → CNN Backbone → ArcFace Embedding
                                              │
                                        Phone Model ID
```

---

## 🏗️ Key Techniques

- **PRNU Extraction** — Wavelet denoising + Wiener filtering in the frequency domain to isolate sensor-level fixed pattern noise
- **PCE (Peak to Correlation Energy)** — Fingerprint matching and verification metric
- **ArcFace Loss** — Additive angular margin loss for metric learning; enforces consistent, geometrically uniform class separation on a unit hypersphere
- **Multi-signal Fusion** — Combines hand-crafted forensic features with learned deep features for robust identification

---

## 📁 Project Structure

```
camera-fingerprinting/
│
├── data/
│   ├── raw/                  # Raw images per device
│   └── processed/            # Preprocessed noise residuals
│
├── features/
│   ├── prnu.py               # PRNU + Wiener filter extraction
│   ├── cfa.py                # CFA / demosaicing artifact analysis
│   ├── isp.py                # ISP pipeline fingerprint
│   ├── lens.py               # Lens aberration profiling
│   └── quant.py              # JPEG quantization table parsing
│
├── models/
│   ├── backbone.py           # CNN feature extractor (ResNet / EfficientNet)
│   ├── arcface.py            # ArcFace loss implementation
│   └── classifier.py        # Full model pipeline
│
├── train.py                  # Training script
├── evaluate.py               # Evaluation and PCE scoring
├── inference.py              # Single image inference
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

```bash
Python >= 3.9
PyTorch >= 2.0
CUDA (recommended)
```

### Installation

```bash
git clone https://github.com/your-username/camera-fingerprinting.git
cd camera-fingerprinting
pip install -r requirements.txt
```

### Training

```bash
python train.py --data_dir ./data/raw --epochs 50 --batch_size 32
```

### Inference

```bash
python inference.py --image_path ./test.jpg
```

---

## 📊 Results

| Method | Accuracy |
|---|---|
| PRNU + Wiener (reference-based) | ~99%+ |
| CFA + ISP Features (reference-free) | ~87% |
| Deep Learning + ArcFace (fusion) | ~96% |

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python |
| Deep Learning | PyTorch |
| Computer Vision | OpenCV, scikit-image |
| Signal Processing | NumPy, SciPy |
| Denoising | Wavelet (scikit-image), BM3D |
| Model Backbone | ResNet / EfficientNet |
| Loss Function | ArcFace |

---

## ⚠️ Limitations

- PRNU degrades on images recompressed by social media platforms (WhatsApp, Instagram)
- AI-driven computational photography (e.g. Google Pixel Night Sight) aggressively suppresses sensor noise, reducing PRNU reliability
- Requires sufficient flat/uniform regions in the image for noise statistics modeling
- Closed-set identification only — unknown device models are not supported

---

