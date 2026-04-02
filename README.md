# 🛡️ DeepShield AI — Deepfake Detection & Prevention

A comprehensive deepfake detection and prevention system powered by a **hybrid AI engine** combining neural networks with signal-processing forensics. Achieves **97.5% accuracy** on unseen test data.

## 📑 Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Evaluation & Benchmarks](#evaluation--benchmarks)
- [Tech Stack](#tech-stack)

## ✨ Features

- **Deepfake Detection** — Upload any face image and get a real/fake verdict with confidence score
- **Grad-CAM Heatmaps** — Visual explanation of which regions the AI flagged as suspicious
- **Image Protection** — Apply adversarial noise to protect your photos from being deepfaked
- **Hybrid Detection Engine** — Combines 4 complementary analysis methods for robust detection
- **Modern Web Interface** — Built with Next.js, featuring drag-and-drop upload

## 🧠 Architecture

The detection engine uses a **multi-signal ensemble** approach — no single model is trusted alone:

| Signal | Weight | Method | What It Detects |
|--------|--------|--------|-----------------|
| **Neural (ViT)** | 30% | Vision Transformer (`dima806/deepfake_vs_real_image_detection`) | Structural face anomalies |
| **ELA** | 30% | Error Level Analysis (JPEG recompression artifacts) | Compression inconsistencies |
| **Frequency** | 30% | 2D DCT high-frequency energy ratio | GAN spectral fingerprints |
| **Color** | 10% | HSV color space statistics | Color distribution anomalies |

### Why a Hybrid Approach?

Single neural models fail in different ways:
- **ViT models** miss high-quality StyleGAN3 fakes (they look too realistic)
- **ELA alone** can be fooled by post-processing
- **Frequency analysis alone** has edge cases

By combining all four signals, the system compensates for each method's weaknesses.

## 📁 Project Structure

```
├── run.py                          # 🚀 One-click launcher (starts both backend + frontend)
├── README.md
├── .gitignore
│
└── Code/
    ├── backend/
    │   ├── main.py                 # FastAPI server (API endpoints)
    │   ├── requirements.txt        # Python dependencies
    │   ├── services/
    │   │   ├── detection.py        # 🧠 Hybrid detection engine (ViT + ELA + Freq + Color)
    │   │   └── prevention.py       # 🛡️ Adversarial noise protection (FGSM)
    │   ├── benchmark.py            # 📊 Accuracy evaluation script
    │   ├── diagnose.py             # 🔬 Raw signal diagnostic tool
    │   ├── download_test_data.py   # ⬇️ Download calibration test set
    │   └── download_validation_data.py  # ⬇️ Download blind validation set
    │
    ├── frontend/
    │   ├── app/
    │   │   ├── page.tsx            # Main UI page
    │   │   ├── layout.tsx          # App layout
    │   │   └── globals.css         # Global styles
    │   ├── package.json            # Node.js dependencies
    │   ├── next.config.ts          # Next.js configuration
    │   ├── tailwind.config.ts      # Tailwind CSS configuration
    │   ├── tsconfig.json           # TypeScript configuration
    │   └── postcss.config.mjs      # PostCSS configuration
    │
    ├── train.py                    # Original MobileNetV2 training script
    ├── predict.py                  # Original prediction script
    └── app.py                      # Original Streamlit app (legacy)
```

## 🛠️ Setup & Installation

### Prerequisites
- Python 3.10+
- Node.js 18+
- pip

### Backend Setup

```bash
# Create virtual environment
cd Code
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r backend/requirements.txt
```

### Frontend Setup

```bash
cd Code/frontend
npm install
```

### One-Click Start

```bash
# From the project root
python3 run.py
```

This starts both services:
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000

## 🚀 Usage

1. **Start the application** using `python3 run.py`
2. **Open** http://localhost:3000 in your browser
3. **Upload** a face image (drag-and-drop or click to browse)
4. **View results**: Real/Fake verdict, confidence score, and heatmap visualization
5. **Protect images**: Use the Protection tab to add adversarial noise to your photos

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/detect` | Upload image for deepfake detection |
| `POST` | `/api/protect` | Apply adversarial protection to image |
| `GET` | `/api/health` | Health check |

## 📊 Evaluation & Benchmarks

### Accuracy Results (After Fine-Tuning)

| Dataset | Real Accuracy | Fake Accuracy | Overall |
|---------|--------------|---------------|---------|
| Calibration (Val Set) | 96% | 90% | **93.3%** |
| **Test Set (Held-out)** | **100%** | **100%** | **100.0%** |

*Note: Results achieved after fine-tuning ViT-Base on a curated dataset of 400 images using Apple Silicon GPU acceleration (MPS).*

### Test Data Sources
- **Real faces**: randomuser.me (high-resolution real photos)
- **Fake faces**: thispersondoesnotexist.com (StyleGAN-generated)

### Run Benchmarks Yourself

```bash
cd Code/backend

# Download test data
python3 download_test_data.py

# Download separate validation data (unseen)
python3 download_validation_data.py

# Run benchmark
python3 benchmark.py test_data
python3 benchmark.py validation_data

# View raw signal diagnostics
python3 diagnose.py test_data
```

## 🔧 Tech Stack

### Backend
- **FastAPI** — High-performance Python API framework
- **PyTorch + HuggingFace Transformers** — ViT model inference
- **TensorFlow/Keras** — EfficientNet for adversarial protection (FGSM)
- **OpenCV** — Image processing, ELA, frequency analysis
- **NumPy** — Signal processing and numerical computation

### Frontend
- **Next.js 15** — React framework with SSR
- **TypeScript** — Type-safe development
- **Tailwind CSS** — Utility-first styling
- **Framer Motion** — Smooth animations
- **Axios** — HTTP client for API calls

---

⭐ If you found this project useful, please consider giving it a star on GitHub!
