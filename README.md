# No-Reference Image Quality Assessment via Frequency-Amplitude Decoupling and Dynamic Ranking Regularization

> **🔒 Peer Review Statement**
> This repository contains the official **inference code** and **pre-trained models** provided strictly for the peer-review process of the manuscript submitted to *Multimedia Systems*. 
> 
> To maintain double-blind integrity, all author identities, affiliations, and training scripts (including the *Dynamic Intra-Batch Ranking Regularization* module) have been explicitly removed. The full training framework will be fully open-sourced upon paper acceptance.

---

## 📝 Overview

Current No-Reference Image Quality Assessment (NR-IQA) models often suffer from semantic content bias during spatial feature learning. To address this, we propose a novel framework that decouples semantic interference from pure degradation characteristics. 

This repository provides the core implementation of the **Frequency-Amplitude Context Bridge (FACB)** module (see `models/maniqa.py`). Unlike conventional spatial-only attention mechanisms, our FACB operates sequentially:
1. It projects spatial features into the frequency domain via Orthogonal 2D-FFT.
2. It isolates the **Amplitude** spectrum to robustly capture intrinsic distortion energy (e.g., noise, blur, compression artifacts).
3. It adaptively modulates spatial features using amplitude-derived weights, successfully suppressing phase-dominated semantic structures without heavy computational overhead.

## 📁 Repository Structure

The directory is structured as follows to facilitate easy evaluation:

```text
NR-IQA-Inference/
├── checkpoints/          # Directory to store the downloaded pre-trained weights
│   └── best_model.pt     # (Requires manual download, see instructions below)
├── models/
│   ├── maniqa.py         # Core model architecture including the FACB module
│   └── swin.py           # Swin Transformer backbone dependencies
├── utils/                # Utility scripts for data processing and generic operations
├── config.py             # Configuration parameters for model inference
├── inference.py          # Entry script for single-image quality assessment
├── requirements.txt      # Python environment dependencies
└── test.jpg              # Sample distorted image for quick testing
```

## 🛠️ Prerequisites

Ensure your environment meets the following requirements. We recommend using a virtual environment (e.g., Anaconda).

*   Python >= 3.8
*   PyTorch >= 2.0.0 (CUDA highly recommended for optimal FFT performance)
*   Torchvision

Install the required dependencies using:

```bash
pip install -r requirements.txt
```
*(Key dependencies include: `timm`, `einops`, `Pillow`, `numpy`, `opencv-python`)*

## 🚀 Quick Start (Inference)

We provide a streamlined inference script to evaluate the predicted perceptual quality score of any given distorted image. 

### Step 1: Download Pre-trained Weights
Download the pre-trained weights (`best_model.pt`) evaluated and achieved SOTA on robust benchmark datasets. 

*   **Baidu Netdisk (百度网盘):** [Download Link](https://pan.baidu.com/s/1QsxX0e8E0b9MmVydtBR3UA) 
*   **Extraction Code (提取码):** `1234`

After downloading, place the `best_model.pt` file directly inside the `./checkpoints/` directory.

### Step 2: Run Single-Image Evaluation

> **💡 Note on Signal Processing:** In our `inference.py`, we intentionally use `CenterCrop` instead of standard `Resize`. Interpolation during resizing inherently destroys high-frequency distortion characteristics (such as blockiness or Gaussian noise). Cropping perfectly aligns with our frequency-amplitude decoupling logic to capture authentic degradation energy.

You can test the model on the provided `test.jpg` or any custom image. Run the following command in your terminal:

```bash
python inference.py --img_path ./test.jpg --weight_path ./checkpoints/best_model.pt --device cuda
```

**Expected Output:**
The script will perform a deterministic center-crop preprocessing, followed by a forward pass through the FACB-enhanced architecture.

```text
🚀 Initialize Inference on device: CUDA
📦 Loading pre-trained weights from: ./checkpoints/best_model.pt
🖼️ Processing image: ./test.jpg
--------------------------------------------------
✅ Prediction Complete!
📈 Predicted Quality Score: 0.xxxx
--------------------------------------------------
```

## 📜 License & Citation

*   **License:** This code is provided **exclusively for peer review**. Unauthorized distribution, modification, or commercial use is strictly prohibited prior to the official publication of the manuscript.
*   **Citation:** The citation information and the complete training codebase will be updated upon the acceptance of the paper.
