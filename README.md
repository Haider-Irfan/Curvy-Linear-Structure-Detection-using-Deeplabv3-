
# Curvy-Linear-Structure-Detection-using-Deeplabv3-
Semantic segmentation of cracks in concrete structures using DeepLabV3+ (ResNet50V2 backbone) — 98% accuracy, 0.83 test IoU. Built with TensorFlow/KerasCV.

Overview

Manual inspection of cracks in infrastructure is time-consuming, expensive, and prone to human error. This project automates crack detection using a DeepLabV3+ semantic segmentation model, classifying every pixel in a concrete surface image as either background or crack.

<img width="2968" height="766" alt="test_combo_0000" src="https://github.com/user-attachments/assets/f848aad1-49d3-4df5-bb63-fe81783edaec" />

<img width="2968" height="766" alt="test_combo_0038" src="https://github.com/user-attachments/assets/642be863-b39a-44ac-83c6-cd3af9807217" />

<img width="2968" height="766" alt="test_combo_0010" src="https://github.com/user-attachments/assets/b3c6fa57-1ae5-4f18-82fb-30c7a1806ea9" />

Key Results
Metric	Value
Test Accuracy	98%
Test Mean IoU	0.8324
Training Time	~1 hour (single T4 GPU)
Background Class F1	0.9816
Crack Class F1	0.8362
Table of Contents
Problem Statement
Dataset
Model Architecture
Training Configuration
Results
Limitations & Future Work
Tech Stack
Project Structure
Getting Started
Usage
License
Problem Statement

Manual crack inspection on infrastructure (bridges, pavements, buildings) is slow, costly, and inconsistent across inspectors. This project develops an automated, deep learning-based system for pixel-level crack segmentation that generalizes to unseen concrete surface images.

Objectives

Develop an automated crack detection system
Achieve high accuracy in pixel-level crack segmentation
Build a model that generalizes well to unseen data
Provide detailed performance metrics and visualizations
Dataset
Total images: 5,185 concrete surface images with binary crack masks (post-augmentation)
Split: 3,111 train / 1,037 validation / 1,037 test
Image size: 256×256 (resized to 128×128 for model input)
Classes: 2 — Background, Crack
Format: RGB images with binary segmentation masks
dataset/
├── train/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/

Note: update this structure to match your actual data directory layout in the repo.

Model Architecture
Architecture: DeepLabV3+
Backbone: ResNet50V2 (pretrained on ImageNet)
Input shape: 128×128×3
Output shape: 256×256×2
Total parameters: 39,186,624

<img width="416" height="222" alt="image" src="https://github.com/user-attachments/assets/a49a0e63-719e-4abd-9264-575b2e72a7a5" />



Key components

Encoder: ResNet50V2 for feature extraction
ASPP (Atrous Spatial Pyramid Pooling) for multi-scale context aggregation
Decoder: upsampling with skip connections
Output head: pixel-wise binary classification
Training Configuration
Setting	Value
Optimizer	Adam
Learning Rate	1e-4
Batch Size	16
Epochs	10
Loss Function	Sparse Categorical Crossentropy
Metrics	Accuracy, Mean IoU, Betti number, clDice

Data augmentation: random horizontal flip, random rotation, random brightness (±20%), random contrast (±20%)

Callbacks: ModelCheckpoint (best validation IoU), EarlyStopping (patience 10), ReduceLROnPlateau, TensorBoard

Results

Final training metrics (epoch 10)

Split	Loss	Accuracy	Mean IoU
Training	0.0381	98%	0.8553
Validation	0.0426	98%	0.8334
Test	0.0437	98%	0.8324

Per-class test performance

Class	Precision	Recall	F1-Score
Background	99%	99%	0.9816
Crack	84%	83%	0.8362

Qualitative observations: the model reliably detects visible cracks across varying lighting conditions, with weaker performance on very thin cracks (< 2–3 pixels wide).

Limitations & Future Work

The model struggles with extremely thin cracks due to a combination of factors:

Spatial resolution constraints at the 128×128 input size
Severe class imbalance between background and crack pixels
Downsampling in the encoder path losing fine-grained features
Standard cross-entropy loss under-weighting the minority crack class

Planned improvements

Higher-resolution model inputs
Specialized loss functions (Focal Loss / Dice Loss) to address class imbalance
Multi-scale feature fusion techniques
Tech Stack
Python 3.10
TensorFlow 2.15.0
KerasCV
Google Colab (NVIDIA T4 GPU, 16GB)
Project Structure
crack-segmentation-deeplabv3/
├── dataset/                # Training/validation/test data (not tracked in git)
├── notebooks/               # Training & evaluation notebooks
├── src/                      # Model, data pipeline, and training scripts
├── models/                  # Saved model checkpoints (not tracked in git)
├── results/                  # Metrics, plots, and qualitative outputs
├── requirements.txt
└── README.md

Adjust this to match your actual repository layout before pushing.

Getting Started
bash
git clone https://github.com/<your-username>/crack-segmentation-deeplabv3.git
cd crack-segmentation-deeplabv3
pip install -r requirements.txt
Usage
bash
# Train the model
python src/train.py --epochs 10 --batch_size 16 --lr 1e-4

# Evaluate on the test set
python src/evaluate.py --checkpoint models/best_model.h5

# Run inference on a single image
python src/predict.py --image path/to/image.png --checkpoint models/best_model.h5

Replace these with your actual script names/entry points.

License

This project is licensed under the MIT License — see the LICENSE file for details.

Acknowledgments
ResNet50V2 pretrained weights (ImageNet)
Concrete crack segmentation dataset (credit the original dataset source here)
