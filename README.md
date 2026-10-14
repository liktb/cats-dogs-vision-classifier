# Cats vs. Dogs Image Classification Pipeline

A production-grade Deep Learning computer vision pipeline built using PyTorch, transfer learning with **EfficientNet-B0**, and accelerated locally via NVIDIA CUDA. Designed for high accuracy and rapid inference on custom image classification tasks.

---

## Model Performance & Evaluation Metrics
Evaluated on **4,653 unseen test images**:

- **Overall Test Accuracy:** `96.20%`
- **Class Metrics:**
  - **Cat:** Precision: `0.98` | Recall: `0.95` | F1-Score: `0.96`
  - **Dog:** Precision: `0.95` | Recall: `0.98` | F1-Score: `0.96`

---

## Project Structure
```text
trial/
│
├── cats_dogs_244/            # Dataset folder (train / test splits)
├── train_cv.py               # Data augmentation, transfer learning setup, training loop & artifact saving
├── predict_cv.py             # Single-image inference script loading the saved .pt weights
├── evaluate_cv.py            # Comprehensive evaluation script (Accuracy, Classification Report, Confusion Matrix)
├── vision_model_best.pt      # Serialized PyTorch model weights & class mapping dictionary
├── requirements.txt          # Locked python package dependencies
└── README.md                 # Project documentation & metrics summary

How to Run Locally
Clone or download this repository.(vision_model_best.pt is a provided pretrained model,(5 EPOCH))

Install dependencies:
   pip install -r requirements.txt

Train the model (Generates vision_model_best.pt):Bash
   python train_cv.py

Evaluate model performance across the test set:
   python evaluate_cv.py

Run single-image prediction:
   python predict_cv.py
```
## Contributors
- **Dataset Curation & Train/Test Split:** [Aaryan Kalra](https://github.com/aaryankalra1510-lgtm)
- **Model Pipeline & Training:** [Ryan Dinto](https://github.com/liktb)
