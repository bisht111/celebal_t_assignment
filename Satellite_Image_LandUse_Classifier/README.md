# 🛰️ Satellite Image Land-Use Classifier & Temporal Change Detector

## Overview

This project implements a deep learning system for satellite image analysis using **ResNet-18 Transfer Learning**. It performs:

- Land-use classification on satellite imagery
- Temporal change detection using feature embeddings
- Interactive Streamlit dashboard for inference

---

## Features

- Transfer Learning with ResNet-18
- Two-phase fine-tuning
- 10-class land-use classification
- 512-dimensional embedding extraction
- Cosine similarity based change detection
- ROC curve and threshold selection
- Interactive Streamlit dashboard
- EuroSAT evaluation
- UC Merced transfer evaluation

---

## Datasets

The datasets are **not included** because of their large size.

### EuroSAT RGB Dataset

Official Dataset

https://github.com/phelber/EuroSAT

Official Download

https://madm.dfki.de/downloads

Extract to:

```
data/
└── 2750/
```

---

### UC Merced Land Use Dataset

Official Dataset

https://weegee.vision.ucmerced.edu/datasets/landuse.html

Extract to:

```
data/
└── UCMerced_LandUse/
```

---

## Project Structure

```
Satellite_Image_LandUse_Classifier/
│
├── app/
├── checkpoints/
├── notebooks/
├── reports/
├── src/
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Installation

```bash
git clone https://github.com/bisht111/celebal_t_assignment.git

cd celebal_t_assignment/Satellite_Image_LandUse_Classifier

python -m venv venv

source venv/bin/activate

pip install -r requirements.txt
```

---

## Run Dashboard

```bash
streamlit run app/app.py
```

Upload:

- Before satellite image
- After satellite image

The dashboard displays:

- Predicted land-use class
- Confidence score
- Cosine similarity
- Change detection
- Heatmap

---

## Model

Backbone:

- ResNet-18 (ImageNet Pretrained)

Training Strategy

Phase 1

- Freeze backbone
- Train classifier head
- 3 epochs

Phase 2

- Unfreeze last two residual blocks
- Learning rate reduced by 10×
- Train 5 epochs

---

## Results

### EuroSAT

| Metric | Value |
|--------|------:|
| Accuracy | **96.98%** |
| Macro F1 | **96.92%** |

### Change Detection

| Metric | Value |
|--------|------:|
| ROC AUC | **0.966** |
| Threshold | **0.638** |

---

## Spatial Leakage

The RGB version of EuroSAT does not include geographic coordinates or region identifiers. Therefore, a true spatial block split could not be reproduced. The evaluation uses a random train/validation split and discusses this limitation.

---

## Future Improvements

- True temporal satellite imagery
- Geographic block split evaluation
- Grad-CAM visualization
- Siamese Network based change detection
- UMAP / t-SNE embedding visualization

---

## Author

Satellite Image Land-Use Classifier & Temporal Change Detector

Developed using PyTorch and Streamlit.
 

## Dataset

The datasets are **not included** in this repository because of their large size.

Please download them from the official sources.

### 1. EuroSAT RGB Dataset

* Official dataset page: [EuroSAT Dataset](https://unilnet.github.io/datasets.html?utm_source=chatgpt.com)
* Official GitHub repository: [EuroSAT GitHub Repository](https://github.com/phelber/EuroSAT?utm_source=chatgpt.com)

After downloading, extract the RGB dataset into:

```text
Satellite_Image_LandUse_Classifier/
└── data/
    └── 2750/
        ├── AnnualCrop/
        ├── Forest/
        ├── HerbaceousVegetation/
        ├── Highway/
        ├── Industrial/
        ├── Pasture/
        ├── PermanentCrop/
        ├── Residential/
        ├── River/
        └── SeaLake/
```

---

### 2. UC Merced Land Use Dataset

Download from the official UC Merced dataset page:

* [UC Merced Land Use Dataset](https://unilnet.github.io/datasets.html?utm_source=chatgpt.com)

Extract it into:

```text
Satellite_Image_LandUse_Classifier/
└── data/
    └── UCMerced_LandUse/
        └── Images/
            ├── agricultural/
            ├── airplane/
            ├── ...
            └── tenniscourt/
```

---

### Dataset Structure

The final project directory should look like:

```text
Satellite_Image_LandUse_Classifier/
│
├── app/
├── checkpoints/
├── notebooks/
├── reports/
├── src/
├── data/
│   ├── 2750/
│   └── UCMerced_LandUse/
├── README.md
└── requirements.txt
```

> **Note:** The datasets are publicly available and should be downloaded separately. They are excluded from this repository to keep the repository lightweight and comply with GitHub storage limits.
