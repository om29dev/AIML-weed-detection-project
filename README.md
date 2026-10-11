# MH-Weed16: YOLO26 Weed Detection and Agronomic Vision Pipeline

Computer vision pipeline for precision agriculture, weed identification, multi-modal agronomic feature extraction, and edge deployment using Ultralytics YOLO26 on the MH-Weed16 dataset.

Dataset Source: [SayaliS069 / MH-Weed16 on Kaggle](https://www.kaggle.com/datasets/sayalis069/mh-weed16)

---

## Table of Contents

- [Overview](#overview)
- [Dataset Specifications](#dataset-specifications)
- [Weed Species Taxonomy](#weed-species-taxonomy)
- [Architectural Ablation and Edge Benchmarks](#architectural-ablation-and-edge-benchmarks)
- [Multi-Modal Feature Engineering](#multi-modal-feature-engineering)
- [Pipeline Workflow](#pipeline-workflow)
- [Repository Structure](#repository-structure)
- [Installation and Setup](#installation-and-setup)
- [Usage Guide](#usage-guide)
- [Edge Deployment with ONNX Runtime](#edge-deployment-with-onnx-runtime)
- [Contributing](#contributing)
- [References and Citation](#references-and-citation)

---

## Overview

Weed infestation reduces crop yields through resource competition for soil nutrients, moisture, and sunlight. Automating real-time weed management on agricultural machinery (smart sprayers, field rovers, and UAVs) requires models that balance high detection accuracy on varied leaf morphologies with minimal latency and memory footprint on low-power edge hardware.

This repository provides an end-to-end agronomic vision framework:

- **Object Detection**: YOLO26 custom architectures tailored for real-time bounding box localization of 15 weed classes.
- **Architectural Ablations**: 7 custom network designs benchmarked for parameters, GFLOPs, ONNX model size, and CPU/NPU runtime latency.
- **Agronomic Feature Extraction**: 29 engineered descriptors combining spatial geometry, RGB/HSV color metrics, texture operators, and spectral vegetation indices (ExG, ExR, NDI, MExG).
- **Latent Embedding and Anomaly Auditing**: Deep visual feature vector extraction via YOLO26 backbones, nearest-neighbor similarity search, PCA/t-SNE manifold projections, and Isolation Forest anomaly filtering.
- **High-Resolution Inference**: Slicing Aided Hyper Inference (SAHI) to resolve small emerging weeds in high-resolution field captures without loss of spatial detail.
- **Edge Deployment**: FP32 and FP16 ONNX models with dynamic batching optimized for embedded inference engines.

---

## Dataset Specifications

The MH-Weed16 dataset consists of field captures taken in crop-weed environments across Maharashtra, India.

- **Sensor**: Intel RealSense Depth Camera
- **Resolution**: 1920 x 1080 pixels
- **Total Images**: 6,656 raw field captures
- **Annotated Instances**: 59,010 bounding box labels
- **Format**: YOLO normalized coordinates `(class_id, x_center, y_center, width, height)`
- **Data Partitions**:
  - Training split (`train.txt`): 5,324 images (80%)
  - Validation split (`val.txt`): 665 images (10%)
  - Held-out test split (`test.txt`): 667 images (10%)

---

## Weed Species Taxonomy

The dataset annotates 15 botanical weed species commonly found in western and central Indian agricultural fields:

| Class ID | Species Name | Botanical Name | Common English Name | Regional Indian Name |
|:---:|:---|:---|:---|:---|
| 0 | `kena` | *Commelina benghalensis* | Benghal Dayflower / Tropical Spiderwort | Kena / Kanka |
| 1 | `lavhala` | *Cyperus rotundus* | Purple Nutsedge / Nut Grass | Lavhala / Motha |
| 2 | `lambs_quarters` | *Chenopodium album* | Lamb's Quarters / Goosefoot | Bathua / Chandan Batwa |
| 3 | `little_mallow` | *Malva parviflora* | Little Mallow / Cheeseweed | Khaparkhuti |
| 4 | `moti_dudhi` | *Euphorbia hirta* | Asthma-plant / Hairy Spurge | Moti Dudhi / Dudhani |
| 5 | `obscure_morning_glory` | *Ipomoea obscura* | Obscure Morning Glory | Pili Bhamri |
| 6 | `asian_pigeonwings` | *Clitoria ternatea* | Butterfly Pea / Asian Pigeonwings | Gokarna / Aparajita |
| 7 | `bilayat` | *Parthenium hysterophorus* | Santa Maria Feverfew / Carrot Grass | Bilayat / Congress Gavat |
| 8 | `choti_dudhi` | *Euphorbia microphylla* | Small Spurge / Thyme-leafed Spurge | Choti Dudhi |
| 9 | `digitaria` | *Digitaria sanguinalis* | Large Crabgrass / Finger Grass | Digitaria / Karkat |
| 10 | `gajar_gavat` | *Parthenium hysterophorus* (variant) | Wild Carrot / Bitter Weed | Gajar Gavat |
| 11 | `graceful_sandmat` | *Euphorbia hypericifolia* | Graceful Sandmat / Large Spotted Spurge | Dudhali |
| 12 | `sicklepod` | *Senna obtusifolia* | Sicklepod / Coffee Weed | Tarota / Takla |
| 13 | `harali` | *Cynodon dactylon* | Bermuda Grass / Couch Grass | Harali / Durva |
| 14 | `dwarf_cassia` | *Chamaecrista pumila* | Dwarf Cassia | Choti Tarota |

Configuration mapping is maintained in [mhweed16.yaml](file:///d:/proj/MHWeed16/mhweedaiml/mhweed16.yaml).

---

## Architectural Ablation and Edge Benchmarks

Standard YOLO26n employs dense 3x3 convolutions and a convolutional multi-head self-attention module (`C2PSA`) at the deep P5 level. While effective for representation capacity, multi-head attention decomposes into fragmented reshape, transpose, and matrix multiplication operators in ONNX, creating execution overhead on memory-constrained edge hardware.

Seven architectural configurations were designed, compiled, and benchmarked on ONNX Runtime (CPU) at 640x640 input resolution:

1. **`baseline_yolo26n`**: Standard YOLO26n baseline with full `C2PSA` self-attention.
2. **`ghost_yolo26n`**: Replaces standard dense convolution blocks with `GhostConv` and `C3Ghost`, generating intrinsic feature maps using cheap linear transformations.
3. **`dw_yolo26n`**: Employs Depthwise Separable Convolutions (`DWConv`) across backbone and neck, decoupling spatial filtering from channel mixing.
4. **`ghost_dw_hybrid_yolo26n`**: Combines Ghost feature generation in the backbone with depthwise processing in the feature pyramid neck.
5. **`no_psa_yolo26n`**: Strips `C2PSA` self-attention, replacing it with pure `SPPF` pooling to produce a fully fused ONNX graph.
6. **`slim_neck_yolo26n`**: Compresses channel capacity across the PANet neck stages to reduce memory bandwidth.
7. **`micro_scaled_yolo26n`**: Compound-scaled ultra-compact model with reduced depth and width multipliers (`scale=[0.33, 0.25, 512]`).

### Benchmark Results

Results evaluated on 640x640 input tensors (sorted by FP16 model size):

| Model Variant | Layers | Parameters | GFLOPs | FP32 Size (MB) | FP16 Size (MB) | Size Reduction (%) | Latency FP32 (ms) | Throughput FP32 (FPS) | Latency FP16 (ms) | Throughput FP16 (FPS) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **`micro_scaled_yolo26n`** | 209 | **914,985** | **4.58** | **3.72** | **1.90** | -62.8% | **24.14** | **41.4** | 28.63 | 34.9 |
| **`ghost_dw_hybrid_yolo26n`** | 201 | 1,454,813 | 5.09 | 5.76 | 2.93 | -42.7% | 24.44 | 40.9 | 31.24 | 32.0 |
| **`ghost_yolo26n`** | 211 | 1,489,173 | 4.94 | 5.90 | 3.00 | -41.3% | 25.21 | 39.7 | **26.85** | **37.2** |
| **`dw_yolo26n`** | 163 | 1,680,925 | 4.63 | 6.63 | 3.36 | -34.2% | 25.40 | 39.4 | 27.10 | 36.9 |
| **`slim_neck_yolo26n`** | 197 | 1,956,469 | 5.58 | 7.69 | 3.89 | -23.9% | 25.85 | 38.7 | 28.85 | 34.7 |
| **`no_psa_yolo26n`** | 163 | 2,343,037 | 6.25 | 9.15 | 4.62 | -9.6% | 27.35 | 36.6 | 29.53 | 33.9 |
| **`baseline_yolo26n`** | 181 | 2,592,765 | 6.51 | 10.12 | 5.11 | Reference | 32.08 | 31.2 | 29.15 | 34.3 |

Benchmark data logged in [runs/ablation_results.csv](file:///d:/proj/MHWeed16/mhweedaiml/runs/ablation_results.csv).

### Key Findings

- **Balanced Edge Recommendation (`ghost_yolo26n`)**: Reduces parameters by 42.6% (1.49M vs 2.59M) and cut ONNX footprint to 3.00 MB in FP16, while improving CPU throughput from 31.2 FPS to 39.7 FPS.
- **Ultra-Low-Power Recommendation (`micro_scaled_yolo26n`)**: Reaches sub-1M parameters (914K) and a 1.90 MB FP16 model weight, achieving the lowest FP32 latency (24.14 ms / 41.4 FPS).
- **Attention Stripping (`no_psa_yolo26n`)**: Removing `C2PSA` eliminates fragmented ONNX subgraphs, speeding up CPU inference by 4.73 ms per frame without altering convolutional feature hierarchies.

Promoted edge weights are stored in [weights/](file:///d:/proj/MHWeed16/mhweedaiml/weights):
- `weights/best_edge_yolo_fp16.onnx` (Balanced Ghost model, FP16)
- `weights/best_edge_yolo_fp32.onnx` (Balanced Ghost model, FP32)
- `weights/micro_edge_yolo_fp16.onnx` (Micro-scaled model, FP16)

---

## Multi-Modal Feature Engineering

In addition to bounding box detection, the pipeline extracts a 29-dimensional tabular feature representation for every weed instance. This feature space combines morphological geometry, color distributions, spatial texture, and agronomic spectral indices.

Feature dataset: [weed_feature_vectors.csv](file:///d:/proj/MHWeed16/mhweedaiml/weed_feature_vectors.csv) (59,010 samples).

### Feature Schema

| Group | Column Name | Mathematical Formulation / Description | Agronomic Context |
|:---|:---|:---|:---|
| **Identity** | `image_name` | Source image filename | Traceability |
| | `class_id` | Class index (0 to 14) | Target ground truth |
| | `species` | Botanical species name | Categorical label |
| **Geometry** | `width` | Normalized bbox width $w = W_{box} / W_{img}$ | Spatial scale |
| | `height` | Normalized bbox height $h = H_{box} / H_{img}$ | Vertical growth |
| | `aspect_ratio` | $w / (h + 10^{-6})$ | Broadleaf vs grass differentiation |
| | `area` | $w \times h$ | Relative canopy coverage |
| | `perimeter` | $2(w + h)$ | Canopy boundary span |
| | `center_dist` | $\sqrt{(x_c - 0.5)^2 + (y_c - 0.5)^2}$ | Spatial offset from optical axis |
| | `boxes_in_frame` | Total count of detections in current image | Local weed infestation density |
| | `area_to_aspect_ratio` | $\text{area} / (\text{aspect\_ratio} + 10^{-6})$ | Shape-normalized biomass proxy |
| **Color Distribution** | `r_mean`, `g_mean`, `b_mean` | $\mu_R, \mu_G, \mu_B \in [0, 255]$ | Mean RGB reflectance of patch |
| | `r_std`, `g_std`, `b_std` | $\sigma_R, \sigma_G, \sigma_B$ | Color variance across foliage |
| | `hue_mean` | HSV Hue channel mean $\in [0, 180]$ | Dominant leaf hue angle |
| | `sat_mean` | HSV Saturation channel mean $\in [0, 255]$ | Foliage color purity |
| | `val_mean` | HSV Value channel mean $\in [0, 255]$ | Canopy brightness |
| **Spectral Indices** | `exg_mean` | $\text{ExG} = 2G - R - B$ | Excess Green: separates vegetative tissue from soil |
| | `exg_std` | Standard deviation of ExG values in patch | Canopy density heterogeneity |
| | `exr_mean` | $\text{ExR} = 1.4R - G$ | Excess Red: soil and residue background signal |
| | `ndi_mean` | $\text{NDI} = (G - R) / (G + R + 10^{-6})$ | Normalized Difference Index: greenness level |
| | `mexg_mean` | $\text{MExG} = 1.262G - 0.866R - 0.411B$ | Modified Excess Green: shadow-tolerant index |
| | `is_high_exg` | Binary flag: $\text{exg\_mean} > 15.0$ | High-vigor vegetative indicator |
| **Texture & Focus** | `contrast_std` | Standard deviation of grayscale pixel intensities | Surface texture contrast |
| | `laplacian_var` | $\text{Var}(\nabla^2 I)$ | Laplacian variance: focus sharpness and micro-venation |
| | `sobel_energy` | $\text{Mean}(\sqrt{G_x^2 + G_y^2})$ | Edge strength: leaf serration and boundary clarity |

### Analytical Extensions in the Pipeline

- **Principal Component Analysis (PCA)**: Evaluates feature variance explained by top components; identifies morphological vs spectral loadings.
- **t-Distributed Stochastic Neighbor Embedding (t-SNE)**: Projects non-linear 2D clusters demonstrating species separability.
- **Anomaly Detection (Isolation Forest)**: Prunes mislabeled annotations, extreme aspect ratio distortions, and motion-blurred outliers.
- **Deep Latent Embeddings**: Extracts 256-dimensional feature vectors $\mathbf{z} \in \mathbb{R}^{256}$ directly from pre-trained YOLO backbones (`model.embed`), enabling content-based visual similarity search and nearest-neighbor retrieval.
- **Anchor Geometry Clustering**: K-Means clustering ($K=6$) over $(w, h)$ coordinate space to model optimal bounding box prior scales.

---

## Pipeline Workflow

The primary end-to-end pipeline is documented in [MHWeed16_YOLO26_Detection.ipynb](file:///d:/proj/MHWeed16/mhweedaiml/MHWeed16_YOLO26_Detection.ipynb):

1. **Hardware Diagnostic**: Auto-detects execution targets (Intel Arc XPU, NVIDIA CUDA, Apple MPS, or CPU).
2. **Zero-Duplication Data Linking**: Uses NTFS directory junctions / POSIX symlinks to link raw datasets without storage duplication.
3. **Exploratory Data Analysis (EDA)**: Computes class frequencies, box geometry histograms, and ground-truth verification overlays.
4. **Feature Engineering**: Calculates geometric, color, texture, and spectral vegetation metrics; saves `weed_feature_vectors.csv`.
5. **PCA, t-SNE, and Isolation Forest**: Reduces dimensionality, inspects feature loadings, and filters anomalous bounding boxes.
6. **Deep Feature Retrieval**: YOLO26 backbone latent vector extraction and cosine similarity search.
7. **Architectural Ablation Suite**: Compiles 7 model variants, exports to ONNX (FP32/FP16), and benchmarks latency and throughput.
8. **Genetic Hyperparameter Evolution**: Optional GA search over learning rates, augmentation weights, and box loss gains.
9. **Two-Phase Accelerated Training**:
   - *Phase 1 (Warmup)*: Freeze backbone layers 0-10 (`freeze=11`, epochs 1-20) to quickly align neck and detection heads.
   - *Phase 2 (Fine-tuning)*: Unfreeze all layers (`freeze=0`, epochs 21-50, `lr0=0.0005`) for fine-grained leaf venation adaptation.
10. **Test Set Evaluation**: Validates against held-out `test.txt` to report $mAP_{50}$ and $mAP_{50-95}$.
11. **High-Resolution SAHI Inference**: Slices 1080p images into overlapping 640x640 patches to identify small weeds.
12. **Edge ONNX Export**: Produces dynamic-batch, FP16 ONNX models for field hardware.

---

## Repository Structure

```
D:\proj\MHWeed16\mhweedaiml\
│
├── MHWeed16_YOLO26_Detection.ipynb    # Master Jupyter pipeline notebook
├── mhweed16.yaml                      # YOLO dataset specification (15 classes)
├── weed_feature_vectors.csv           # Cleaned 29-D feature matrix (59,010 samples)
├── mhweed16_extracted_features.csv    # Extracted feature records with center coords
├── train.txt                          # 80% Training split image index (5,324 paths)
├── val.txt                            # 10% Validation split image index (665 paths)
├── test.txt                           # 10% Held-out test split image index (667 paths)
├── yolo26n.pt                         # Pre-trained YOLO26 nano base weights
├── README.md                          # Project documentation
├── CONTRIBUTING.md                    # Contribution guidelines and engineering standards
│
├── configs/
│   └── models/                        # Custom YOLO26 architectural specifications
│       ├── baseline_yolo26n.yaml      # Baseline model with full C2PSA self-attention
│       ├── ghost_yolo26n.yaml         # Ghost convolution architecture
│       ├── dw_yolo26n.yaml            # Depthwise separable convolution architecture
│       ├── ghost_dw_hybrid_yolo26n.yaml # Hybrid Ghost + Depthwise architecture
│       ├── no_psa_yolo26n.yaml        # Self-attention stripped architecture
│       ├── slim_neck_yolo26n.yaml     # Slim PANet feature pyramid architecture
│       └── micro_scaled_yolo26n.yaml  # Micro-scale compound scaled architecture
│
├── weights/                           # Exported edge deployment weights
│   ├── best_edge_yolo_fp16.onnx       # Balanced Ghost YOLO26n (FP16, 3.00 MB)
│   ├── best_edge_yolo_fp32.onnx       # Balanced Ghost YOLO26n (FP32, 5.90 MB)
│   ├── micro_edge_yolo_fp16.onnx      # Micro-scaled YOLO26n (FP16, 1.90 MB)
│   └── yolo26n.pt                     # Base PyTorch checkpoint
│
├── runs/
│   ├── ablation_results.csv           # Comprehensive latency, size, parameter benchmark
│   └── detect/                        # Training runs, validation plots, and checkpoints
│       └── ablation_baseline_yolo26n/ # Baseline model training run artifacts
│
├── MH-Weed16/                         # Raw dataset root (downloaded from Kaggle)
│   ├── Crop with Weeds/               # Field captures and annotations
│   └── Individual Weed Species/       # Isolated weed species captures
│
├── images -> [MH-Weed16/...]          # NTFS Junction / symlink to raw image directory
├── labels -> [MH-Weed16/...]          # NTFS Junction / symlink to raw labels directory
├── annotations -> labels              # Alias junction for label annotations
│
└── archive/                           # Legacy experiments (YOLO11 notebook and weights)
```

---

## Installation and Setup

### Prerequisites

- Windows 10/11, Linux, or macOS
- Python 3.10 or higher
- PowerShell (on Windows) or Bash (on Linux)
- Git

### Environment Setup (PowerShell)

Create and activate a virtual environment:

```powershell
# Create virtual environment
python -m venv .venv

# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Upgrade pip
python -m pip install --upgrade pip
```

### Install Dependencies

```powershell
pip install ultralytics torch torchvision opencv-python onnx onnxruntime sahi scikit-learn pandas numpy matplotlib seaborn pyyaml tqdm
```

For GPU acceleration:
- **NVIDIA CUDA**: Install the matching PyTorch CUDA build from [pytorch.org](https://pytorch.org/).
- **Intel Arc (XPU)**: Install `intel-extension-for-pytorch` matching your oneAPI and PyTorch installation.

### Dataset Configuration (Zero-Duplication Setup)

Download and extract the dataset into `MH-Weed16/` in the project root:

```powershell
# Verify dataset location
Test-Path ".\MH-Weed16\Crop with Weeds"
```

Create directory junctions on Windows to expose `images/` and `labels/` without duplicating disk storage:

```powershell
cmd /c mklink /J "images" "MH-Weed16\Crop with Weeds\intel Real Sense Depth_Clicks\intel Real Sense Depth_Clicks"
cmd /c mklink /J "labels" "MH-Weed16\Crop with Weeds\intel Real Sense Depth_Annotations\intel Real Sense Depth_Annotations"
cmd /c mklink /J "annotations" "labels"
```

Verify link status:

```powershell
Get-Item images, labels, annotations | Select-Object Name, Target, LinkType
```

---

## Usage Guide

### 1. Running the Pipeline Notebook

Launch Jupyter and open the pipeline notebook:

```powershell
jupyter notebook MHWeed16_YOLO26_Detection.ipynb
```

Execute cells sequentially to run data validation, feature extraction, ablation profiling, training, and edge export.

### 2. Multi-Modal Feature Extraction

To extract the 29-D feature vectors programmatically without opening the notebook:

```python
import pandas as pd
from pathlib import Path

# Load extracted weed feature vectors
df_features = pd.read_csv("weed_feature_vectors.csv")
print(f"Loaded {len(df_features)} weed feature vectors across {df_features['species'].nunique()} species.")
print(df_features[["species", "area", "aspect_ratio", "exg_mean", "laplacian_var"]].head())
```

### 3. Model Training via Python Script / Ultralytics API

Train any custom architecture variant using the provided YAML configuration:

```python
from ultralytics import YOLO

# Select custom architecture configuration
model = YOLO("configs/models/ghost_yolo26n.yaml")

# Train on MH-Weed16 dataset
results = model.train(
    data="mhweed16.yaml",
    epochs=50,
    batch=8,
    imgsz=640,
    optimizer="AdamW",
    project="runs/detect",
    name="ghost_yolo26n_run",
    device=0  # or 'xpu' / 'cpu'
)
```

### 4. Evaluating on Test Split

```python
from ultralytics import YOLO

model = YOLO("weights/best_edge_yolo_fp32.onnx")
metrics = model.val(
    data="mhweed16.yaml",
    split="test",
    imgsz=640
)

print(f"Test mAP50:    {metrics.box.map50:.4f}")
print(f"Test mAP50-95: {metrics.box.map:.4f}")
```

### 5. High-Resolution SAHI Sliced Inference

For high-resolution (1920x1080) field images where small weed seedlings need detection:

```python
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

detection_model = AutoDetectionModel.from_pretrained(
    model_type="yolov8",  # Compatible with YOLO26 export
    model_path="weights/best_edge_yolo_fp32.onnx",
    confidence_threshold=0.25,
    device="cpu"
)

result = get_sliced_prediction(
    "images/sample_field_capture.jpeg",
    detection_model,
    slice_height=640,
    slice_width=640,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2
)

result.export_visuals(export_dir="runs/detect/sahi_output")
```

---

## Edge Deployment with ONNX Runtime

The exported models run on embedded processors (NVIDIA Jetson, Intel Core Ultra / Arc, Raspberry Pi, and field industrial PCs) via ONNX Runtime without PyTorch dependencies.

### Minimal Python Inference Example

```python
import cv2
import numpy as np
import onnxruntime as ort

# Load ONNX session
model_path = "weights/best_edge_yolo_fp32.onnx"
session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name

# Preprocess image to 640x640 RGB tensor
image = cv2.imread("images/sample_image.jpeg")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
resized = cv2.resize(image_rgb, (640, 640))
input_tensor = resized.transpose(2, 0, 1).astype(np.float32) / 255.0
input_tensor = np.expand_dims(input_tensor, axis=0)

# Run inference
outputs = session.run(None, {input_name: input_tensor})
detections = outputs[0]  # Shape: (1, 19, 8400) -> [xc, yc, w, h, 15 class scores]

print("Inference successful. Raw detection tensor shape:", detections.shape)
```

---

## Contributing

Contributions to the pipeline are welcome. Please refer to [CONTRIBUTING.md](file:///d:/proj/MHWeed16/mhweedaiml/CONTRIBUTING.md) for full guidelines on:

- **Development Setup**: Virtual environment creation, dependency installation, and zero-duplication dataset junctions.
- **Branch and Commit Conventions**: Standardized branch naming (`feature/`, `fix/`, `perf/`, `docs/`) and imperative commit messages.
- **Notebook Hygiene**: Stripping redundant notebook outputs and execution counts in [MHWeed16_YOLO26_Detection.ipynb](file:///d:/proj/MHWeed16/mhweedaiml/MHWeed16_YOLO26_Detection.ipynb) to maintain clean Git diffs.
- **Model Submissions**: Adding new architecture YAML files to [configs/models/](file:///d:/proj/MHWeed16/mhweedaiml/configs/models), validating ONNX exports, and updating benchmark metrics in [runs/ablation_results.csv](file:///d:/proj/MHWeed16/mhweedaiml/runs/ablation_results.csv).
- **Feature Extraction Standards**: Maintaining the 29-D feature schema, normalized bounding box coordinates, and epsilon protections in [weed_feature_vectors.csv](file:///d:/proj/MHWeed16/mhweedaiml/weed_feature_vectors.csv).
- **Pull Request Checklist**: PR descriptions, benchmark verification, and review workflows.

---

## References and Citation

If using this project or the MH-Weed16 dataset in research or commercial applications, cite the source dataset:

```bibtex
@dataset{mhweed16_dataset,
  author    = {Sayali Sonavane},
  title     = {MH-Weed16: Multi-Class Agricultural Weed Dataset with RealSense Captures},
  year      = {2024},
  publisher = {Kaggle},
  url       = {https://www.kaggle.com/datasets/sayalis069/mh-weed16}
}
```

Frameworks used:
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [SAHI: Slicing Aided Hyper Inference](https://github.com/obss/sahi)
- [ONNX Runtime](https://github.com/microsoft/onnxruntime)
