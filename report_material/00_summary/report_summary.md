# Research Report Summary
## MH-Weed16: Lightweight YOLO26-Based Weed Detection with Architectural Ablations for Edge Deployment

---

## 1. Project Overview

This project presents an end-to-end computer vision pipeline for **automated weed detection in precision agriculture** using a custom-trained YOLO26 object detection model on the **MH-Weed16 dataset**. The primary contributions are:

- Detection of **15 botanical weed species** commonly found in Maharashtra, India using bounding box localization.
- A systematic **architectural ablation study** across 7 custom YOLO26 network designs, benchmarked for parameters, GFLOPs, ONNX model size, CPU latency, and throughput.
- A **multi-modal agronomic feature extraction** pipeline producing 29 engineered descriptors per weed instance.
- A **two-phase transfer learning** training strategy (frozen backbone warmup → full fine-tuning).
- **Edge deployment** via ONNX Runtime (FP32 and FP16) targeting low-power embedded hardware.

---

## 2. Dataset Specifications

| Property | Value |
|:---|:---|
| **Dataset Name** | MH-Weed16 |
| **Source** | Kaggle — SayaliS069 / MH-Weed16 |
| **Sensor** | Intel RealSense Depth Camera |
| **Image Resolution** | 1920 × 1080 pixels |
| **Total Raw Images** | 6,656 field captures |
| **Annotated Instances** | 59,010 bounding box labels |
| **Annotation Format** | YOLO normalized `(class_id, x_center, y_center, width, height)` |
| **Training Split** | 5,324 images (80%) |
| **Validation Split** | 665 images (10%) |
| **Held-out Test Split** | 667 images (10%) |
| **Detection Backbone** | YOLO26 Nano (YOLO26n) |
| **Input Resolution** | 640 × 640 pixels |

---

## 3. Weed Species Taxonomy (15 Classes)

| Class ID | Species | Botanical Name | Common Name | Regional Name |
|:---:|:---|:---|:---|:---|
| 0 | kena | *Commelina benghalensis* | Benghal Dayflower | Kena / Kanka |
| 1 | lavhala | *Cyperus rotundus* | Purple Nutsedge | Lavhala / Motha |
| 2 | lambs_quarters | *Chenopodium album* | Lamb's Quarters | Bathua |
| 3 | little_mallow | *Malva parviflora* | Little Mallow | Khaparkhuti |
| 4 | moti_dudhi | *Euphorbia hirta* | Asthma-plant | Moti Dudhi |
| 5 | obscure_morning_glory | *Ipomoea obscura* | Obscure Morning Glory | Pili Bhamri |
| 6 | asian_pigeonwings | *Clitoria ternatea* | Butterfly Pea | Gokarna |
| 7 | bilayat | *Parthenium hysterophorus* | Santa Maria Feverfew | Bilayat / Congress Gavat |
| 8 | choti_dudhi | *Euphorbia microphylla* | Small Spurge | Choti Dudhi |
| 9 | digitaria | *Digitaria sanguinalis* | Large Crabgrass | Digitaria |
| 10 | gajar_gavat | *Parthenium hysterophorus* (var.) | Wild Carrot / Bitter Weed | Gajar Gavat |
| 11 | graceful_sandmat | *Euphorbia hypericifolia* | Graceful Sandmat | Dudhali |
| 12 | sicklepod | *Senna obtusifolia* | Sicklepod | Tarota / Takla |
| 13 | harali | *Cynodon dactylon* | Bermuda Grass | Harali / Durva |
| 14 | dwarf_cassia | *Chamaecrista pumila* | Dwarf Cassia | Choti Tarota |

---

## 4. Architectural Ablation Study

Seven custom YOLO26n configurations were designed to balance accuracy vs. edge efficiency. All models compiled to ONNX and benchmarked on CPU (ONNX Runtime) at 640×640 input resolution.

### 4.1 Model Variants Description

| Variant | Key Design Choice |
|:---|:---|
| **baseline_yolo26n** | Standard YOLO26n with full C2PSA multi-head self-attention |
| **ghost_yolo26n** | GhostConv + C3Ghost: cheap linear feature generation |
| **dw_yolo26n** | Depthwise Separable Convolutions across backbone and neck |
| **ghost_dw_hybrid_yolo26n** | Ghost backbone + depthwise PANet neck (hybrid) |
| **no_psa_yolo26n** | C2PSA stripped; replaced with SPPF pooling (fully fused graph) |
| **slim_neck_yolo26n** | Compressed PANet channel capacity for reduced bandwidth |
| **micro_scaled_yolo26n** | Compound-scaled ultra-compact model `scale=[0.33, 0.25, 512]` |

### 4.2 Benchmark Results Table

*(Sorted by FP16 model size — ascending)*

| Model Variant | Layers | Parameters | GFLOPs | FP32 (MB) | FP16 (MB) | FP32 Latency (ms) | FP32 FPS | FP16 Latency (ms) | FP16 FPS |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **micro_scaled_yolo26n** | 209 | 914,985 | 4.58 | 3.72 | **1.90** | **24.14** | **41.4** | 28.63 | 34.9 |
| **ghost_dw_hybrid_yolo26n** | 201 | 1,454,813 | 5.09 | 5.76 | 2.93 | 24.44 | 40.9 | 31.24 | 32.0 |
| **ghost_yolo26n** | 211 | 1,489,173 | 4.94 | 5.90 | 3.00 | 25.21 | 39.7 | **26.85** | **37.2** |
| **dw_yolo26n** | 163 | 1,680,925 | 4.63 | 6.63 | 3.36 | 25.40 | 39.4 | 27.10 | 36.9 |
| **slim_neck_yolo26n** | 197 | 1,956,469 | 5.58 | 7.69 | 3.89 | 25.85 | 38.7 | 28.85 | 34.7 |
| **no_psa_yolo26n** | 163 | 2,343,037 | 6.25 | 9.15 | 4.62 | 27.35 | 36.6 | 29.53 | 33.9 |
| **baseline_yolo26n** | 181 | 2,592,765 | 6.51 | 10.12 | 5.11 | 32.08 | 31.2 | 29.15 | 34.3 |

→ *Figure: `02_ablation_architecture/ablation_architecture_dashboard.png`*
→ *Data: `02_ablation_architecture/ablation_results_table.csv`*

### 4.3 Key Ablation Findings

- **Balanced Edge Model (`ghost_yolo26n`)**: Reduces parameters by **42.6%** (1.49M vs 2.59M) and compresses ONNX to **3.00 MB FP16** while improving CPU throughput from 31.2 → **39.7 FPS**. Best overall trade-off for field hardware.
- **Ultra-Low-Power Model (`micro_scaled_yolo26n`)**: Sub-1M parameters (914K) and **1.90 MB FP16** — the smallest footprint. Achieves lowest FP32 latency at 24.14 ms (41.4 FPS). Ideal for Raspberry Pi or microcontrollers.
- **Attention Stripping (`no_psa_yolo26n`)**: Removing C2PSA eliminates fragmented ONNX subgraph operators (reshape, transpose, matmul), saving **4.73 ms/frame** on CPU without changing convolutional feature hierarchy.
- **GFLOPs vs Latency**: GFLOPs alone do not predict latency — `dw_yolo26n` (4.63 GFLOPs) runs faster than `no_psa_yolo26n` (6.25 GFLOPs), confirming operator fusion and graph structure matter for CPU throughput.

---

## 5. Two-Phase Training Protocol

A transfer learning strategy was adopted to prevent catastrophic forgetting of YOLO26 pre-trained ImageNet features:

| Phase | Layers Frozen | Epochs | Learning Rate | Purpose |
|:---|:---|:---:|:---:|:---|
| **Phase 1 — Warmup** | Backbone layers 0–10 (`freeze=11`) | 1–20 | Default (`lr0`) | Align neck + detection head to MH-Weed16 distribution |
| **Phase 2 — Fine-tuning** | None (`freeze=0`) | 21–50 | `lr0 = 0.0005` | Full network adaptation for leaf venation and fine-grained textures |

- **Optimizer**: AdamW
- **Batch Size**: 8
- **Input Resolution**: 640 × 640

→ *Figure Phase 1: `01_training_curves/phase1_results.png`*
→ *Figure Phase 2: `01_training_curves/phase2_results.png`*
→ *Label Distribution: `01_training_curves/phase1_labels.jpg`*

---

## 6. Detection Performance — Final Model

### 6.1 Confusion Matrices

- **Phase 2 validation (absolute):** `03_confusion_matrices/phase2_confusion_matrix.png`
- **Phase 2 validation (normalized):** `03_confusion_matrices/phase2_confusion_matrix_normalized.png`
- **Held-out test set (absolute):** `03_confusion_matrices/test_split_confusion_matrix.png`
- **Held-out test set (normalized):** `03_confusion_matrices/test_split_confusion_matrix_normalized.png`

### 6.2 Precision-Recall & Detection Curves

All curves evaluated at COCO-standard IoU thresholds (mAP50 / mAP50-95):

| Curve | Phase 2 Validation | Test Split |
|:---|:---|:---|
| Precision-Recall | `phase2_BoxPR_curve.png` | `test_BoxPR_curve.png` |
| F1-Confidence | `phase2_BoxF1_curve.png` | `test_BoxF1_curve.png` |
| Precision-Confidence | `phase2_BoxP_curve.png` | `test_BoxP_curve.png` |
| Recall-Confidence | `phase2_BoxR_curve.png` | `test_BoxR_curve.png` |

→ *All in: `04_pr_and_detection_curves/`*

---

## 7. Per-Class Test Set Metrics (Held-out Split)

Evaluated on 667 held-out test images. Results from `test_evaluation_metrics.csv`:

| Class | Species | Precision | Recall | mAP50 | mAP50-95 |
|:---:|:---|:---:|:---:|:---:|:---:|
| 3 | little_mallow | 0.730 | 0.804 | **0.837** | 0.504 |
| 14 | dwarf_cassia | 0.882 | 0.891 | 0.901 | 0.390 |
| 0 | kena | 0.729 | 0.711 | 0.748 | 0.380 |
| 7 | bilayat | 0.551 | 0.733 | 0.830 | 0.370 |
| 8 | choti_dudhi | 0.785 | 0.732 | 0.800 | 0.336 |
| 10 | gajar_gavat | 0.791 | 0.538 | 0.623 | 0.313 |
| 4 | moti_dudhi | 0.753 | 0.657 | 0.676 | 0.312 |
| 12 | sicklepod | 0.627 | 0.515 | 0.527 | 0.272 |
| 5 | obscure_morning_glory | 0.629 | 0.552 | 0.577 | 0.264 |
| 2 | lambs_quarters | 0.705 | 0.574 | 0.656 | 0.251 |
| 6 | asian_pigeonwings | 0.787 | 0.571 | 0.541 | 0.239 |
| 11 | graceful_sandmat | 0.486 | 0.556 | 0.543 | 0.215 |
| 9 | digitaria | 0.685 | 0.436 | 0.528 | 0.199 |
| 1 | lavhala | 0.610 | 0.419 | 0.495 | 0.190 |
| 13 | harali | 0.404 | 0.525 | 0.413 | 0.089 |

→ *Data file: `06_per_class_metrics/test_evaluation_metrics.csv`*

**Observations:**
- `dwarf_cassia` and `little_mallow` achieve the highest mAP50 (>83%), likely due to distinct morphology and larger annotated bounding boxes.
- `harali` (*Cynodon dactylon* / Bermuda Grass) is the hardest class (mAP50-95 = 0.089), consistent with its fine grass-like appearance closely resembling soil texture.
- Grass-type weeds (`digitaria`, `lavhala`, `harali`) consistently show lower recall, suggesting they are visually similar and require more diverse augmentation.

---

## 8. Qualitative Detection Examples

Side-by-side ground truth vs. model prediction visualizations:

| Batch | Ground Truth | Prediction |
|:---|:---|:---|
| Phase 2 Val — Batch 0 | `phase2_val_batch0_labels.jpg` | `phase2_val_batch0_pred.jpg` |
| Phase 2 Val — Batch 1 | `phase2_val_batch1_labels.jpg` | `phase2_val_batch1_pred.jpg` |
| Phase 2 Val — Batch 2 | `phase2_val_batch2_labels.jpg` | `phase2_val_batch2_pred.jpg` |
| Test Split — Batch 0 | `test_val_batch0_labels.jpg` | `test_val_batch0_pred.jpg` |
| Test Split — Batch 1 | `test_val_batch1_labels.jpg` | `test_val_batch1_pred.jpg` |
| Test Split — Batch 2 | `test_val_batch2_labels.jpg` | `test_val_batch2_pred.jpg` |

→ *All in: `05_qualitative_predictions/`*

---

## 9. Multi-Modal Feature Engineering (29-D Descriptor)

In addition to detection, each weed bounding box is described by a 29-dimensional tabular feature vector:

| Group | Features | Count |
|:---|:---|:---:|
| **Geometry** | width, height, aspect_ratio, area, perimeter, center_dist, boxes_in_frame, area_to_aspect_ratio | 8 |
| **RGB Color** | r_mean, g_mean, b_mean, r_std, g_std, b_std | 6 |
| **HSV Color** | hue_mean, sat_mean, val_mean | 3 |
| **Spectral Indices** | exg_mean, exg_std, exr_mean, ndi_mean, mexg_mean, is_high_exg | 6 |
| **Texture & Focus** | contrast_std, laplacian_var, sobel_energy | 3 |
| **Identity** | image_name, class_id, species | 3 |

Key spectral indices used:
- **ExG** (Excess Green) = 2G − R − B → separates vegetative tissue from soil
- **ExR** (Excess Red) = 1.4R − G → captures soil/residue background
- **NDI** (Normalized Difference Index) = (G−R)/(G+R) → greenness level
- **MExG** (Modified Excess Green) = 1.262G − 0.866R − 0.411B → shadow-tolerant index

Total samples extracted: **59,010 weed instances** → `weed_feature_vectors.csv`

---

## 10. Edge Deployment Summary

| Model | Format | Size | Target Hardware |
|:---|:---|:---:|:---|
| `ghost_yolo26n` | ONNX FP16 | 3.00 MB | NVIDIA Jetson Nano, Intel Core Ultra, field PCs |
| `ghost_yolo26n` | ONNX FP32 | 5.90 MB | Intel Arc, embedded Linux |
| `micro_scaled_yolo26n` | ONNX FP16 | 1.90 MB | Raspberry Pi, MCUs, UAV edge modules |

- **Inference engine**: ONNX Runtime (no PyTorch dependency on edge)
- **Input tensor**: `[1, 3, 640, 640]` float32 / float16
- **Output tensor**: `[1, 19, 8400]` → `[xc, yc, w, h, 15 class scores]`
- **Dynamic batching** enabled in ONNX export

---

## 11. Figure Index (for Paper)

| Fig # | Description | File |
|:---:|:---|:---|
| Fig 1 | Phase 1 Training Curves | `01_training_curves/phase1_results.png` |
| Fig 2 | Phase 2 Training Curves (Final Model) | `01_training_curves/phase2_results.png` |
| Fig 3 | Ground Truth Label Distribution | `01_training_curves/phase1_labels.jpg` |
| Fig 4 | Architectural Ablation Dashboard | `02_ablation_architecture/ablation_architecture_dashboard.png` |
| Fig 5 | Phase 2 Confusion Matrix (Absolute) | `03_confusion_matrices/phase2_confusion_matrix.png` |
| Fig 6 | Phase 2 Confusion Matrix (Normalized) | `03_confusion_matrices/phase2_confusion_matrix_normalized.png` |
| Fig 7 | Test Split Confusion Matrix (Absolute) | `03_confusion_matrices/test_split_confusion_matrix.png` |
| Fig 8 | Test Split Confusion Matrix (Normalized) | `03_confusion_matrices/test_split_confusion_matrix_normalized.png` |
| Fig 9 | Phase 2 Precision-Recall Curve | `04_pr_and_detection_curves/phase2_BoxPR_curve.png` |
| Fig 10 | Phase 2 F1-Confidence Curve | `04_pr_and_detection_curves/phase2_BoxF1_curve.png` |
| Fig 11 | Test Split Precision-Recall Curve | `04_pr_and_detection_curves/test_BoxPR_curve.png` |
| Fig 12 | Test Split F1-Confidence Curve | `04_pr_and_detection_curves/test_BoxF1_curve.png` |
| Fig 13 | Qualitative Val Predictions vs GT (Phase 2) | `05_qualitative_predictions/phase2_val_batch*` |
| Fig 14 | Qualitative Test Predictions vs GT | `05_qualitative_predictions/test_val_batch*` |

---

## 12. Citation

```bibtex
@dataset{mhweed16_dataset,
  author    = {Sayali Sonavane},
  title     = {MH-Weed16: Multi-Class Agricultural Weed Dataset with RealSense Captures},
  year      = {2024},
  publisher = {Kaggle},
  url       = {https://www.kaggle.com/datasets/sayalis069/mh-weed16}
}
```

**Frameworks:**
- Ultralytics YOLO — https://github.com/ultralytics/ultralytics
- SAHI: Slicing Aided Hyper Inference — https://github.com/obss/sahi
- ONNX Runtime — https://github.com/microsoft/onnxruntime
