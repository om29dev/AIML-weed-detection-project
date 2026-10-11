# Contributing Guidelines

Thank you for contributing to the MH-Weed16 agronomic vision and weed detection pipeline. This document outlines our development workflow, coding standards, pull request procedures, and contribution expectations.

---

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Branching and Commit Conventions](#branching-and-commit-conventions)
- [Jupyter Notebook Etiquette](#jupyter-notebook-etiquette)
- [Architectural and Model Contributions](#architectural-and-model-contributions)
- [Feature Engineering and Data Integrity](#feature-engineering-and-data-integrity)
- [Submitting a Pull Request](#submitting-a-pull-request)

---

## Code of Conduct

We are committed to providing a respectful, collaborative, and inclusive environment. Please communicate professionally in all issue discussions, pull request reviews, and commit messages.

---

## How Can I Contribute?

Contributions can take multiple forms:

1. **Bug Reports**: Identifying pipeline bugs, broken links, dependency conflicts, or edge export issues.
2. **Feature Additions**: Implementing new spectral vegetation indices, morphological descriptors, or visualization utilities.
3. **Model Architectures**: Designing lighter or faster YOLO26 backbone or neck variants in `configs/models/`.
4. **Edge Optimization**: Adding deployment runtimes (e.g., OpenVINO, TensorRT, CoreML, TFLite) or INT8 quantization scripts.
5. **Documentation**: Clarifying setup instructions, adding usage examples, or improving docstrings.

---

## Development Setup

### 1. Fork and Clone

```powershell
# Clone your fork
git clone https://github.com/<your-username>/AIML-weed-detection-project.git
cd AIML-weed-detection-project
```

### 2. Environment Preparation

Use Python 3.10 or higher. Configure a virtual environment using PowerShell:

```powershell
# Create virtual environment
python -m venv .venv

# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Upgrade package manager
python -m pip install --upgrade pip

# Install project dependencies
pip install ultralytics torch torchvision opencv-python onnx onnxruntime sahi scikit-learn pandas numpy matplotlib seaborn pyyaml tqdm
```

### 3. Dataset Linking

Do not duplicate the dataset files across drives. Use symbolic directory junctions on Windows:

```powershell
cmd /c mklink /J "images" "MH-Weed16\Crop with Weeds\intel Real Sense Depth_Clicks\intel Real Sense Depth_Clicks"
cmd /c mklink /J "labels" "MH-Weed16\Crop with Weeds\intel Real Sense Depth_Annotations\intel Real Sense Depth_Annotations"
cmd /c mklink /J "annotations" "labels"
```

Verify that dataset junctions are recognized before running code:

```powershell
Get-Item images, labels, annotations | Select-Object Name, Target, LinkType
```

---

## Branching and Commit Conventions

### Branch Naming

Create a feature or fix branch from `master`:

- Features: `feature/<descriptive-name>` (e.g., `feature/int8-quantization`)
- Bug Fixes: `fix/<issue-name>` (e.g., `fix/label-cache-path`)
- Performance: `perf/<optimization-target>` (e.g., `perf/faster-feature-extraction`)
- Documentation: `docs/<topic>` (e.g., `docs/onnx-benchmark-guide`)

### Commit Messages

Write clear, imperative commit messages:

- `Add INT8 quantization script for YOLO26 edge deployment`
- `Fix relative path resolution in dataset yaml generator`
- `Update ablation benchmark table with latest FP16 metrics`
- `Refactor spectral index calculation to vectorize numpy operations`

Avoid vague commit messages such as `fix stuff` or `update code`.

---

## Jupyter Notebook Etiquette

Because Git tracks JSON differences inside `.ipynb` files, untracked cell outputs cause noisy diffs and repository bloat. Follow these steps when editing notebooks such as [MHWeed16_YOLO26_Detection.ipynb](file:///d:/proj/MHWeed16/mhweedaiml/MHWeed16_YOLO26_Detection.ipynb):

1. **Clear Redundant Outputs**: Strip large intermediate outputs, binary base64 plots, or execution traces that are not necessary for documentation.
2. **Deterministic Seeds**: Keep fixed random seeds (`seed=0` or `random_state=42`) across Python, NumPy, and PyTorch cells.
3. **Never Commit Checkpoints**: Keep `.ipynb_checkpoints/` directories excluded via `.gitignore`.
4. **Modular Logic**: If a helper function exceeds 50 lines, consider extracting it into a reusable Python module.

---

## Architectural and Model Contributions

When introducing new model configurations:

1. **YAML Configuration**: Place the model definition file in [configs/models/](file:///d:/proj/MHWeed16/mhweedaiml/configs/models) using standard Ultralytics architecture schema (`nc: 15`, `scales`, `backbone`, `head`).
2. **Naming Standard**: Name files descriptively using lowercase alphanumeric characters and underscores (e.g., `configs/models/repvgg_yolo26n.yaml`).
3. **ONNX Export Validation**: Verify that the new architecture exports cleanly to ONNX without unsupported custom operators:
   ```python
   from ultralytics import YOLO
   model = YOLO("configs/models/your_model.yaml")
   model.export(format="onnx", imgsz=640, dynamic=True)
   ```
4. **Benchmarking**: Profile the model using `onnxruntime` on CPU for at least 100 iterations. Record:
   - Layer count
   - Parameter count
   - GFLOPs
   - FP32 model size (MB)
   - FP16 model size (MB)
   - Average latency (ms) and throughput (FPS)
5. **Update Benchmark Log**: Append benchmark figures to [runs/ablation_results.csv](file:///d:/proj/MHWeed16/mhweedaiml/runs/ablation_results.csv).

---

## Feature Engineering and Data Integrity

When updating [weed_feature_vectors.csv](file:///d:/proj/MHWeed16/mhweedaiml/weed_feature_vectors.csv) or feature extraction functions:

1. **Preserve Coordinate Schema**: Bounding box coordinates must strictly conform to normalized YOLO format:
   $$\{x_c, y_c, w, h\} \in [0.0, 1.0]$$
2. **Species Index Alignment**: Maintain strict alignment with class mappings in [mhweed16.yaml](file:///d:/proj/MHWeed16/mhweedaiml/mhweed16.yaml) (classes 0 through 14).
3. **Zero Division Protections**: Always add epsilon constants (e.g., `+ 1e-6`) to denominators when computing ratios (aspect ratio, NDI, normalized indices).
4. **NaN Handling**: Ensure any introduced features cleanly handle solid color or clipped edge patches without producing `NaN` or `Inf` values.

---

## Submitting a Pull Request

1. Ensure the branch is rebased onto the latest `master`.
2. Verify all modified notebooks and scripts run without errors.
3. Ensure `.gitignore` is respected (do not commit raw images, large video captures, or temporary `.cache` files).
4. Open a Pull Request on GitHub with:
   - A summary of what changes were made and why.
   - Any relevant performance metrics (FPS, mAP, model size delta).
   - Before and after screenshots or visual outputs where applicable.
5. Address reviewer feedback promptly.

