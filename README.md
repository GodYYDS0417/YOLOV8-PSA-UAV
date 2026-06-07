# YOLOv8-PSA: Partial Self-Attention Enhanced YOLOv8 for UAV Detection

<div align="center">
  <img src="assets/YOLOv8-PSA network.png" width="100%" alt="YOLOv8-PSA Architecture">
</div>

---

## 🌟 Abstract

This research proposes a Partial Self-Attention (PSA) mechanism-enhanced YOLOv8 model for UAV detection in complex urban environments. The proposed YOLOv8-PSA achieves state-of-the-art performance with an mAP50 of **0.979** and a real-time inference speed of **666.7 FPS**, demonstrating superior accuracy-efficiency trade-off compared to baseline YOLOv8 and other SOTA models.

---

## 📋 Table of Contents

- [🌟 Abstract](#-abstract)
- [📋 Table of Contents](#-table-of-contents)
- [🚀 Key Contributions](#-key-contributions)
- [🏗️ Model Architecture](#️-model-architecture)
  - [Partial Self-Attention (PSA) Module](#partial-self-attention-psa-module)
  - [C2fCIB Module](#c2fcib-module)
- [📊 Experimental Results](#-experimental-results)
  - [Comparison with SOTA Models](#comparison-with-sota-models)
  - [Ablation Study](#ablation-study)
  - [Cross-Dataset Generalization](#cross-dataset-generalization)
- [💾 Installation](#-installation)
- [🚀 Usage](#-usage)
  - [Training](#training)
  - [Inference](#inference)
- [📁 Project Structure](#-project-structure)
- [📝 Citation](#-citation)
- [📄 License](#-license)

---

## 🚀 Key Contributions

1. **Partial Self-Attention (PSA) Mechanism**: Integrates PSA into YOLOv8 for efficient global context modeling with reduced computational overhead through channel partitioning.

2. **Lightweight Multi-Scale Feature Fusion (C2fCIB)**: A novel module designed to synergize with PSA for handling dynamic scale variations in cluttered urban environments.

3. **Superior Speed-Accuracy Trade-off**: Maintains real-time inference capability (FPS ≥ 60) while improving mAP50 by **2.6%** over the baseline YOLOv8.

---

## 🏗️ Model Architecture

### Partial Self-Attention (PSA) Module

<div align="center">
  <img src="assets/detailed diagram of PSA.png" width="80%" alt="PSA Module">
</div>

The PSA module leverages channel partitioning and multi-headed self-attention to effectively capture global dependencies in complex urban backgrounds without incurring severe latency penalties typically associated with standard self-attention.

### C2fCIB Module

The C2fCIB (CSP Bottleneck with Conditional Identity Block) module enhances cross-scale feature fusion while optimizing network efficiency, reducing both parameters and FLOPs compared to the baseline.

---

## 📊 Experimental Results

### Comparison with SOTA Models

| Method | Precision (P) | Recall (R) | mAP50 |
|--------|---------------|------------|-------|
| YOLOv8 (Baseline) | 0.949 | 0.900 | 0.939 |
| YOLOv9 | 0.984 | 0.917 | 0.956 |
| YOLOv10 | 0.971 | 0.917 | 0.956 |
| YOLOv11 | 0.977 | 0.918 | 0.950 |
| YOLOv12 | 0.986 | 0.910 | 0.952 |
| RT-DETR | 0.958 | 0.931 | 0.950 |
| **YOLOv8-PSA (Ours)** | **0.984** | **0.961** | **0.979** |

### Comparison of Attention Mechanisms

| Method | P | R | mAP50 | Latency (ms) | FPS |
|--------|---|---|-------|--------------|-----|
| YOLOv8 | 0.949 | 0.900 | 0.939 | 2.0 | 500.0 |
| YOLOv8 + SE | 0.968 | 0.905 | 0.942 | 0.8 | 1250.0 |
| YOLOv8 + CBAM | 0.977 | 0.897 | 0.944 | 0.9 | 1111.1 |
| YOLOv8 + PSA | 0.966 | 0.910 | 0.949 | 1.4 | 714.3 |

### Ablation Study

| Model | CIoU | PSA | C2fCIB | P | R | mAP50 | Params (M) | FLOPs (G) |
|-------|------|-----|--------|---|---|-------|------------|-----------|
| 1 (Baseline) | — | — | — | 0.949 | 0.900 | 0.939 | 3.01 | 8.1 |
| 2 | ✓ | — | — | 0.973 | 0.908 | 0.944 | 3.01 | 8.1 |
| 3 | — | — | ✓ | 0.952 | 0.904 | 0.945 | 2.85 | 7.8 |
| 4 | ✓ | — | ✓ | 0.973 | 0.908 | 0.944 | 2.79 | 7.9 |
| 5 | — | ✓ | — | 0.971 | 0.897 | 0.941 | 3.29 | 8.4 |
| 6 | ✓ | ✓ | — | 0.969 | 0.913 | 0.948 | 3.29 | 8.4 |
| 7 | — | ✓ | ✓ | 0.966 | 0.912 | 0.950 | 3.07 | 8.2 |
| 8 (Complete) | ✓ | ✓ | ✓ | **0.978** | **0.925** | **0.965** | **3.07** | **8.2** |

### Cross-Dataset Generalization

| Model | Precision | Recall | mAP50 |
|-------|-----------|--------|-------|
| YOLOv8 (Baseline) | 0.772 | 0.549 | 0.689 |
| YOLOv8-PSA (Ours) | **0.867** | **0.623** | **0.713** |

### Performance Across Scenarios

<div align="center">
  <img src="assets/Detection Results .png" width="80%" alt="Detection Results">
</div>

| Scenario | Model | P | R |
|----------|-------|---|---|
| Forest | YOLOv8_PSA | 0.875 | 0.737 |
| Forest | Complete Model | 0.893 | 0.760 |
| Urban | YOLOv8+PSA | 0.944 | 0.813 |
| Urban | Complete Model | 0.949 | 0.893 |

---

## 💾 Installation

```bash
# Clone the repository
git clone https://github.com/GodYDS0417/YOLOv8-PSA-UAV.git
cd YOLOv8-PSA-UAV

# Install dependencies
pip install -e .
```

---

## 🚀 Usage

### Training

```python
from ultralytics import YOLO

# Load the YOLOv8-PSA model
model = YOLO("yolov8n.yaml")

# Train the model
results = model.train(
    data="path/to/your/dataset.yaml",
    epochs=100,
    imgsz=640,
    device="0",
    batch=16,
)
```

### Inference

```python
from ultralytics import YOLO

# Load the trained model
model = YOLO("runs/detect/train/weights/best.pt")

# Perform inference
results = model("path/to/image.jpg")

# Show results
results[0].show()
```

### CLI Usage

```bash
# Training
yolo train data=dataset.yaml model=yolov8n.yaml epochs=100

# Inference
yolo predict model=best.pt source=input.jpg

# Validation
yolo val model=best.pt data=dataset.yaml
```

---

## 📁 Project Structure

```
YOLOv8-PSA-UAV/
├── ultralytics/                    # Main source code
│   ├── nn/
│   │   └── modules/
│   │       ├── block.py            # Contains PSA, CIB, C2fCIB modules
│   │       └── ...
│   ├── cfg/
│   │   └── models/
│   │       └── v8/                 # YOLOv8 model configurations
│   ├── engine/                     # Training/Inference engines
│   ├── models/                     # Model definitions
│   └── ...
├── assets/                         # Documentation images
│   ├── YOLOv8-PSA network.png
│   ├── detailed diagram of PSA.png
│   ├── Detection Results .png
│   ├── FPN_UAV.png
│   └── loss curve.png
├── examples/                       # Usage examples
├── docs/                           # Documentation
└── README.md
```

---

## 📝 Citation

If you find this work useful in your research, please consider citing:

```bibtex
@article{YOLOv8-PSA,
  title={YOLOv8-PSA: Partial Self-Attention Enhanced YOLOv8 for UAV Detection in Complex Urban Environments},
  author={Your Name},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2024}
}
```

---

## 📄 License

This project is licensed under the AGPL-3.0 License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">
  <img src="assets/loss curve.png" width="60%" alt="Training Loss Curve">
</div>

---

*Built with ❤️ for UAV detection research*
