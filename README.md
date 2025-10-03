# AgriDetectVL: Agriculture-Focused Vision-Language Model with Interactive Capabilities

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.3.0%2B-orange)

An efficient and interactive vision-language framework for monitoring counterfeit agricultural products

This repository hosts the official code implementation for the research paper titled "AgriDetectVL: Emphasizes the agriculture-focused application combined with Visual-Language integration," authored by Dat Tran, Anh Duc Nguyen, and Hoai Nam Vu, and published in IEEE Access (Volume 11, 2023).

## Project Summary
AgriDetectVL is a specialized vision-language model (VLM) tailored for detecting and monitoring counterfeit agricultural items. It tackles essential issues in agricultural surveillance, such as:

- Incorporating Temporal Data: Processing sequences of time-series images alongside user-provided feedback.
- Edge Device Compatibility: Operating within limited computational resources, suitable for devices like mobile units or drones.
- User Interaction Support: Allowing real-time refinements via textual inputs to enhance accuracy progressively.

The framework combines prototype-driven classification with linguistic cues, supporting fast inference and adaptable zero/few-shot learning.

## System Design
AgriDetectVL extends an optimized VLM foundation with key modules:

- **Visual Processing Unit (VE)**: Utilizes Vision Transformer (ViT-B) to handle main images and reference prompts.
- **Language Interpreter (LLM)**: Employs Qwen2 to process text-based instructions.
- **Selective Prompt Mechanism (TPS)**: Identifies top-matching historical prompts from a stored Prompt Pool (PP) using similarity metrics.
- **Temporal Prompt Processor (SPT)**: Combines chosen prompts into a cohesive vector, accounting for sequence relationships.
- **Feature Fusion Layer (RVF)**: Merges SPT results with LLM outputs for task-specific applications.
- **Prototype Classifier**: Employs text-derived prototypes for categories, using cosine-based scoring for outputs.

Optimized for performance with methods like FP8 mixed-precision and FlashAttention-2.

![System Diagram](path/to/system-diagram.png)  <!-- Placeholder for Figure 1 or 2 from the paper -->

## Performance Highlights
Tested on key agricultural datasets: Food-101, TLU-Fruit (variety-specific), and TLU-States (ripeness/stage assessment).

### Model Metrics Comparison (F1-Score, AUC, MCC)
| Model                  | Dataset    | F1-Score      | AUC           | MCC           |
|------------------------|------------|---------------|---------------|---------------|
| 3*InternVL2 (Fine-tuned) | Food-101  | 85.1 ± 0.6   | 0.912 ± 0.05 | 0.795 ± 0.08 |
|                        | TLU-Fruit | 82.5 ± 0.7   | 0.881 ± 0.06 | 0.751 ± 0.09 |
|                        | TLU-States| 80.2 ± 0.9   | 0.856 ± 0.08 | 0.713 ± 0.11 |
| 3*LLaVA-OV (Fine-tuned) | Food-101  | 86.3 ± 0.5   | 0.925 ± 0.04 | 0.810 ± 0.06 |
|                        | TLU-Fruit | 84.1 ± 0.6   | 0.903 ± 0.05 | 0.776 ± 0.07 |
|                        | TLU-States| 81.9 ± 0.8   | 0.874 ± 0.07 | 0.742 ± 0.10 |
| 3*AgriDetectVL (Ours) | Food-101  | 88.2 ± 0.3   | 0.941 ± 0.02 | 0.835 ± 0.04 |
|                        | TLU-Fruit | 86.5 ± 0.4   | 0.922 ± 0.03 | 0.803 ± 0.05 |
|                        | TLU-States| 84.3 ± 0.5   | 0.898 ± 0.04 | 0.781 ± 0.06 |

### Resource Usage on Food-101
| Model                  | F1-Score (%) | Latency (ms) | Power (W) | Size (MB) |
|------------------------|--------------|--------------|-----------|-----------|
| LLaVA-OV (FP16)       | 86.3        | 210         | 18.5     | 14500    |
| InternVL2 (FP16)       | 85.1        | 195         | 17.2     | 12800    |
| AgriDetectVL (FP16)    | 88.2        | 55          | 8.1      | 310      |
| AgriDetectVL (INT8)    | 87.5        | 32          | 6.5      | 155      |

AgriDetectVL excels over competitors while adhering to resource limits for field use.

## Setup Guide
### Prerequisites
- Python 3.8 or higher
- PyTorch 2.3.0 or later
- Transformers from Hugging Face
- Supporting packages: NumPy, SciPy, Matplotlib (for analysis)

```bash
# Clone repository
git clone https://github.com/yourusername/AgriDetectVL.git
cd AgriDetectVL

# Install required packages
pip install -r requirements.txt
```

## Getting Started
### 1. Preparing Data
- **Food-101**: Obtain from the standard source and place in `datasets/Food-101/`.
- **TLU-Fruit and TLU-States**: Acquire from the paper's referenced links and organize in `datasets/TLU-Fruit/` and `datasets/TLU-States/`.

### 2. Running Examples
```bash
# Execute on TLU-Fruit
python main.py --config configs/tlu_fruit.yaml

# Execute on Food-101
python main.py --config configs/food101.yaml

# Execute on TLU-States
python main.py --config configs/tlu_states.yaml
```

### 3. Configuration Adjustments
Edit files in `configs/` for customizations:
- Model backbone (e.g., ViT-B/16)
- Class labels and prompt sets
- Optimizer settings (learning rate: 5e-4, batch: 16)
- Runtime mode (FP16/INT8)
- Prompt count (Top-K value)

## Performance Scaling
```bash
# Evaluate with alternate backbones
python main.py --config configs/tlu_fruit.yaml --backbone ViT-B/16

# Apply quantization
python main.py --config configs/tlu_fruit.yaml --quantization INT8
```

## Assessment Metrics
Supports key evaluations:
- F1-Score: Precision-recall harmony
- AUC: Separation quality
- MCC: Imbalance handling
- Accuracy: Total correctness
- Efficiency: Time and energy metrics

## Data Resources
- **TLU-Fruit**: Dataset for detailed fruit type identification in counterfeit scenarios, with professional labels.
- **TLU-States**: Focuses on assessing fruit conditions, mimicking varied field environments.
- **Food-101**: Benchmark for initial training and cross-comparisons.

Designed for tough, real-world agricultural data with environmental variations.

## Core Advantages
- Optimized for Resources: Achieves 32ms latency (INT8) and 155MB footprint for on-device operation.
- Feedback-Enabled: Integrates user text for ongoing improvements.
- Temporal Integration: Uses SPT for context over time.
- Flexible Adaptation: Introduce categories via text without full retraining.
- Reliable: Strong results in detailed tasks, minimizing similar-item errors.
- Expandable: Compatible with standard GPUs (NVIDIA 3090) and portable hardware.

## Reference Citation
Please cite this work as:
```
@article{Tran2023AgriDetectVL,
  author = {Tran, Dat and Nguyen, Anh Duc and Vu, Hoai Nam},
  journal = {IEEE Access},
  title = {AgriDetectVL: Emphasizes the agriculture-focused application combined with Visual-Language integration},
  volume = {11},
  year = {2023},
  doi = {10.1109/ACCESS.2024.0429000}
}
```

## Research Team
- Dat Tran - Thuyloi University, Hanoi, Vietnam  
  Email: dat.trananh@tlu.edu.vn
- Anh Duc Nguyen - Thuyloi University, Hanoi, Vietnam  
- Hoai Nam Vu - Young Innovation Research Laboratory (YIRLoDT), Posts and Telecommunications Institute of Technology, Hanoi, Vietnam  
  Email: namvh@ptit.edu.vn

## Credits
Supported by Thuyloi University and YIRLoDT. Testing performed on NVIDIA 3090 systems.

## Search Terms
Agricultural counterfeit monitoring, Vision systems, Image analysis, VLM framework, Temporal prompt handling, Interactive feedback, Field-ready deployment, Detailed classification
