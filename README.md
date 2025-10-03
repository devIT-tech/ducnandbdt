# AgriDetectVL: A Vision-Language Model for Advanced Agricultural Counterfeit Detection

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.3.0%2B-orange)

**A cutting-edge, resource-efficient vision-language model tailored for detecting counterfeit agricultural products**

This repository provides the official implementation of the research paper: "AgriDetectVL: Emphasizing Agriculture-Focused Applications through Visual-Language Integration" by Dat Tran, Anh Duc Nguyen, and Hoai Nam Vu, published in *IEEE Access* (Volume 11, 2023).

## Introduction

AgriDetectVL is a pioneering vision-language model (VLM) designed to address the critical challenge of counterfeit product detection in agriculture. By integrating advanced visual processing with natural language understanding, the model offers a robust, interactive, and computationally efficient solution for agricultural monitoring. Key features include:

- **Temporal Context Awareness**: Seamlessly processes time-series imagery and incorporates operator feedback for enhanced decision-making.
- **Resource Efficiency**: Optimized for low-compute environments, enabling deployment on edge devices such as drones and mobile phones.
- **Human-in-the-Loop Interaction**: Leverages textual prompts to refine predictions dynamically, supporting real-time adaptability.
- **Scalable and Versatile**: Supports zero/few-shot learning and fine-grained classification, making it adaptable to diverse agricultural scenarios.

The model combines prototype-based classification with language-guided reasoning, achieving low-latency inference and high accuracy in detecting counterfeit agricultural products. This implementation is optimized for practical deployment, balancing performance with computational constraints.
## Architecture
AgriDetectVL builds on an efficient VLM backbone with the following core components:

- **Vision Encoder (VE)**: Based on Vision Transformer (ViT-B) for processing primary images and prompt images.
- **Large Language Model (LLM)**: Qwen2 for interpreting textual prompts.
- **Top-K Prompt Selector (TPS)**: Selects the most similar historical prompts from a Prompt Pool (PP) based on similarity.
- **Sequence Prompt Transformer (SPT)**: Aggregates selected prompts into a refined vector, modeling sequential dependencies.
- **Refined Vision Feature (RVF)**: Fuses SPT output with LLM semantic vectors for downstream tasks.
- **Prototype-Based Classifier**: Uses text prototypes for class names, with cosine scoring for decisions.

The architecture supports multimodal fusion and is optimized for efficiency using techniques like mixed-precision training (FP8) and FlashAttention-2.

![AgriDetectVL Architecture](path/to/architecture-diagram.png)  <!-- Replace with actual diagram if available, e.g., from Figure 1 or 2 in the paper -->

## Results
Evaluated on agricultural benchmarks: Food-101, TLU-Fruit (fine-grained varieties), and TLU-States (state/ripeness).

### Comparison Across Models (F1-Score, AUC, MCC)
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

### Efficiency on Food-101 Dataset
| Model                  | F1-Score (%) | Latency (ms) | Power (W) | Size (MB) |
|------------------------|--------------|--------------|-----------|-----------|
| LLaVA-OV (FP16)       | 86.3        | 210         | 18.5     | 14500    |
| InternVL2 (FP16)       | 85.1        | 195         | 17.2     | 12800    |
| AgriDetectVL (FP16)    | 88.2        | 55          | 8.1      | 310      |
| AgriDetectVL (INT8)    | 87.5        | 32          | 6.5      | 155      |

AgriDetectVL outperforms baselines while meeting edge-device constraints.

## Installation
### Requirements
- Python 3.8+
- PyTorch 2.3.0+
- Hugging Face Transformers
- Additional libraries: NumPy, SciPy, Matplotlib (for evaluation)

```bash
# Clone the repository
git clone https://github.com/yourusername/AgriDetectVL.git
cd AgriDetectVL

# Install dependencies
pip install -r requirements.txt
```

## Quick Start
### 1. Dataset Preparation
- **Food-101**: Download from [official source] and extract to `datasets/Food-101/`.
- **TLU-Fruit and TLU-States**: Download from [link provided in paper] and extract to `datasets/TLU-Fruit/` and `datasets/TLU-States/`.

### 2. Basic Usage
```bash
# Run AgriDetectVL on TLU-Fruit dataset
python main.py --config configs/tlu_fruit.yaml

# Run on Food-101
python main.py --config configs/food101.yaml

# Run on TLU-States
python main.py --config configs/tlu_states.yaml
```

### 3. Custom Configuration
Modify configuration files in `configs/` to adjust:
- Backbone (e.g., ViT-B/16)
- Prompt templates and class names
- Training hyperparameters (e.g., learning rate: 5e-4, batch size: 16)
- Inference mode (FP16 or INT8 quantization)
- Number of prompts (Top-K)

## Scalability Analysis
```bash
# Test with different backbones
python main.py --config configs/tlu_fruit.yaml --backbone ViT-B/16

# Test with quantization
python main.py --config configs/tlu_fruit.yaml --quantization INT8
```

## Evaluation
The framework supports evaluation metrics:
- F1-Score: Balance between precision and recall
- AUC: Class separability
- MCC: Robustness to class imbalance
- Accuracy: Overall performance
- Latency and Power: For efficiency profiling

## Datasets
- **TLU-Fruit**: Fine-grained fruit varieties dataset for counterfeit detection, with expert-verified annotations.
- **TLU-States**: Fruit state/ripeness recognition, simulating real-world agricultural scenarios with varying lighting and backgrounds.
- **Food-101**: General food recognition benchmark used for pretraining and comparison.

These datasets focus on challenging agricultural tasks with noisy, in-the-wild data.

## Key Features
- Resource-Efficient: Low latency (32ms in INT8) and memory (155MB) for edge deployment.
- Interactive: Supports human-in-the-loop with textual feedback.
- Sequence-Aware: Integrates temporal context via SPT.
- Zero/Few-Shot Capable: Add new categories via textual descriptions without retraining.
- Robust: Consistent gains on fine-grained tasks, reducing look-alike confusions.
- Scalable: Works on commodity GPUs (e.g., NVIDIA 3090) and embedded devices.

## Citation
If you find this work useful, please cite:
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

## Authors
- Dat Tran - Thuyloi University, Hanoi, Vietnam  
  Email: dat.trananh@tlu.edu.vn
- Anh Duc Nguyen - Thuyloi University, Hanoi, Vietnam  
  Email: (not provided in paper)
- Hoai Nam Vu - Young Innovation Research Laboratory (YIRLoDT), Posts and Telecommunications Institute of Technology, Hanoi, Vietnam  
  Email: namvh@ptit.edu.vn

## Acknowledgments
We acknowledge the support from Thuyloi University and the Young Innovation Research Laboratory (YIRLoDT). Evaluations were conducted on NVIDIA 3090 GPUs.

## Keywords
Counterfeit agricultural detection, Computer vision, Image processing, Visual-Language model, Sequence Prompt Transformer, Human-in-the-loop, Edge deployment, Fine-grained recognition
