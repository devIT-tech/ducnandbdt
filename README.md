# AgriDetectVL: Interactive and Resource-Efficient Vision–Language Model for Counterfeit Agricultural Detection

This repository contains the official implementation of our IEEE Access paper:
"AgriDetectVL: Emphasizes the Agriculture-Focused Application Combined with Visual-Language Integration"
by Dat Tran, Anh Duc Nguyen (Thuyloi University), and Hoai Nam Vu (PTIT).

## Overview

AgriDetectVL is an interactive and resource-efficient Vision–Language Model (VLM) designed for counterfeit-product detection in agriculture.
It combines visual features, textual prototypes, temporal context, and human feedback to achieve accurate and efficient detection under real-world constraints (drones, edge devices, field robots).

Key challenges addressed:

* Temporal context: tracking agricultural products over time.
* User feedback: integrating operator corrections into predictions.
* Resource efficiency: enabling real-time deployment on edge devices.

## Key Contributions

* Sequence Prompt Transformer (SPT): aggregates temporal visual context and user prompts.
* Text Prototypes: class names and domain phrases encoded as anchors in embedding space.
* Low-latency inference: single-pass cosine scoring with temperature scaling.
* Human-in-the-loop: integrates corrections to refine future predictions.

## Architecture

* Vision Encoder (ViT-B/16): efficient backbone for visual features.
* Large Language Model (Qwen2): interprets textual inputs.
* Top-K Prompt Selector (TPS): retrieves most relevant visual prompts.
* Sequence Prompt Transformer (SPT): models temporal dependencies.
* Fusion Head: integrates refined visual and semantic vectors.

## Results

Datasets: Food-101, TLU-Fruit, and TLU-States.

| Model        | Food-101 F1 | TLU-Fruit F1 | TLU-States F1 |
| ------------ | ----------- | ------------ | ------------- |
| InternVL2    | 85.1        | 82.5         | 80.2          |
| LLaVA-OV     | 86.3        | 84.1         | 81.9          |
| AgriDetectVL | 88.2        | 86.5         | 84.3          |

* Latency: 55 ms (FP16), 32 ms (INT8)
* Model Size: 310 MB (FP16), 155 MB (INT8)
* Power: 8.1 W (FP16), 6.5 W (INT8)

## Installation

Requirements:

* Python 3.8+
* PyTorch 2.3+
* Hugging Face Transformers
* CUDA-enabled GPU (e.g., NVIDIA 3090)

```bash
# Clone repo
git clone https://github.com/your-repo/AgriDetectVL.git
cd AgriDetectVL

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

1. Dataset Preparation

* Food-101 (auto-download via torchvision)
* TLU-Fruit, TLU-States: [link to dataset]

2. Run Training

```bash
python main.py --config configs/tlu_fruit.yaml
python main.py --config configs/tlu_states.yaml
```

3. Run Inference

```bash
python eval.py --checkpoint checkpoints/agri_fruit.pth --dataset TLU-Fruit
```

## Evaluation

Metrics supported:

* F1-Score
* Accuracy
* AUC
* MCC
* Latency / Power (edge deployment profiling)

## Datasets

* TLU-Fruit: fine-grained fruit varieties (genuine vs counterfeit).
* TLU-States: ripeness and state classification.
* Food-101: standard benchmark for food recognition.

## Features

* Training and evaluation pipelines with configs.
* Edge-device ready (FP16 and INT8 quantization).
* Human-in-the-loop feedback integration.
* Supports zero-shot and few-shot extension.

## Citation

If you use this work, please cite:

```
@ARTICLE{AgriDetectVL2024,
  author={Dat Tran and Anh Duc Nguyen and Hoai Nam Vu},
  journal={IEEE Access},
  title={AgriDetectVL: Emphasizes the Agriculture-Focused Application Combined with Visual-Language Integration},
  year={2024},
  doi={10.1109/ACCESS.2024.0429000}
}
```

## Authors

* Dat Tran – Thuyloi University – [dat.trananh@tlu.edu.vn](mailto:dat.trananh@tlu.edu.vn)
* Anh Duc Nguyen – Thuyloi University
* Hoai Nam Vu – PTIT – [namvh@ptit.edu.vn](mailto:namvh@ptit.edu.vn)

## Acknowledgments

We thank the Faculty of Information Technology at Thuyloi University and YIRLoDT Lab (PTIT) for supporting this research.

Keywords: Vision–Language Models, Agriculture, Counterfeit Detection, Deep Learning, Computer Vision, Multimodal Learning
