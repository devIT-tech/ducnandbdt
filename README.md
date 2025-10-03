# AgriDetectVL: Agriculture-Focused Vision–Language Model

## Overview

AgriDetectVL is an interactive and resource‑efficient vision–language model designed for counterfeit product detection and monitoring in agriculture. It integrates temporal visual context with human feedback and uses text prototypes for classification.

## Key Features

* **Sequence Prompt Transformer (SPT):** Aggregates historical visual prompts for temporal awareness.
* **Top‑K Prompt Selector (TPS):** Retrieves the most relevant visual prompts from a prompt pool.
* **Prototype‑Based Classification:** Encodes class names and domain phrases as text prototypes.
* **Human‑in‑the‑Loop:** Allows operator feedback through textual prompts to refine decisions.
* **Resource Efficiency:** Supports low‑latency, low‑memory inference for edge devices.

## Architecture

* Efficient visual encoder (ViT‑B) with lightweight adapters.
* Frozen text encoder (e.g., Qwen2) to generate semantic vectors.
* Shared L2‑normalized embedding space for images and text.
* Fusion of SPT output and LLM semantic vectors through the RVF block.

## Training & Inference

* Loss functions: prototype cross‑entropy, vision–language alignment (InfoNCE), and prototype regularization.
* Single‑pass inference using temperature‑scaled cosine similarity.
* Zero/few‑shot extension by adding textual labels without retraining.

## Datasets

* **Food‑101:** General food classification.
* **TLU‑Fruit:** Fine‑grained agricultural varieties.
* **TLU‑States:** Ripeness and product state recognition.

## Experimental Results

AgriDetectVL outperforms baseline models (LLaVA‑OV, InternVL2) across metrics:

* Higher F1‑Score, AUC, and MCC.
* Lower variance and increased robustness.
* Superior latency, power consumption, and model size (especially in INT8 mode).

### Efficiency (Example on Food‑101)

| Model             | F1‑Score | Latency |  Power |    Size |
| ----------------- | -------: | ------: | -----: | ------: |
| LLaVA‑OV (FP16)   |    86.3% |  210 ms | 18.5 W | 14.5 GB |
| InternVL2 (FP16)  |    85.1% |  195 ms | 17.2 W | 12.8 GB |
| AgriDetectVL FP16 |    88.2% |   55 ms |  8.1 W |  310 MB |
| AgriDetectVL INT8 |    87.5% |   32 ms |  6.5 W |  155 MB |

## Ablation Insights

* Vision encoder alone: baseline performance.
* * Text prompts improves accuracy.
* * TPS visual prompts yields significant gains.
* * SPT provides the highest overall performance.

## Limitations

* Relies on historical visual data for best performance.
* Resource demands may still challenge ultra‑low‑power devices.
* Generalization outside agriculture requires further work.

## Future Directions

* Few‑shot learning for reduced annotation.
* Pruning and quantization for improved efficiency.
* Additional modalities (e.g., spectral data).
* Enhanced multimodal dialogue and explainability.

## Citation

If using AgriDetectVL in academic work, please cite the corresponding IEEE Access article:

```
Dat Tran, Anh Duc Nguyen, Hoai Nam Vu. "AgriDetectVL: Emphasizes the agriculture-focused application combined with Visual-Language integration," IEEE Access, 2024.
```
