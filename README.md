# AgriDetectVL: Emphasizes the Agriculture-Focused Application Combined with Visual-Language Integration

This repository contains resources related to the IEEE Access paper on AgriDetectVL, a vision-language model specialized for counterfeit-product detection and monitoring in agriculture.

## Abstract

Counterfeit-product monitoring in agriculture demands models that exploit temporal context, accept operator feedback, and run under tight compute budgets. We introduce AgriDetectVL, an interactive, resource-efficient vision–language model that fuses time-series imagery with human inputs. AgriDetectVL couples an efficient visual backbone with a lightweight Sequence Prompt Transformer that summarizes recent observations and feedback into compact prompts. Class names and domain phrases are encoded as text prototypes, and images are mapped into a shared, L2-normalized space; decisions are made by temperature-scaled cosine scoring, enabling single-pass, low-latency inference and straightforward zero/few-shot extension. Evaluated on TLU-Fruit (fine-grained varieties) and TLU-States (state/ripeness), AgriDetectVL consistently surpasses strong CNN, transformer, and VLM baselines across F1, accuracy, AUC, and MCC, while meeting edge-device constraints. Ablations confirm that sequence-aware prompting and prototype guidance are the primary sources of gain. In longitudinal tests, human-in-the-loop operation reduces manual corrections over time, indicating practical readiness for field deployment.

## Index Terms

Counterfeit agricultural detection, Computer vision, Image processing, Visual-Language model.

## Authors

- DAT TRAN (Member, IEEE), Thuyloi University, Hanoi, Vietnam (e-mail: dat.trananh@tlu.edu.vn)
- ANH DUC NGUYEN, Thuyloi University, Hanoi, Vietnam
- Hoai Nam Vu (Member, IEEE), Young Innovation Research Laboratory (YIRLoDT), Posts and Telecommunications Institute of Technology, Hanoi, Vietnam (e-mail: namvh@ptit.edu.vn)

Corresponding author: Dat Tran (e-mail: dat.trananh@tlu.edu.vn).

## Introduction

Accurate and timely monitoring of counterfeit agricultural products is crucial for sustainable agriculture, food safety, and market transparency, yet it requires tools that can efficiently analyze complex, noisy visual data acquired in the wild. Vision–Language Models (VLMs) provide an attractive foundation because they jointly process images and text, enabling richer semantics and more natural human–machine interaction. In principle, such models can link appearances to names, descriptions, and usage instructions, and they can incorporate user feedback directly through language. However, deploying VLMs in agriculture remains challenging. Most existing VLMs are architected for isolated images and short prompts; they do not natively encode temporally evolving context, such as tracking a suspicious batch of produce as it moves across locations and weeks, or accumulating operator feedback over repeated inspections. In addition, their computational and memory demands during both training and inference are often incompatible with edge scenarios—e.g., drones, mobile phones, or field robots—where energy, latency, and bandwidth are tightly constrained.

Two recent lines of research point to a way forward. First, the efficiency literature argues that VLMs should be optimized end-to-end for resource budgets rather than scaled indiscriminately. Methods that “scale then compress” can maintain competitive accuracy while substantially reducing training cost and inference latency; notably, careful architectural choices allow handling high-resolution images and longer sequences without incurring prohibitive overheads. Second, interactive vision methods are beginning to exploit sequences of signals—images, clicks, prompts, and prior masks—instead of treating each image independently. For example, click-based segmentation with iterative prompting demonstrates that user interaction and temporal context can systematically improve quality. While there are emerging principles for efficient VLM design and prompt tuning, prior work has not integrated these ideas into a time-aware, domain-tailored VLM that learns from interaction in agriculture.

We introduce AgriDetectVL, a VLM specialized for counterfeit-product detection and related monitoring tasks in agriculture. The design goal is twofold: (1) computational efficiency that meets the latency and memory budgets of field devices, and (2) sequential, interaction-aware learning that leverages temporal context and user feedback to improve accuracy over time.

## Key Contributions

- (i) We formulate the problem of interactive, time-aware counterfeit detection in agriculture and identify practical constraints that hinder existing VLMs.
- (ii) We present AgriDetectVL, an efficient VLM with a Sequence Prompt Transformer that integrates temporal context and operator feedback, drawing on efficiency principles from recent work and the benefits of interactive sequential prompting.
- (iii) We demonstrate that this integration yields improved accuracy–efficiency trade-offs on agricultural datasets and supports low-latency deployment on resource-constrained devices, a key requirement for real-world monitoring.

## Proposed Method

### Overview

To address the challenges in agricultural product analysis that require contextual understanding, we propose a novel model architecture illustrated in Figure 1(b), designed as an enhancement over a standard Vision Transformer (ViT) as shown in Figure 1(a). Our AgriDetectVL model is built to improve analytical performance by incorporating multimodal prompts and sequential information processing. The architecture processes multiple input streams in parallel, including: a primary image under analysis, a set of historically relevant prompt images, and user-provided textual prompts. Each input modality is handled by a specialized module: Vision Encoder (VE) blocks encode visual information, while a Large Language Model (LLM) interprets natural language instructions. The features extracted from the prompt images are aggregated into a Prompt Pool (PP), which serves as a memory of referential visual knowledge.

The subsequent stage centers on prompt selection and contextual refinement — a key component of the model’s ability to learn from context, shown in Figure 2. To optimize computational efficiency and focus only on the most relevant information, a Top-K Prompt Selector (TPS) is employed. This module compares the primary image’s features with all entries in the Prompt Pool and selects the top K most similar prompts, forming a structured sequence of visual cues. These K prompts are then passed through the Sequence Prompt Transformer (SPT), which models sequential and contextual dependencies among prompts. SPT captures temporal progression and salient similarities to produce a single refined vector representation, enriched with contextual information from the selected prompt sequence.

Finally, multimodal information fusion occurs in the Refined Vision Feature (RVF) block. This component receives two inputs: the sequence-refined visual vector from SPT and the semantic vector from the LLM. This fusion allows the model to simultaneously integrate language understanding (“what to look for”) and historical visual context (“what it used to look like”). The final fused representation from RVF — now contextually enriched — is passed to downstream task-specific heads, such as generating a segmentation mask to highlight disease-affected regions in the agricultural input.

### Model Training

AgriDetectVL unifies vision and language by constructing prompt-ensembled text prototypes for each class and learning an L2-normalized shared embedding space. Class names (and synonyms) are fed through a frozen text encoder and projected to form per-class prototypes. Images are mapped by a visual encoder and projection head, then trained with a prototype-based cross-entropy that directly optimizes class cosine similarities. An optional InfoNCE alignment term pulls each image toward the closest textual prompt of its ground-truth class, while a mild prototype regularizer (orthogonality + weight decay) improves separability and stability. At inference, predictions are obtained by cosine scoring to the class prototypes with temperature scaling, enabling both standard supervised evaluation and seamless zero/few-shot extensions by simply adding new textual labels without retraining the classifier. The entire inference process of AgriDetectVL is summarized in Algorithm 1.

## Experiments

### Implementation Details

The AgriDetectVL model is implemented using PyTorch 2.3.0 and the Hugging Face Transformers library. Architecturally, it leverages a Vision Transformer (ViT-B) as the Vision Encoder (VE), Qwen2 as the Large Language Model (LLM), and a two-layer MLP head for feature fusion and alignment. Core modules for sequential processing—including the Top-K Prompt Selector (TPS) and the Sequence Prompt Transformer (SPT)—are integrated to capture contextual dependencies. The model is initially pretrained on large-scale datasets such as COCO and LVIS, followed by task-specific fine-tuning on our counterfeit agricultural product recognition datasets: Food-101, TLU-Fruit, and TLU-States, shown in Figure 3. Evaluation is performed using standard metrics including F1-Score, AUC (Area Under the ROC Curve), and MCC (Matthews Correlation Coefficient) for prompt-based visual-linguistic interaction tasks.

Our training pipeline consists of several stages: initialization of the fusion head, pretraining of the vision encoder, and multimodal instruction tuning, followed by targeted sequential fine-tuning on downstream datasets. We adopt the AdamW optimizer with a cosine learning rate schedule, using an initial learning rate of 5 × 10−4. All experiments are conducted using a batch size of 16 across four NVIDIA 3090 GPUs (32GB VRAM each), paired with 64GB of system RAM and an Intel i7 10th-generation CPU. To optimize for speed and memory efficiency, we utilize techniques such as mixed-precision training with FP8 and FlashAttention-2 throughout deployment.

### Datasets

- **TLU-Fruit**: Fine-grained fruit varieties dataset.
- **TLU-States**: Fruit state/ripeness dataset.
- **Food-101**: General food recognition dataset used for comparison.

### Results

#### Comparison with Baselines

| Model                  | Dataset    | F1-Score     | AUC          | MCC          |
|------------------------|------------|--------------|--------------|--------------|
| 3*InternVL2 (Fine-tuned) | Food-101  | 85.1 ± 0.6  | 0.912 ± 0.05| 0.795 ± 0.08|
|                        | TLU-Fruit | 82.5 ± 0.7  | 0.881 ± 0.06| 0.751 ± 0.09|
|                        | TLU-States| 80.2 ± 0.9  | 0.856 ± 0.08| 0.713 ± 0.11|
| 3*LLaVA-OV (Fine-tuned) | Food-101  | 86.3 ± 0.5  | 0.925 ± 0.04| 0.810 ± 0.06|
|                        | TLU-Fruit | 84.1 ± 0.6  | 0.903 ± 0.05| 0.776 ± 0.07|
|                        | TLU-States| 81.9 ± 0.8  | 0.874 ± 0.07| 0.742 ± 0.10|
| 3*AgriDetectVL (Ours) | Food-101  | 88.2 ± 0.3  | 0.941 ± 0.02| 0.835 ± 0.04|
|                        | TLU-Fruit | 86.5 ± 0.4  | 0.922 ± 0.03| 0.803 ± 0.05|
|                        | TLU-States| 84.3 ± 0.5  | 0.898 ± 0.04| 0.781 ± 0.06|

#### Performance and Efficiency on Food Dataset

| Model                  | F1-Score (%) | Latency (ms) | Power (W) | Size (MB) |
|------------------------|--------------|--------------|-----------|-----------|
| LLaVA-OV (FP16)       | 86.3        | 210         | 18.5     | 14500    |
| InternVL2 (FP16)       | 85.1        | 195         | 17.2     | 12800    |
| AgriDetectVL (FP16)    | 88.2        | 55          | 8.1      | 310      |
| AgriDetectVL (INT8)    | 87.5        | 32          | 6.5      | 155      |

#### Ablation Studies

| ID | Model Configuration                  | F1-Score     |
|----|--------------------------------------|--------------|
| #1 | Baseline (Vision Encoder only)      | 78.5 ± 0.9  |
| #2 | #1 + Text Prompt (LLM)              | 80.1 ± 0.8  |
| #3 | #2 + Image Prompts (Random K)       | 80.9 ± 1.1  |
| #4 | #2 + Image Prompts (Most recent K)  | 81.8 ± 0.7  |
| #5 | #2 + Image Prompts (TPS - Ours)     | 83.2 ± 0.6  |
| #6 | AgriDetectVL (#5 + SPT)             | 84.3 ± 0.5  |

| Text Prompt | Visual Prompt       | F1-Score (Food-101) | F1-Score (TLU-States) |
|-------------|---------------------|----------------------|------------------------|
| ×          | ×                   | 82.3 ± 0.7          | 78.5 ± 0.9            |
| ✓          | ×                   | 84.1 ± 0.6          | 80.1 ± 0.8            |
| ✓          | ✓ (w/ TPS)          | 85.9 ± 0.4          | 83.2 ± 0.6            |
| ✓          | ✓ (w/ TPS)          | 88.2 ± 0.3          | 84.3 ± 0.5            |

## Limitations and Future Work

While AgriDetectVL demonstrates promising performance, the model still presents several limitations. Its effectiveness remains dependent on the availability of historical visual data, which poses challenges when encountering entirely novel objects in zero-shot scenarios. Additionally, despite architectural optimizations, the model’s composite structure can still be resource-intensive for deployment on ultra-low-power edge devices. Furthermore, the model’s generalization capacity to domains beyond agriculture remains an open question that warrants further investigation.

These limitations inform our future research directions. We plan to explore the integration of few-shot learning techniques to reduce dependency on large-scale annotated datasets, and to investigate more aggressive model compression strategies—including quantization and pruning—for enhanced edge-device efficiency. In parallel, we aim to extend the model’s capabilities toward supporting more complex multimodal dialogues and incorporating additional sensor modalities such as spectral data, ultimately moving toward a more comprehensive and robust decision-support system for real-world applications.

## Conclusions

This paper introduced AgriDetectVL, a domain-specialized vision–language architecture for sequential agricultural analysis and counterfeit-product monitoring. The design combines an efficient “scale-then-compress” backbone with a lightweight Sequence Prompt Transformer (SPT) that aggregates observations across time, and a Text Prompt/Prototype Selection (TPS) mechanism that anchors decisions to language-derived prototypes. This coupling enables the model to leverage historical context and operator feedback while preserving low-latency, low-memory inference suitable for edge platforms.

Extensive experiments on TLU-Fruit and TLU-States show that AgriDetectVL achieves consistent gains over strong CNN, transformer, and VLM baselines across F1, accuracy, AUC, and MCC, with stable behavior under illumination and viewpoint shifts. Ablation studies indicate that (i) sequence-aware prompting (SPT) is the primary driver of temporal robustness, and (ii) TPS/prototype guidance improves calibration and reduces look-alike confusions. Together, these results validate that context accumulation + language prototypes yields a favorable accuracy–efficiency trade-off for field deployment.

We acknowledge limitations. First, while SPT handles short-to-moderate temporal windows, very long-horizon dependencies and abrupt distribution shifts (seasonality, sensors) deserve deeper treatment. Second, current prompting assumes reliable textual labels; noisy or inconsistent terminology can weaken prototype quality. Third, our evaluation focuses on RGB imagery; multimodal sensing (hyperspectral, thermal) and geo-temporal priors could further enhance discriminability.

Future work will pursue (i) continual and federated learning for privacy-preserving on-farm adaptation; (ii) uncertainty estimation and calibration to trigger active queries only when beneficial; (iii) data-efficient domain adaptation to new cultivars and regions; (iv) explainability tools mapping prompts to visual evidence; and (v) energy-aware scheduling for long-endurance deployments. Overall, AgriDetectVL provides a practical and extensible foundation for context-aware VLMs in smart agriculture and offers a blueprint that can transfer to other safety-critical, resource-constrained domains.

## Citation

Please cite this paper as:

D. Tran, A. D. Nguyen and H. N. Vu, "AgriDetectVL: Emphasizes the agriculture-focused application combined with Visual-Language integration," in IEEE Access, vol. 11, 2023, doi: 10.1109/ACCESS.2024.0429000.
