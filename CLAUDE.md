# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Meddies-OCR: VLM Research & Development Project

Đây là project nghiên cứu **Vision Language Models (VLMs)** tập trung vào:
- **Document Understanding & OCR**: Trích xuất text từ tài liệu, đặc biệt handwritten Vietnamese
- **Knowledge Distillation**: On-policy distillation, sparse logit sampling, ACID method
- **Architecture Comparison**: Qwen VL, SigLIP 2, LFM2, CLIP, Chandra OCR

### Team
| Member | Strengths |
|--------|-----------|
| Leo | Research discovery, exploration |
| Bảo | Mathematical formulations, deep paper reading |
| Tân | Mathematical formulations, deep paper reading |
| Hoàng (Lead) | Full-stack expertise |

---

## Domain Knowledge (Bắt buộc đọc)

### 1. VLM Architecture Fundamentals

```
┌─────────────┐     ┌─────────────────┐     ┌─────────────┐
│   Image     │────▶│  Vision Encoder │────▶│             │
│             │     │  (ViT/SigLIP)   │     │   VL        │
└─────────────┘     └─────────────────┘     │  Adapter    │────▶ LLM Decoder ────▶ Output
                                            │ (Projector) │
┌─────────────┐     ┌─────────────────┐     │             │
│   Text      │────▶│   Tokenizer     │────▶│             │
└─────────────┘     └─────────────────┘     └─────────────┘
```

**Layers to fine-tune decision matrix:**
| Task Type | Vision Layers | Adapter | LLM Layers |
|-----------|---------------|---------|------------|
| Image-heavy (drawings, diagrams) | Required | Yes | Optional |
| Text-heavy (OCR) | Optional | Yes | Required |
| Balanced | Yes | Yes | Yes |

### 2. Key Models Reference

#### Qwen 2.5 VL (Primary cho OCR tasks)
```python
# Base performance: 90-95% accuracy on handwriting
# After fine-tuning: 99%+ accuracy
# Critical: Label correctness - 0.5% errors có thể degrade performance

from unsloth import FastVisionModel
model, tokenizer = FastVisionModel.from_pretrained(
    "unsloth/Qwen2.5-VL-7B-Instruct",
    load_in_4bit=False,  # Set True nếu GPU memory limited
    use_gradient_checkpointing="unsloth",
)
```

#### SigLIP 2 (Vision encoder cho downstream tasks)
- **Sigmoid loss** thay vì softmax → better batch efficiency
- **LocCa decoder** cho captioning + localization
- **Self-distillation** + masked prediction (20% cuối training)
- **NaFlex variant**: Native aspect ratio support (tốt cho documents)

#### LFM2-VL-450M (Lightweight option)
- 450M params, suitable cho edge deployment
- Trade-off: Lower accuracy vs inference speed

### 3. Knowledge Distillation Techniques

#### On-Policy Distillation (Chưng cất trên chính sách)
```
Student generates output → Compare with Teacher → Learn from mistakes
                ↑__________________________________|
```
- **Advantage**: Học từ chính mistakes của mình, không chỉ teacher's distribution
- **Use case**: Khi student architecture khác biệt nhiều so với teacher

#### Sparse Logit Sampling
```python
# Problem: Full KD cần lưu ALL teacher logits (vocab_size = 128k+)
# Solution: Chỉ cần 0.01% logits (12 tokens thay vì 64000)

# WRONG: Top-K sampling (biased estimate)
top_k_logits = torch.topk(teacher_logits, k=12)  # Biased!

# CORRECT: Random Sampling (unbiased estimate)
indices = torch.multinomial(probs, num_samples=12)  # Unbiased
sampled_logits = teacher_logits.gather(-1, indices)
```

**Storage comparison:**
| Method | Storage per token | 1T tokens |
|--------|------------------|-----------|
| Full KD | 128K × 2 bytes | 128 PB |
| Top-K (biased) | K × 2 bytes | ~KB |
| Sparse Sampling | K × 2 bytes | ~KB |

---

## Coding Conventions

### Python Style
```python
# Imports order: stdlib → third-party → local
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForVision2Seq

from src.models import QwenVLWrapper
from src.utils import load_config

# Type hints bắt buộc cho public functions
def compute_kl_divergence(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 1.0
) -> torch.Tensor:
    """
    Compute KL divergence for knowledge distillation.

    Args:
        student_logits: Shape (batch, seq_len, vocab_size)
        teacher_logits: Shape (batch, seq_len, vocab_size)
        temperature: Softmax temperature, higher = softer distribution

    Returns:
        KL divergence loss scalar
    """
    # Forward KLD: student learns to match teacher
    student_probs = F.softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)

    return F.kl_div(
        student_probs.log(),
        teacher_probs,
        reduction='batchmean'
    ) * (temperature ** 2)
```

### Config Management
```yaml
# configs/finetune_qwen.yaml
model:
  name: "unsloth/Qwen2.5-VL-7B-Instruct"
  load_in_4bit: false

training:
  learning_rate: 1e-5  # Low LR cho nudging existing capability
  lora_rank: 16        # Low rank cho subtle updates
  epochs: 3
  batch_size: 4
  gradient_accumulation: 8

data:
  max_blank_ratio: 0.3  # Balance blank vs non-blank images
  validation_split: 0.1
```

### Logging & Experiment Tracking
```python
import wandb
from loguru import logger

# Structured logging
logger.info(
    "Training started",
    model=config.model.name,
    lr=config.training.learning_rate,
    dataset_size=len(train_dataset)
)

# Metrics tracking
wandb.log({
    "train/loss": loss.item(),
    "train/accuracy": accuracy,
    "train/lr": scheduler.get_last_lr()[0]
})
```

---

## Project Structure

```
meddies-ocr/
├── CLAUDE.md                 # This file
├── configs/
│   ├── finetune_qwen.yaml
│   ├── distillation.yaml
│   └── evaluation.yaml
├── data/
│   ├── raw/                  # Original datasets
│   ├── processed/            # Preprocessed data
│   └── annotations/          # Label files
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   ├── qwen_wrapper.py
│   │   ├── siglip_encoder.py
│   │   └── distillation.py
│   ├── data/
│   │   ├── dataset.py
│   │   ├── augmentation.py
│   │   └── preprocessing.py
│   ├── training/
│   │   ├── trainer.py
│   │   ├── losses.py
│   │   └── schedulers.py
│   └── evaluation/
│       ├── metrics.py
│       └── benchmarks.py
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_label_review.ipynb      # Critical: Label correction workflow
│   └── 03_model_analysis.ipynb
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   └── inference.py
├── tests/
│   └── test_*.py
└── requirements.txt
```

---

## Common Tasks & Commands

### 1. Fine-tuning Qwen VL cho OCR

```bash
# Step 1: Prepare data
python scripts/prepare_data.py \
    --input data/raw/handwriting/ \
    --output data/processed/ \
    --max_blank_ratio 0.3

# Step 2: Initial prediction (để tạo pseudo-labels)
python scripts/inference.py \
    --model unsloth/Qwen2.5-VL-7B-Instruct \
    --input data/processed/ \
    --output data/annotations/v1.json

# Step 3: Manual correction (Jupyter notebook)
jupyter notebook notebooks/02_label_review.ipynb

# Step 4: Fine-tune
python scripts/train.py \
    --config configs/finetune_qwen.yaml \
    --labels data/annotations/v1_corrected.json

# Step 5: Repeat từ Step 2 cho đến khi converge
```

### 2. Knowledge Distillation Setup

```bash
# Sparse logit distillation
python scripts/distill.py \
    --teacher Qwen/Qwen2.5-VL-72B-Instruct \
    --student Qwen/Qwen2.5-VL-7B-Instruct \
    --method sparse_sampling \
    --num_samples 12 \
    --temperature 2.0
```

### 3. Evaluation

```bash
# Run full benchmark suite
python scripts/evaluate.py \
    --model checkpoints/qwen_finetuned/ \
    --benchmarks ocr_vietnamese,docvqa,handwriting \
    --output results/
```

---

## Critical Pitfalls (Những lỗi cần tránh)

### 1. Label Quality Issues
```python
# WRONG: Bỏ qua label errors
train_dataset = load_dataset(labels_path)
model.fit(train_dataset)

# CORRECT: Validate labels trước khi train
def validate_labels(dataset):
    """
    Check for common labeling errors:
    - Brackets vs parentheses confusion: [ ] vs ( )
    - Border artifacts mistaken as digit "1"
    - Inconsistent whitespace handling
    """
    suspicious = []
    for idx, sample in enumerate(dataset):
        if has_bracket_parens_ambiguity(sample):
            suspicious.append(idx)
    return suspicious

suspicious_indices = validate_labels(train_dataset)
logger.warning(f"Found {len(suspicious_indices)} suspicious labels")
# Manual review required before training!
```

### 2. Learning Rate cho Fine-tuning
```python
# WRONG: Standard LR cho pre-trained model (quá cao)
optimizer = AdamW(model.parameters(), lr=1e-4)

# CORRECT: Low LR vì chỉ cần "nudge" model
optimizer = AdamW(model.parameters(), lr=1e-5)

# BETTER: Different LR cho different layers
optimizer = AdamW([
    {"params": model.vision_encoder.parameters(), "lr": 1e-6},
    {"params": model.adapter.parameters(), "lr": 5e-6},
    {"params": model.llm.parameters(), "lr": 1e-5},
])
```

### 3. Data Imbalance
```python
# WRONG: Train on imbalanced data (70% blank images)
train_dataset = RawDataset(data_path)

# CORRECT: Balance dataset
def balance_dataset(dataset, max_blank_ratio=0.3):
    """
    Limit blank images to avoid model becoming lazy.
    """
    blank = [s for s in dataset if is_blank(s)]
    non_blank = [s for s in dataset if not is_blank(s)]

    max_blank = int(len(non_blank) * max_blank_ratio / (1 - max_blank_ratio))
    balanced_blank = random.sample(blank, min(len(blank), max_blank))

    return non_blank + balanced_blank
```

### 4. Sparse Distillation Bias
```python
# WRONG: Top-K sampling (biased gradient estimate)
def top_k_distill(teacher_logits, k=12):
    values, indices = torch.topk(teacher_logits, k)
    return values, indices  # Biased toward high-probability tokens!

# CORRECT: Importance sampling (unbiased)
def sparse_random_distill(teacher_logits, k=12, temperature=1.0):
    probs = F.softmax(teacher_logits / temperature, dim=-1)
    indices = torch.multinomial(probs, num_samples=k, replacement=False)
    sampled_logits = teacher_logits.gather(-1, indices)
    return sampled_logits, indices
```

---

## Benchmarks & Metrics

### OCR Tasks
| Metric | Description | Target |
|--------|-------------|--------|
| CER (Character Error Rate) | Levenshtein distance / total chars | < 1% |
| WER (Word Error Rate) | Word-level errors | < 5% |
| Exact Match | Full string match | > 95% |

### Document Understanding
| Benchmark | Description | Baseline |
|-----------|-------------|----------|
| DocVQA | Document question answering | Qwen: 91.4 |
| TextVQA | Text reading in natural images | Qwen: 84.3 |
| OCRBench | Comprehensive OCR evaluation | Qwen: 866/1000 |

---

## Key References (Project Knowledge)

### Papers
1. **SigLIP 2** - Sigmoid loss + LocCa decoder + self-distillation
2. **Sparse Logit Sampling** - 0.01% logits, unbiased gradient
3. **CLIP** - Contrastive Language-Image Pre-training baseline

### Implementations
- Unsloth: `https://github.com/unslothai/unsloth` - Fast fine-tuning
- Qwen VL: `https://github.com/QwenLM/Qwen2-VL`
- Chandra OCR: `https://github.com/datalab-to/chandra`

### Blogs & Tutorials
- VLM Fine-tuning for Document Understanding (TowardsDataScience)
- On-Policy Distillation (HuggingFace, ThinkingMachines)

---

## Instructions for Claude Code

### Khi được hỏi về code:
1. **Luôn explain WHY** trước khi viết code
2. **Include type hints** và docstrings
3. **Add comments tiếng Việt** cho logic phức tạp
4. **Reference project knowledge** khi relevant

### Khi debug issues:
1. **Root cause analysis**: Data issue? Architecture? Hyperparams?
2. **Check label quality** - thường là nguyên nhân chính
3. **Verify data balance** - blank images ratio
4. **Review learning rate** - thường quá cao

### Khi so sánh approaches:
1. **Memory efficiency**: GPU VRAM requirements
2. **Training time**: Hours/epochs needed
3. **Inference cost**: Latency, throughput
4. **Accuracy trade-offs**: Benchmark scores

### Response format:
```
1. Intuition: Tại sao approach này make sense
2. Technical: Math/code/architecture details
3. Practical: Deployment, costs, limitations
4. References: Cite papers trong project knowledge
```

---

## Quick Start

```bash
# Clone và setup
git clone <repo-url>
cd meddies-ocr
pip install -r requirements.txt

# Verify installation
python -c "from unsloth import FastVisionModel; print('OK')"

# Run example inference
python scripts/inference.py --model qwen --input examples/
```

---

*Last updated: 2025-01*
*Maintained by: Meddies-OCR Team*
