#  The Fine-Tuning Spectrum

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-yellow)](https://huggingface.co/)
[![PEFT](https://img.shields.io/badge/PEFT-LoRA%20%7C%20QLoRA-purple.svg)](https://github.com/huggingface/peft)
[![TRL](https://img.shields.io/badge/TRL-SFTTrainer-green.svg)](https://github.com/huggingface/trl)

A structured, hands-on journey across the entire landscape of model adaptation techniques—from freezing representations to low-rank adapters, instruction fine-tuning, and 4-bit quantization.

---

##  Overview & Conceptual Framework

When adapting foundation models for specialized tasks, machine learning practitioners face trade-offs between **compute budget**, **memory footprint**, **training speed**, and **task accuracy**.

A crucial distinction explored in this repository is:

> **Parameter Update Strategies** define *which* weights are updated during training:
> - **Feature Extraction** (Heads only)
> - **Full Fine-Tuning** (100% of weights)
> - **LoRA** (Low-Rank adapter matrices)
> - **QLoRA** (Low-Rank adapters on top of a 4-bit quantized base)
>
> **Training Methodologies** define the *objective and data format*:
> - **Discriminative / Sequence Classification**
> - **Supervised Fine-Tuning (SFT)** / **Instruction Fine-Tuning** on conversational/prompt-response datasets.

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                             THE FINE-TUNING SPECTRUM                             │
├─────────────────────┬──────────────────────┬─────────────────────────────────────┤
│ Method              │ Trainable Parameters │ Memory & Compute Demand             │
├─────────────────────┼──────────────────────┼─────────────────────────────────────┤
│ Feature Extraction  │ < 1% (Head only)     │ Lowest (Base frozen)                │
│ LoRA (PEFT)         │ ~ 1% (Adapters)      │ Low (Base frozen, small gradients)  │
│ QLoRA               │ ~ 1% (Adapters)      │ Ultra-Low (4-bit NF4 base model)    │
│ Full Fine-Tuning    │ 100% (All weights)   │ Highest (Full optimizer & gradients)│
└─────────────────────┴──────────────────────┴─────────────────────────────────────┘
```

---

## Benchmark Results

### 1. Sequence Classification on IMDb (`distilbert-base-uncased`)

Evaluated on the binary sentiment classification task with 2,000 training samples and 500 test samples:

| Method | Trainable Params | Total Params | % Trainable | Accuracy | Training Speed & Resource Footprint |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Feature Extraction** | 592,130 | 66,955,010 | **0.8844%** | **79.80%** | Fastest training; lowest memory; base model untouched. |
| **LoRA (PEFT)** | 739,586 | 67,694,596 | **1.0925%** | **86.40%** | **Near-full accuracy (-0.8%)** with only ~1.1% trainable weights! |
| **Full Fine-Tuning** | 66,955,010 | 66,955,010 | **100.00%** | **87.20%** | Highest accuracy; requires full gradient + optimizer state memory. |

**Key Takeaway**: LoRA achieves **99.08% of Full Fine-Tuning performance** while modifying barely **1% of total parameters**, avoiding catastrophic forgetting and dramatically slashing training memory requirements.

---

### 2. Instruction Tuning & Generative LLMs (`Qwen/Qwen3-0.6B`)

Evaluated using instruction-tuning datasets (`trl-lib/Capybara`) for chat generation:

| Approach | Base Precision | Quantization | Model Memory Footprint | Method / Trainer |
| :--- | :---: | :---: | :---: | :--- |
| **SFT with LoRA** | FP16 (Half) | None | ~1.20 GB | `TRL` SFTTrainer + `PEFT` LoRA |
| **QLoRA** | FP16 Compute | 4-bit NormalFloat (NF4) | **0.53 GB** *(>55% reduction)* | `BitsAndBytes` + `PEFT` LoRA |

---

## 🛠️ The 5 Fine-Tuning Strategies Explored

### 1. Feature Extraction (Linear Probing)
* **Concept**: The pre-trained transformer acts as a fixed feature extractor. All backbone layers are frozen (`requires_grad = False`), and only the final classification head is trained.
* **Target Architecture**: `distilbert-base-uncased`
* **Use Case**: Quick baselines, minimal compute, or small datasets where full fine-tuning risks catastrophic overfitting.

### 2. Full Fine-Tuning
* **Concept**: Every parameter across every transformer block is updated end-to-end using gradient descent.
* **Target Architecture**: `distilbert-base-uncased`
* **Use Case**: Maximizing task accuracy when sufficient GPU memory and large labeled datasets are available.

### 3. LoRA (Low-Rank Adaptation)
* **Concept**: Instead of modifying the frozen weight matrix $W_0 \in \mathbb{R}^{d \times k}$, LoRA injects trainable rank decomposition matrices:
  $$W = W_0 + \Delta W = W_0 + \frac{\alpha}{r} (B \cdot A)$$
  where $B \in \mathbb{R}^{d \times r}$ and $A \in \mathbb{R}^{r \times k}$ with rank $r \ll \min(d, k)$.
* **Configuration**:
  - Rank $r = 8$
  - Scaling factor $\alpha = 16$
  - Dropout = $0.1$
  - Target modules: `q_lin`, `v_lin`
* **Use Case**: Production fine-tuning, modular multi-tenant adapter deployment, memory-constrained GPUs.

### 4. Supervised Fine-Tuning (SFT) with LoRA
* **Concept**: Combines parameter-efficient adaptation with instruction-following objectives. Converts prompt-completion pairs using standard chat templates and trains causal language models with cross-entropy loss over assistant responses.
* **Model**: `Qwen/Qwen3-0.6B`
* **Dataset**: `trl-lib/Capybara` (multi-turn conversation dataset)
* **Framework**: `trl.SFTTrainer` + `peft.LoraConfig` (`r=16`, `lora_alpha=32`, targets `q_proj`, `v_proj`)
* **Use Case**: Building specialized AI assistants, conversational agents, and domain-adapted chat models.

### 5. QLoRA (Quantized LoRA)
* **Concept**: Quantizes the frozen base model weights to **4-bit NormalFloat (NF4)** with double quantization, while training 16-bit LoRA adapter matrices via dequantization during the forward pass.
* **Memory Reduction**: Drops base model memory from **1.2 GB to 0.53 GB** on standard GPUs.
* **Configuration**:
  - `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)`
* **Use Case**: Fine-tuning medium-to-large language models (7B, 13B, 70B) on consumer-grade hardware or free-tier cloud GPUs (e.g., Google Colab T4).

---

##  Repository Structure

```
FineTuningSpectrum/
├── .gitignore              # Ignores checkpoints, cache files, virtual envs, & local notebooks
├── requirements.txt        # Python dependency specifications
├── README.md               # Project documentation and benchmark report
└── (Colab Notebook)        # Committed directly via Google Colab integration
```

> **Note on Notebook**: The interactive experimentation notebook is maintained and run via **Google Colab** with GPU acceleration. Local checkpoints and large binary outputs are excluded via `.gitignore`.

---

##  Getting Started

### Option A: Run in Google Colab (Recommended)

1. Open Google Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)
2. Switch runtime to GPU: **Runtime** ➔ **Change runtime type** ➔ **T4 GPU** (or higher).
3. Run the cells sequentially to install dependencies and execute each fine-tuning phase.

### Option B: Local Environment Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/<your-username>/FineTuningSpectrum.git
   cd FineTuningSpectrum
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On macOS/Linux:
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify GPU availability**:
   ```python
   import torch
   print("CUDA Available:", torch.cuda.is_available())
   if torch.cuda.is_available():
       print("Device Name:", torch.cuda.get_device_name(0))
   ```

---

##  Engineering Insights & Troubleshooting

* **FP16 vs. BF16 on Tesla T4 GPUs**:
  - Nvidia Turing architecture (Tesla T4) does not have native hardware acceleration for BFloat16 matrix operations.
  - When training with `BitsAndBytes` and `Trainer` on T4, ensure `fp16=True` is used and compute dtype is set to `torch.float16` to prevent gradient scaling unscale errors (`NotImplementedError: "_amp_foreach_non_finite_check_and_unscale_cuda"`).
* **Adapter Portability**:
  - Instead of saving full multi-gigabyte models for each downstream task, LoRA adapters require only **megabytes of storage** and can be dynamically swapped or merged onto the base model at inference time.

---

## References & Further Reading

* **LoRA**: [LoRA: Low-Rank Adaptation of Large Language Models (Hu et al., 2021)](https://arxiv.org/abs/2106.09685)
* **QLoRA**: [QLoRA: Efficient Finetuning of Quantized LLMs (Dettmers et al., 2023)](https://arxiv.org/abs/2305.14314)
* **Hugging Face PEFT**: [Parameter-Efficient Fine-Tuning](https://github.com/huggingface/peft)
* **Hugging Face TRL**: [Transformer Reinforcement Learning & SFT](https://github.com/huggingface/trl)
