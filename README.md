# Knowledge Distillation for Citation-Intent Classification
### Accuracy–Efficiency Trade-offs and Robustness Preservation in Scientific NLP

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)](https://huggingface.co/)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-ff4b4b.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 1. Overview & Research Objectives

Scientific literature is expanding exponentially. Analyzing and mapping citation graphs requires categorizing why one paper cites another (e.g., as a foundational methodology, context background, or conflicting result comparison). 

While large pre-trained scientific transformer models such as **SciBERT** achieve state-of-the-art intent classification accuracy, their multi-hundred-megabyte size and heavy inference latency impede practical deployment in high-throughput bibliometric search engines, graph indexing, or edge devices.

This research framework investigates **Knowledge Distillation (KD)** to compress domain-specific scientific language models into compact student architectures (**DistilBERT**, **BERT-Mini**, and **BERT-Tiny**) while evaluating:
1. **Accuracy and Semantic Retention**: How much scientific intent categorization is preserved across student model sizes?
2. **Minority-Class Sensitivity**: Whether soft probability targets help student models identify low-frequency classes (specifically, the critical `result` comparison intent).
3. **Inference Latency & Hardware Throughput**: The real empirical latency speedups and memory reductions achieved under standardized hardware benchmarks.
4. **Perturbation Robustness**: Whether compressed models retain semantic stability against typographical errors and citation formatting shifts (e.g., bracket alterations).
5. **Pareto Optimality**: Which student model architecture and distillation objective deliver the optimal balance under strict compute budgets.

---

## 2. Dataset: SciCite

We evaluate on the benchmark **SciCite** dataset (Cohan et al., NAACL 2019) loaded via Hugging Face Datasets:

| Split | Number of Citations | Percentage |
| :--- | :--- | :--- |
| **Train** | 8,194 | 74.7% |
| **Validation** | 916 | 8.3% |
| **Test (Holdout)** | 1,859 | 17.0% |

### Target Classes & Imbalance
The dataset contains three citation intent categories:
- `method` (ID: 0): The cited paper is used as an algorithm, tool, dataset, or methodology (~28-32%).
- `background` (ID: 1): The cited paper provides contextual foundation or related work (~54-59%, Majority).
- `result` (ID: 2): The cited paper's results are directly compared or contrasted (~13-14%, Minority).

Because of this class imbalance, **Macro-F1** and **Minority-Class Recall/F1** are primary evaluation metrics.

---

## 3. Mathematical Foundations of Distillation

### 3.1 Logit Distillation Loss (Hinton et al.)
Let $z_s$ and $z_t$ represent the output logit vectors of the student and teacher models respectively, and $y$ denote the ground-truth class label. The temperature-scaled distillation loss is defined as:

$$\mathcal{L}_{\text{KD}} = T^2 \cdot D_{\text{KL}}\left( \sigma\left(\frac{z_s}{T}\right) \,\Big\|\, \sigma\left(\frac{z_t}{T}\right) \right)$$

where $\sigma(\cdot)$ is the softmax operator, $T > 1$ is the distillation temperature smoothing logit distributions, and $D_{\text{KL}}$ is Kullback-Leibler divergence with `batchmean` reduction.

The complete training objective balances distillation loss with hard cross-entropy:

$$\mathcal{L}_{\text{total}} = \alpha \mathcal{L}_{\text{KD}} + (1 - \alpha) \mathcal{L}_{\text{CE}}(z_s, y)$$

### 3.2 Feature-Based Distillation (Hidden State Projection)
To align intermediate representations between teacher hidden states $h_t \in \mathbb{R}^{d_t}$ and student hidden states $h_s \in \mathbb{R}^{d_s}$ (where $d_s \le d_t$), a learnable linear projection $W_{\text{proj}} \in \mathbb{R}^{d_s \times d_t}$ aligns dimensions:

$$\mathcal{L}_{\text{feat}} = \frac{1}{\sum_{i,j} M_{i,j}} \sum_{i=1}^B \sum_{j=1}^L M_{i,j} \left\| \frac{W_{\text{proj}} h_{s, i, j}}{\|W_{\text{proj}} h_{s, i, j}\|_2} - \frac{h_{t, i, j}}{\|h_{t, i, j}\|_2} \right\|_2^2$$

where $M \in \{0, 1\}^{B \times L}$ is the binary attention mask strictly masking out padding tokens.

---

## 4. Evaluated Model Architectures

| Role | Model Checkpoint | Layers | Hidden Dim | Parameters | Compression Factor |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Teacher** | `allenai/scibert_scivocab_uncased` | 12 | 768 | 110M | 1.0x (Reference) |
| **Student 1** | `distilbert-base-uncased` | 6 | 768 | 66M | 1.7x |
| **Student 2** | `prajjwal1/bert-mini` | 4 | 256 | 11M | **10.0x** |
| **Student 3** | `prajjwal1/bert-tiny` | 2 | 128 | 4.4M | **25.0x** |

---

## 5. Repository Structure

```
knowledge-distillation-citation/
├── app.py                          # Streamlit interactive research platform entrypoint
├── requirements.txt                # Verified dependency specifications
├── README.md                       # Comprehensive project documentation
├── configs/
│   ├── default.yaml                # Standard full-dataset research configuration
│   └── quick_run.yaml              # Fast CPU/smoke-test configuration
├── data/
│   └── README.md                   # SciCite dataset documentation
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_teacher_training.ipynb
│   ├── 03_student_distillation.ipynb
│   └── 04_benchmark_analysis.ipynb
├── src/
│   ├── data/
│   │   ├── dataset_loader.py       # SciCite loading & stratified splitting
│   │   └── preprocessing.py        # Tokenization & PyTorch DataLoaders
│   ├── models/
│   │   ├── model_loader.py         # Config, parameter counting & loading
│   │   ├── teacher.py              # SciBERT wrapper & hidden state extraction
│   │   └── student.py              # Compact student wrappers & projection heads
│   ├── training/
│   │   ├── losses.py               # KD loss & masked feature projection loss
│   │   ├── cache_logits.py         # Pre-softmax teacher logit caching
│   │   ├── distillation.py         # Unified distillation engine
│   │   ├── train_teacher.py        # Teacher training & early stopping
│   │   └── train_student.py        # Modular student training pipeline
│   ├── evaluation/
│   │   ├── classification_metrics.py # Macro-F1, minority F1, confusion matrices
│   │   ├── benchmark.py            # Latency, p95, throughput & parameters
│   │   ├── robustness.py           # Perturbation generation & degradation
│   │   └── pareto.py               # Multi-objective Pareto efficiency calculation
│   └── utils/
│       ├── config.py               # YAML config loader & device selector
│       ├── seed.py                 # Reproducible seeding (Python, NumPy, PyTorch)
│       ├── logging_utils.py        # Structured logging & ExperimentTracker
│       └── checkpointing.py        # Checkpoint serialization & manifest
├── dashboard/
│   └── pages/
│       ├── 1_Dataset_Explorer.py   # Split distributions, class balance & samples
│       ├── 2_Model_Comparison.py   # Multi-model comparisons & radar charts
│       ├── 3_Training_Experiments.py # Loss curves & validation trajectories
│       ├── 4_Pareto_Analysis.py    # Accuracy vs Size / Latency trade-offs
│       ├── 5_Robustness_Analysis.py# Perturbation stability analysis
│       ├── 6_Citation_Predictor.py # Real-time interactive model inference
│       └── 7_Research_Report.py    # Formal research report & data exports
├── scripts/
│   ├── download_data.py            # SciCite downloader and validator
│   ├── train_teacher.py            # Teacher training launcher
│   ├── cache_teacher_logits.py     # Standalone logit caching script
│   ├── train_students.py           # Multi-method student distillation launcher
│   ├── run_benchmarks.py           # Latency/throughput/Pareto runner
│   ├── run_robustness.py           # Controlled perturbation robustness runner
│   └── generate_report.py          # Academic research report generator
├── tests/                          # Complete automated test suite
│   ├── test_data.py
│   ├── test_losses.py
│   ├── test_metrics.py
│   ├── test_benchmark.py
│   └── test_distillation.py
├── artifacts/                      # Model checkpoints and cached logits
└── results/                        # Measured JSON metrics, CSVs, and reports
```

---

## 6. Installation & Quickstart

### 6.1 Local Setup

```bash
# Clone the repository
git clone https://github.com/Subhash/Knowledge_Distillation_for_Citation-Intent_Classification.git
cd Knowledge_Distillation_for_Citation-Intent_Classification

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 6.2 Running Automated Unit Tests
Verify all data pipelines, distillation losses, Pareto calculation, and benchmarking harnesses:
```bash
python -m pytest tests/ -v
```

---

## 7. Execution Workflow (Step-by-Step)

### Step 1: Download & Validate SciCite
```bash
python scripts/download_data.py --config configs/quick_run.yaml
```

### Step 2: Train SciBERT Teacher Model & Cache Logits
```bash
python scripts/train_teacher.py --config configs/quick_run.yaml --experiment_id teacher_scibert
```
*This fine-tunes the SciBERT reference model, computes test evaluation metrics, saves best weights to `artifacts/checkpoints/teacher_scibert`, and automatically caches training logits to `artifacts/cache/teacher_train_logits.pt`.*

### Step 3: Train Distilled Student Models
Train students across all distillation techniques:
```bash
# Method A: Supervised baseline (Cross-Entropy with ground truth)
python scripts/train_students.py --config configs/quick_run.yaml --students bert_mini --methods supervised

# Method B: Logit Distillation (Hinton soft-label distillation)
python scripts/train_students.py --config configs/quick_run.yaml --students bert_mini --methods logit_distillation

# Method C: Feature Distillation (Intermediate hidden representation matching)
python scripts/train_students.py --config configs/quick_run.yaml --students bert_mini --methods feature_distillation

# Method D: Augmented Distillation (Label-preserving text perturbations)
python scripts/train_students.py --config configs/quick_run.yaml --students bert_mini --methods augmented_distillation

# Additional student sizes:
python scripts/train_students.py --config configs/quick_run.yaml --students bert_tiny --methods logit_distillation
```

### Step 4: Run Latency, Throughput & Pareto Benchmarking
```bash
python scripts/run_benchmarks.py --config configs/quick_run.yaml
```

### Step 5: Run Robustness Perturbation Evaluation
```bash
python scripts/run_robustness.py --config configs/quick_run.yaml
```

### Step 6: Generate Research Report
```bash
python scripts/generate_report.py
```

### Step 7: Launch Interactive Streamlit Research Dashboard
```bash
streamlit run app.py
```

---

## 8. Google Colab / Kaggle Instructions

To execute on a free GPU instance (T4 on Google Colab or Kaggle):
1. Clone this repository into the environment:
   ```bash
   !git clone https://github.com/Subhash/Knowledge_Distillation_for_Citation-Intent_Classification.git
   %cd Knowledge_Distillation_for_Citation-Intent_Classification
   !pip install -r requirements.txt
   ```
2. Set `"device": "cuda"` in `configs/default.yaml`.
3. Execute the workflow sequentially or launch the interactive notebooks in `notebooks/`.

---

## 9. Reproducibility & Ethical Considerations

- **No Fabricated Metrics**: All metrics reported in the dashboard and exported tables reflect actual model evaluation passes.
- **Strict Data Leakage Prevention**: Split indices and label distributions are validated. Data-dependent scalers and early stopping are fit strictly on training and validation splits.
- **Deterministic Seeding**: Seeds are set across Python `random`, NumPy, and PyTorch (CPU and CUDA).

---

## 10. License
This project is licensed under the MIT License.
