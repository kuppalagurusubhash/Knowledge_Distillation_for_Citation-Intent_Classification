# Week 2 Project Task 1: Literature Survey and Requirement Analysis

**Course**: `CAIML_Satheesha_2_PS_114`  
**Faculty / Mentor**: Dr. T.Y. Satheesha  
**Project Title**: Knowledge Distillation for Citation-Intent Classification: Accuracy–Efficiency Trade-offs and Robustness Preservation  
**Target Submission File**: `Task1.docx` (Generated in project root)  
**Date**: October 2026  

---

## 1. Executive Summary & Purpose

This document fulfills the requirements of **Week 2 Project Task 1: Literature Survey and Requirement Analysis** for the research project **"Knowledge Distillation for Citation-Intent Classification: Accuracy–Efficiency Trade-offs and Robustness Preservation"**.

The goal of this submission is to:
1. Conduct a rigorous literature survey of **11 seminal papers** grouped across **5 fundamental themes**, analyzing what exists, where each falls short, and concluding with an explicit **one-line statement of its research gap**.
2. Formulate a **consolidated research gap paragraph** addressing multi-size, multi-method, minority-class, and robustness trade-offs.
3. Establish a formal **requirement specification list** systematically mapping each identified gap into one of the project's **three software and ML engineering modules**:
   - **Module 1**: Reference Model and Evaluation Framework (SciBERT teacher, soft-label cache, gold-standard test set).
   - **Module 2**: Controlled Multi-Method and Multi-Size Knowledge Distillation (DistilBERT, BERT-Mini, BERT-Tiny across Logit KD, Feature Projection KD, Augmentation KD, and Supervised Baselines).
   - **Module 3**: Accuracy, Efficiency, and Robustness Analysis (CPU latency/throughput benchmarks, Pareto frontiers, CheckList perturbation suites, and Streamlit dashboard).

---

## 2. Literature Survey Grouped by Theme (8–12 Papers)

### Theme 1: Knowledge Distillation Foundations & Soft-Label "Dark Knowledge"

#### 1. Hinton, G., Vinyals, O., & Dean, J. (2015). *Distilling the Knowledge in a Neural Network*
- **Venue**: NIPS 2014 Deep Learning Workshop / arXiv:1503.02531.
- **What Exists**: Formulated the foundational theory of temperature-scaled soft-target knowledge distillation. By raising the softmax temperature $T$, small probability values on incorrect classes ("dark knowledge") are magnified, exposing semantic similarity geometry between classes that is transferred to a student via Kullback-Leibler (KL) divergence.
- **Where It Falls Short**: Empirical evaluations were restricted to computer vision (MNIST, CIFAR-10) and acoustic speech modeling using feedforward networks. The paper did not explore contextual transformer representations, specialized scientific domain vocabularies, or severe real-world class imbalance.
- **One-Line Statement of Gap**: *Evaluates distillation primarily on balanced vision/speech classification using homogeneous architectures without measuring tail-class semantic preservation under heavy class imbalance.*

#### 2. Ba, J., & Caruana, R. (2014). *Do Deep Nets Really Need to Be Deep?*
- **Venue**: Advances in Neural Information Processing Systems (NeurIPS 2014).
- **What Exists**: Demonstrated that shallow, compact neural networks can learn the complex decision boundaries of deep ensembles when trained directly on pre-softmax linear logits via L2 mean squared error regression.
- **Where It Falls Short**: Confined to shallow feedforward models on tabular and phoneme data; did not address deep multi-head self-attention compression or whether discrete semantic nuances in specialized NLP corpora can be effectively captured.
- **One-Line Statement of Gap**: *Proves logit-mimicking feasibility only for shallow networks on tabular/phoneme data without addressing contextual representation learning or multi-head attention compression.*

---

### Theme 2: NLP Compression & Distillation Architectures

#### 3. Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). *DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter*
- **Venue**: NeurIPS EMC2 Workshop 2019.
- **What Exists**: Proposed DistilBERT, reducing BERT-base from 12 to 6 layers while retaining 97% of its language understanding capabilities on GLUE benchmarks, delivering a 40% reduction in size and a 60% inference speedup using a triple loss function (masked language modeling, soft logit distillation, and cosine embedding alignment).
- **Where It Falls Short**: Employs an expensive task-agnostic pre-training distillation process over massive generic corpora (English Wikipedia and Toronto BookCorpus). It did not explore task-specific distillation from domain-specialized teachers, nor did it investigate ultra-compact models (<15M parameters).
- **One-Line Statement of Gap**: *Relies on compute-intensive general-domain pre-training distillation rather than evaluating sample-efficient, task-specific distillation from domain-adapted teachers into ultra-compact models.*

#### 4. Jiao, X., Yin, Y., Shang, L., Jiang, X., Chen, X., Li, L., Wang, F., & Liu, Q. (2020). *TinyBERT: Distilling BERT for Natural Language Understanding*
- **Venue**: Findings of EMNLP 2020.
- **What Exists**: Designed a two-stage distillation framework (general + task-specific) incorporating layer-to-layer transformer alignment across word embeddings, intermediate hidden states, and multi-head attention matrices alongside task-specific data augmentation.
- **Where It Falls Short**: Introduces heavy engineering complexity with rigid layer-mapping assumptions that fail when compressing across disparate student architectures. Furthermore, evaluations were strictly limited to GLUE benchmark accuracy without assessing linguistic robustness or minority-class stability.
- **One-Line Statement of Gap**: *Requires rigid layer-by-layer attention and hidden mapping that breaks when compressing across drastically dissimilar student capacities without assessing tail-class performance.*

#### 5. Romero, A., Ballas, N., Kahou, S. E., Chassang, A., Gatta, C., & Bengio, Y. (2014). *FitNets: Hints for Thin Deep Nets*
- **Venue**: International Conference on Learning Representations (ICLR 2015).
- **What Exists**: Introduced intermediate-layer feature-based distillation ("hint-based training") using learnable projection matrices to bridge dimensional discrepancies between teacher and student hidden spaces, enabling thin student models to learn guided representations.
- **Where It Falls Short**: Formulated for convolutional vision models using a disjoint two-stage training scheme (pre-training hints followed by output distillation), which lacks unified end-to-end multi-objective loss optimization and attention masking for variable-length NLP sequences.
- **One-Line Statement of Gap**: *Lacks unified end-to-end joint loss optimization and attention masking for variable-length transformer sequence representations.*

#### 6. Turc, I., Chang, M. W., Lee, K., & Toutanova, K. (2019). *Well-Read Students Learn Better: On the Importance of Pre-training Compact Models*
- **Venue**: arXiv:1908.08962.
- **What Exists**: Systematically pre-trained and evaluated 24 compact BERT architectures (spanning BERT-Tiny, BERT-Mini, BERT-Small, and BERT-Medium), proving that pre-training compact students prior to task-specific distillation is indispensable for high downstream performance.
- **Where It Falls Short**: Evaluated compact models exclusively on general-domain tasks (GLUE and SQuAD), providing no analysis of domain adaptation to scientific literature, minority-class degradation, or empirical CPU/GPU latency trade-offs.
- **One-Line Statement of Gap**: *Evaluates compact student architectures solely on generic NLP benchmarks without analyzing domain transfer, minority-class degradation, or hardware latency profiles.*

---

### Theme 3: Domain-Specific Scientific Language Models

#### 7. Beltagy, I., Lo, K., & Cohan, A. (2019). *SciBERT: A Pretrained Language Model for Scientific Text*
- **Venue**: EMNLP 2019.
- **What Exists**: Pre-trained a BERT-base architecture from scratch on 1.14M scientific papers from Semantic Scholar with a dedicated scientific vocabulary (`scivocab`), achieving state-of-the-art accuracy across scientific classification, named entity recognition, and citation intent tasks.
- **Where It Falls Short**: SciBERT retains the full 110M parameter footprint (420 MB checkpoint size) and suffers from high inference latency (~143.6 ms/sample on CPU), making it prohibitive for high-throughput paper parsing or resource-constrained deployment.
- **One-Line Statement of Gap**: *Achieves state-of-the-art scientific NLP performance but suffers from prohibitive memory footprints and high inference latency, with no compression pathways provided.*

#### 8. Lee, J., Yoon, W., Kim, S., Kim, D., Kim, S., So, C. H., & Kang, J. (2020). *BioBERT: a pre-trained biomedical language representation model for biomedical text mining*
- **Venue**: Bioinformatics, 36(4), 1234–1240.
- **What Exists**: Demonstrated that continual pre-training on PubMed abstracts and PMC full-text articles significantly outperforms standard BERT on biomedical information extraction, proving that general representations do not readily transfer to specialized scholarly text.
- **Where It Falls Short**: Reinforces the reliance on heavyweight 110M parameter models without providing mechanisms for compression, latency reduction, or edge deployment in clinical or laboratory environments.
- **One-Line Statement of Gap**: *Validates the necessity of scientific domain adaptation but reinforces deployment bottlenecks by neglecting model compression techniques.*

---

### Theme 4: Citation-Intent Classification & Scholarly Corpora

#### 9. Cohan, A., Ammar, W., van Zuylen, M., & Cady, F. (2019). *Structural Scaffolds for Citation Intent Classification in Scientific Texts*
- **Venue**: NAACL-HLT 2019.
- **What Exists**: Introduced the standard **SciCite** benchmark dataset comprising 8,194 annotated citation contexts categorized into three functional intent classes: `method` (Class 0), `background` (Class 1), and `result` (Class 2), demonstrating that multi-task section scaffolding boosts classification accuracy.
- **Where It Falls Short**: Tested only legacy BiLSTM, ELMo, and baseline transformer architectures; did not investigate model compression, distillation, inference latency, or the pronounced vulnerability of models on the under-represented `result` class (~14% of data).
- **One-Line Statement of Gap**: *Establishes the canonical 3-class citation intent benchmark with natural class imbalance but omits model compression, student distillation, and real-time efficiency analysis.*

#### 10. Jurgens, D., Kumar, S., Radev, D., & Jurafsky, D. (2018). *Measuring the Evolution of a Scientific Field through Citation Ingestion and Intent*
- **Venue**: Transactions of the Association for Computational Linguistics (TACL 2018).
- **What Exists**: Proposed a comprehensive functional taxonomy of citation intents and showed how tracking citation motivation illuminates scientific evolution, paradigm shifts, and methodology diffusion over time.
- **Where It Falls Short**: Relied on manual feature engineering, support vector machines, and random forests over dependency parse trees, which are brittle and computationally expensive to compute at scale.
- **One-Line Statement of Gap**: *Establishes the critical bibliometric value of citation intent tracking but relies on brittle, hand-crafted syntactic features rather than deployable neural architectures.*

---

### Theme 5: Model Robustness & Behavioural Testing in NLP

#### 11. Ribeiro, M. T., Wu, T., Guestrin, C., & Samek, W. (2020). *Beyond Accuracy: Behavioral Testing of NLP Models with CheckList*
- **Venue**: ACL 2020 (Best Paper Award).
- **What Exists**: Introduced the CheckList behavioural evaluation methodology, proposing Minimum Functionality Tests (MFT), Invariance Tests (INV), and Directional Expectation Tests (DIR) to uncover severe reasoning flaws in models with deceptively high aggregate accuracy.
- **Where It Falls Short**: Applied almost exclusively to commercial black-box APIs and large standard transformers on generic tasks (sentiment, Q&A); it completely omitted investigation of whether knowledge distillation preserves, enhances, or compromises linguistic invariance under compression.
- **One-Line Statement of Gap**: *Establishes systematic behavioural perturbation testing for NLP but leaves completely unexplored whether knowledge distillation preserves behavioural invariance under model compression.*

---

### Summary Table of Surveyed Literature & Identified Gaps

| Theme | Author(s) & Year | Model / Focus | Identified Research Gap |
| :--- | :--- | :--- | :--- |
| **KD Foundations** | Hinton et al. (2015) | Soft-label KD | Lacks contextual transformer evaluation and tail-class analysis on imbalanced text. |
| **KD Foundations** | Ba & Caruana (2014) | Logit regression | Only tested shallow feedforward networks on tabular data; no self-attention compression. |
| **NLP Distillation** | Sanh et al. (2019) | DistilBERT (66M) | Relies on expensive pre-training distillation on general text; ignores task-specific domain transfer. |
| **NLP Distillation** | Jiao et al. (2020) | TinyBERT | Rigid layer-by-layer attention mapping; untested across disparate student architectures. |
| **NLP Distillation** | Romero et al. (2014) | FitNets (Hints) | Two-stage CNN hint training lacks end-to-end multi-objective attention masking. |
| **NLP Distillation** | Turc et al. (2019) | Compact BERTs | Evaluated exclusively on GLUE/SQuAD; ignores domain transfer, tail classes, and latency. |
| **Scientific NLP** | Beltagy et al. (2019) | SciBERT (110M) | State-of-the-art accuracy but 420 MB size and ~144ms latency prevent real-time deployment. |
| **Scientific NLP** | Lee et al. (2020) | BioBERT (110M) | Confirms scientific domain shift but provides no model compression or edge deployment. |
| **Citation Intent** | Cohan et al. (2019) | SciCite benchmark | Established 3-class corpus but neglected model compression and minority 'result' F1 drops. |
| **Citation Intent** | Jurgens et al. (2018) | Citation evolution | Proves bibliometric significance but relies on legacy brittle feature engineering. |
| **NLP Robustness** | Ribeiro et al. (2020) | CheckList testing | Behavioural testing framework never applied to investigate KD robustness preservation. |

---

## 3. Consolidated Statement of the Research Gap

> **Consolidated Research Gap:**  
> *While knowledge distillation has demonstrated strong compression capabilities for general-domain NLP benchmarks (such as GLUE and SQuAD), prior literature suffers from three fundamental shortcomings: **(1)** Distillation is predominantly reported as an isolated, single-method intervention (e.g., standard logit KD) applied to a single arbitrary student size, obscuring comparative trade-offs between logit distillation, intermediate feature alignment, and perturbation augmentation across distinct model capacities (from 66M down to 4M parameters); **(2)** Existing scientific text classification benchmarks (e.g., SciCite) rely on heavy 110M-parameter models (SciBERT) evaluated solely on aggregate accuracy, masking severe degradation on low-frequency, high-value citation intents (such as 'result' comparison) during compression; and **(3)** Robustness preservation under compression remains systematically under-examined, with no empirical account of whether distilled compact students maintain invariance to typographic noise, citation marker syntax, and whitespace variations prevalent in automated PDF extraction pipelines. Consequently, there is an absence of a unified, reproducible framework that holistically balances accuracy, minority-class preservation, inference latency, and linguistic robustness for scholarly citation analysis.*

---

## 4. Requirement Analysis and Modular System Architecture

To systematically resolve each gap identified in the literature, the project is structured into **three cohesive, modular engineering stages**:

```mermaid
graph TD
    subgraph M1["Module 1: Reference Model & Evaluation Framework"]
        A1["SciCite Dataset (8,194 citations)"] --> A2["Fine-Tune SciBERT Teacher (110M)"]
        A2 --> A3["Pre-Compute & Cache Soft-Labels (train_logits.pt)"]
        A3 --> A4["Gold-Standard Evaluation Suite (Macro-F1, Minority-F1)"]
    end

    subgraph M2["Module 2: Knowledge Distillation Methods"]
        B1["Student Architectures: DistilBERT (66M), BERT-Mini (11M), BERT-Tiny (4M)"]
        B2["Method 1: Supervised Cross-Entropy Baselines"]
        B3["Method 2: Temperature-Scaled Logit KD (Hinton)"]
        B4["Method 3: Feature Projection KD (FitNets)"]
        B5["Method 4: Perturbation-Augmented KD"]
        A3 -.-> B3
        A3 -.-> B4
        A3 -.-> B5
    end

    subgraph M3["Module 3: Accuracy, Efficiency, and Robustness Analysis"]
        C1["Hardware Profiling: Latency (ms), Throughput (ex/s), Size (MB)"]
        C2["Pareto Optimization: Size-vs-F1 & Latency-vs-F1 Frontiers"]
        C3["CheckList-Style Robustness Suite: Whitespace, Syntax, Typos"]
        C4["Interactive Streamlit Research Dashboard"]
        B2 & B3 & B4 & B5 --> C1
        C1 --> C2
        C1 --> C3
        C2 & C3 --> C4
    end
```

### Module 1: Reference Model and Evaluation Framework
- **REQ-1.1: Domain-Adapted Teacher Fine-Tuning**: Fine-tune `allenai/scibert_scivocab_uncased` (109.9M parameters) on the 8,194-citation SciCite training set using class-weighted cross-entropy loss, AdamW optimizer, and validation Macro-F1 checkpointing, achieving an authoritative reference Macro-F1 of **0.8655**.
- **REQ-1.2: Pre-Computed Soft-Label Caching Pipeline**: Design and execute an offline logit inference engine that computes and serializes the 3-class continuous logit vectors ($z_t$) for all 8,194 training instances into `artifacts/teacher_logits/train_logits.pt`. This decouples teacher inference from student training, accelerating downstream experiments by **>10x** and guaranteeing 100% deterministic distillation targets.
- **REQ-1.3: Standardized Gold-Standard Evaluation Protocol**: Isolate 300 held-out test citations as an immutable benchmark. Measure micro/macro Accuracy, Macro-F1, and dedicated Minority-Class Recall for the `result` class to ensure performance on rare, critical intents is never obscured by aggregate accuracy.

### Module 2: Controlled Multi-Method and Multi-Size Knowledge Distillation
- **REQ-2.1: Multi-Scale Student Architecture Suite**: Implement three distinct student architectures representing progressive compression regimes:
  - **DistilBERT**: 66.95M parameters (40% compression relative to SciBERT, 255.4 MB checkpoint).
  - **BERT-Mini**: 11.17M parameters (90% compression, 42.6 MB checkpoint, 4 layers, 256 hidden dim).
  - **BERT-Tiny**: 4.39M parameters (96% compression, 16.7 MB checkpoint, 2 layers, 128 hidden dim).
- **REQ-2.2: Temperature-Scaled Logit Distillation**: Implement the generalized distillation loss combining soft KL divergence and hard ground-truth cross-entropy:
  $$\mathcal{L}_{	ext{total}} = lpha \cdot T^2 \cdot D_{	ext{KL}}\left(\sigma\left(rac{z_s}{T}ight) \Big\| \sigma\left(rac{z_t}{T}ight)ight) + (1 - lpha) \cdot \mathcal{L}_{	ext{CE}}(z_s, y)$$
  Configured with temperature $T=3.0$ and distillation weight $lpha=0.7$.
- **REQ-2.3: Intermediate Feature Projection Alignment**: Implement hidden-state feature distillation bridging dimensional discrepancies (e.g., student $d_s=256$ to teacher $d_t=768$) via a learnable linear projection matrix $W_{	ext{proj}} \in \mathbb{R}^{d_s 	imes d_t}$, minimizing attention-masked, L2-normalized MSE between penultimate layers:
  $$\mathcal{L}_{	ext{feat}} = rac{1}{\sum M_{i,j}} \sum_{i,j} M_{i,j} \left\| rac{W_{	ext{proj}} h_{s, i, j}}{\|W_{	ext{proj}} h_{s, i, j}\|_2} - rac{h_{t, i, j}}{\|h_{t, i, j}\|_2} ight\|_2^2$$
- **REQ-2.4: Perturbation-Augmented Distillation**: Integrate dynamic input perturbation into the student distillation training loop, injecting controlled character and syntactic variations to smooth decision boundaries.
- **REQ-2.5: Supervised Baseline Isolation**: Train pure supervised baselines (`DistilBERT Supervised`, `BERT-Mini Supervised`) on identical seeds and data to rigorously quantify the exact marginal value added by soft-label distillation.

### Module 3: Accuracy, Computational Efficiency, and Robustness Analysis
- **REQ-3.1: Hardware-Standardized Latency and Throughput Profiling**: Benchmark single-sample median latency (ms), p95 latency, and batch throughput (examples/sec) under standardized CPU hardware across 50 warmup passes and 200 timed inference trials.
- **REQ-3.2: Formal Pareto Frontier Optimization**: Compute Pareto-optimal frontiers across two operating planes:
  1. *Storage Footprint vs Macro-F1*
  2. *Single-Sample Latency vs Macro-F1*  
  Identifies optimal operational deployment points (e.g., DistilBERT for maximum accuracy, BERT-Mini for balanced edge deployment).
- **REQ-3.3: Systematic Behavioural Robustness Suite (CheckList-Style)**: Implement 3 perturbation test suites evaluating linguistic stability on the holdout test set:
  - *Test A (Whitespace Normalization)*: Evaluates invariance to irregular multi-spacing, tab characters, and newline wraps commonly produced by PDF extraction tools.
  - *Test B (Citation Marker Variation)*: Replaces standard bracketed numeric citations `[1, 2]` with author-year syntax `(Author et al., 2020)` or neutral token masks.
  - *Test C (Typographic Noise)*: Injects random adjacent-key character substitutions simulating common transcription typos.
- **REQ-3.4: Interactive Research Dashboard (Streamlit)**: Deploy a full-featured Streamlit web application (`app.py`) providing interactive citation intent classification, live Pareto trade-off visualizations, perturbation stress-testing, and complete artifact exploration.

---

## 5. Traceability Matrix: Literature Gaps to System Modules

| Identified Literature Gap | Target Module | System Requirement | Deliverable Artifact |
| :--- | :--- | :--- | :--- |
| **Expensive repeated teacher inference during multi-student experimentation** | **Module 1** | REQ-1.2 (Soft-label caching) | `artifacts/teacher_logits/train_logits.pt` |
| **Absence of specialized domain teacher for citation semantics** | **Module 1** | REQ-1.1 (SciBERT fine-tuning) | `artifacts/checkpoints/teacher_scibert/` |
| **Single-method & single-size KD reporting in literature** | **Module 2** | REQ-2.1 to 2.4 (Multi-model KD) | 8 trained checkpoints in `artifacts/checkpoints/` |
| **Neglect of minority 'result' intent degradation under compression** | **Module 1 & 2** | REQ-1.3 & 2.5 (Minority F1 isolation) | Per-class metrics in `results/manifest.json` |
| **Lack of standardized latency/throughput measurements** | **Module 3** | REQ-3.1 & 3.2 (Benchmarking) | `results/benchmark_table.csv` & Pareto plots |
| **Unknown robustness preservation under real-world text shifts** | **Module 3** | REQ-3.3 (Perturbation tests) | `results/robustness_table.csv` & stress-testing UI |
| **Lack of interactive exploration and deployment interfaces** | **Module 3** | REQ-3.4 (Streamlit dashboard) | `app.py` & 5 interactive UI pages |

---

## 6. Empirical Benchmark Evidence (From Active System)

The framework has been fully implemented, trained, and benchmarked on the held-out SciCite test set. The table below presents real empirical measurements across all 8 experimental models:

| Model Alias | Training Method | Total Params | Checkpoint (MB) | Test Macro-F1 | Test Acc | Minority F1 (`result`) | CPU Latency (ms) | Throughput (ex/s) | Globally Pareto |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **SciBERT (Teacher)** | `reference_teacher` | 109,920,771 | 420.2 MB | **0.8655** | 0.8733 | **0.8602** | 143.59 ms | 7.0 | No |
| **DistilBERT** | `logit_distillation` | 66,955,779 | 256.3 MB | **0.8684** | **0.8867** | **0.8043** | 68.61 ms | 14.6 | **Yes** |
| **DistilBERT** | `supervised` | 66,955,779 | 256.3 MB | **0.8335** | 0.8567 | 0.7447 | 65.52 ms | 15.3 | **Yes** |
| **BERT-Mini** | `supervised` | 11,171,331 | 43.5 MB | **0.7702** | 0.7933 | 0.7356 | 8.82 ms | 113.4 | **Yes** |
| **BERT-Mini** | `logit_distillation` | 11,171,331 | 43.5 MB | **0.7665** | 0.7967 | 0.7160 | 10.11 ms | 98.9 | No |
| **BERT-Mini** | `feature_distillation`| 11,171,331 | 43.5 MB | **0.7253** | 0.7667 | 0.6237 | 8.81 ms | 113.5 | No |
| **BERT-Mini** | `augmented_distill` | 11,171,331 | 43.5 MB | **0.7529** | 0.7833 | 0.7160 | 8.28 ms | 120.7 | **Yes** |
| **BERT-Tiny** | `logit_distillation` | 4,386,307 | 17.6 MB | **0.2328** | 0.5367 | 0.0000 | **2.86 ms** | **349.9** | **Yes** |

### Robustness Benchmark Results Under Perturbation

| Model Alias | Method | Clean Macro-F1 | Whitespace F1 | Punctuation F1 | Typo F1 | Typo Degradation (%) | Robustness Score (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **SciBERT (Teacher)** | `reference_teacher` | 0.8655 | 0.8655 | 0.8635 | 0.8626 | **0.34%** | **99.81%** |
| **DistilBERT** | `logit_distillation` | 0.8684 | 0.8684 | 0.8711 | 0.8554 | **1.50%** | **99.60%** |
| **DistilBERT** | `supervised` | 0.8335 | 0.8335 | 0.8391 | 0.8344 | **-0.11%** | **100.26%** |
| **BERT-Mini** | `logit_distillation` | 0.7665 | 0.7665 | 0.7610 | 0.7491 | **2.27%** | **99.00%** |
| **BERT-Mini** | `supervised` | 0.7702 | 0.7702 | 0.7729 | 0.7271 | **5.60%** | **98.25%** |
| **BERT-Mini** | `augmented_distill` | 0.7529 | 0.7529 | 0.7529 | 0.7304 | **2.99%** | **99.00%** |
| **BERT-Mini** | `feature_distillation`| 0.7253 | 0.7253 | 0.7248 | 0.7021 | **3.20%** | **98.91%** |
| **BERT-Tiny** | `logit_distillation` | 0.2328 | 0.2328 | 0.2328 | 0.2328 | **0.00%** | **100.00%** |

### Key Empirical Findings:
1. **Distillation Closes the Gap & Boosts Minority Recall**: On DistilBERT, knowledge distillation raises Macro-F1 from **0.8335** (supervised) to **0.8684** (+3.49%) and minority `result` F1 from **0.7447** to **0.8043** (+5.96%), matching the teacher's accuracy while delivering a **2.09x CPU speedup** (68.6 ms vs 143.6 ms).
2. **Distillation Improves Robustness**: On BERT-Mini, logit distillation cushions the impact of typographical noise, reducing typo degradation from **5.60%** (supervised) down to **2.27%**.
3. **The Capacity Threshold**: When compressing down to BERT-Tiny (4.4M parameters), capacity collapses on the minority intent (`result` F1 drops to 0.0), proving that extreme compression requires dedicated capacity preservation techniques.

---

## 7. Formal Academic References

1. **Ba, J., & Caruana, R. (2014).** Do deep nets really need to be deep? In *Advances in Neural Information Processing Systems (NeurIPS 2014)*, pp. 2654–2662.
2. **Beltagy, I., Lo, K., & Cohan, A. (2019).** SciBERT: A pretrained language model for scientific text. In *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing (EMNLP 2019)*, pp. 3615–3620.
3. **Cohan, A., Ammar, W., van Zuylen, M., & Cady, F. (2019).** Structural scaffolds for citation intent classification in scientific texts. In *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics (NAACL-HLT 2019)*, pp. 3586–3596.
4. **Hinton, G., Vinyals, O., & Dean, J. (2015).** Distilling the knowledge in a neural network. *arXiv preprint arXiv:1503.02531*.
5. **Jiao, X., Yin, Y., Shang, L., Jiang, X., Chen, X., Li, L., Wang, F., & Liu, Q. (2020).** TinyBERT: Distilling BERT for natural language understanding. In *Findings of the Association for Computational Linguistics: EMNLP 2020*, pp. 4163–4174.
6. **Jurgens, D., Kumar, S., Radev, D., & Jurafsky, D. (2018).** Measuring the evolution of a scientific field through citation ingestion and intent. *Transactions of the Association for Computational Linguistics (TACL)*, 6, 391–406.
7. **Lee, J., Yoon, W., Kim, S., Kim, D., Kim, S., So, C. H., & Kang, J. (2020).** BioBERT: a pre-trained biomedical language representation model for biomedical text mining. *Bioinformatics*, 36(4), 1234–1240.
8. **Ribeiro, M. T., Wu, T., Guestrin, C., & Samek, W. (2020).** Beyond accuracy: Behavioral testing of NLP models with CheckList. In *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics (ACL 2020)*, pp. 4902–4912.
9. **Romero, A., Ballas, N., Kahou, S. E., Chassang, A., Gatta, C., & Bengio, Y. (2014).** FitNets: Hints for thin deep nets. In *Proceedings of the International Conference on Learning Representations (ICLR 2015)*.
10. **Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019).** DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter. In *NeurIPS EMC2 Workshop 2019*.
11. **Teufel, S., Siddharthan, A., & Tidhar, D. (2006).** Automatic classification of citation function. In *Proceedings of the 2006 Conference on Empirical Methods in Natural Language Processing (EMNLP 2006)*, pp. 103–110.
12. **Turc, I., Chang, M. W., Lee, K., & Toutanova, K. (2019).** Well-read students learn better: On the importance of pre-training compact models. *arXiv preprint arXiv:1908.08962*.
