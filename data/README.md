# SciCite Dataset

This project uses the official **SciCite** dataset for citation-intent classification:
- Source: Allen Institute for AI (AI2) / Hugging Face Datasets (`scicite`)
- Citation: Cohan et al., "Structural Scaffolds for Citation Intent Classification in Scientific Texts", NAACL 2019.

## Target Classes
The dataset annotates citation intents into 3 classes:
- `method` (ID: 0): The cited work is used as a method, tool, algorithm, or dataset.
- `background` (ID: 1): The cited work provides background information, related work, or context.
- `result` (ID: 2): The cited work is compared with the current paper's results or findings.

## Split Sizes
- Train: 8,194 citations
- Validation: 916 citations
- Test: 1,859 citations

## Class Imbalance
The dataset has noticeable class imbalance:
- `background`: ~54% - 59% (Majority)
- `method`: ~28% - 32% (Intermediate)
- `result`: ~13% - 14% (Minority)

This makes macro-F1 and per-class evaluation (especially minority class `result`) crucial metrics.
