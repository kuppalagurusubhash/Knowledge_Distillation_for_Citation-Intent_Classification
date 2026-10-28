"""Helper script to generate clean, reproducible research notebooks."""
import json
from pathlib import Path

notebooks_dir = Path("notebooks")
notebooks_dir.mkdir(parents=True, exist_ok=True)

def create_nb(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

def code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [s + "\n" for s in source.strip().split("\n")]
    }

def md_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [s + "\n" for s in source.strip().split("\n")]
    }

# 1. Dataset Exploration Notebook
nb1_cells = [
    md_cell("# 01 - SciCite Dataset Exploration and Preprocessing Analysis\nThis notebook inspects the SciCite dataset schema, split counts, and class imbalance."),
    code_cell("import json\nfrom datasets import load_dataset\nimport pandas as pd\nimport matplotlib.pyplot as plt\n\nraw_data = load_dataset('scicite', trust_remote_code=True)\nprint(raw_data)"),
    code_cell("train_df = raw_data['train'].to_pandas()\nprint('Label distribution (0=method, 1=background, 2=result):')\nprint(train_df['label'].value_counts(normalize=True))"),
    code_cell("plt.figure(figsize=(6, 4))\ntrain_df['label'].value_counts().plot(kind='bar', color=['#3B82F6', '#10B981', '#F59E0B'])\nplt.title('SciCite Class Distribution (Train)')\nplt.xticks([0, 1, 2], ['Background', 'Method', 'Result'], rotation=0)\nplt.ylabel('Number of Citations')\nplt.show()")
]
with open(notebooks_dir / "01_dataset_exploration.ipynb", "w", encoding="utf-8") as f:
    json.dump(create_nb(nb1_cells), f, indent=2)

# 2. Teacher Training Notebook
nb2_cells = [
    md_cell("# 02 - SciBERT Teacher Fine-Tuning and Logit Caching\nFine-tunes the SciBERT reference model on citation intent classification and caches logits."),
    code_cell("from src.utils.config import load_config\nfrom src.training.train_teacher import train_teacher\n\nconfig = load_config('configs/quick_run.yaml')\n# Execute teacher training\nresults = train_teacher(config_path='configs/quick_run.yaml', experiment_id='teacher_scibert')\nprint('Test Macro-F1:', results['test_metrics']['macro_f1'])"),
    code_cell("import torch\ncache = torch.load('artifacts/cache/teacher_train_logits.pt')\nprint('Cached teacher logits shape:', cache['logits'].shape)\nprint('Fingerprint:', cache['fingerprint'])")
]
with open(notebooks_dir / "02_teacher_training.ipynb", "w", encoding="utf-8") as f:
    json.dump(create_nb(nb2_cells), f, indent=2)

# 3. Student Distillation Notebook
nb3_cells = [
    md_cell("# 03 - Student Model Knowledge Distillation\nTrains compact students (BERT-Mini, DistilBERT, BERT-Tiny) using Supervised, Logit KD, and Feature Distillation."),
    code_cell("from src.training.train_student import train_student\n\n# 1. Supervised baseline\nres_sup = train_student(student_key='bert_mini', method='supervised', config_path='configs/quick_run.yaml')\n\n# 2. Logit Distillation\nres_kd = train_student(student_key='bert_mini', method='logit_distillation', config_path='configs/quick_run.yaml')\n\nprint('Supervised Macro-F1:', res_sup['test_metrics']['macro_f1'])\nprint('Distilled Macro-F1:', res_kd['test_metrics']['macro_f1'])")
]
with open(notebooks_dir / "03_student_distillation.ipynb", "w", encoding="utf-8") as f:
    json.dump(create_nb(nb3_cells), f, indent=2)

# 4. Benchmark Analysis Notebook
nb4_cells = [
    md_cell("# 04 - Benchmark Analysis and Pareto Frontier Exploration\nEvaluates latency, throughput, model sizes, and robustness."),
    code_cell("import json\nimport pandas as pd\nimport plotly.express as px\n\nwith open('results/benchmarks_summary.json') as f:\n    data = json.load(f)\n\ndf = pd.DataFrame(data['models'])\ndisplay(df[['model_alias', 'method', 'test_macro_f1', 'latency_single_ms', 'throughput_single_eps']])"),
    code_cell("fig = px.scatter(df, x='latency_single_ms', y='test_macro_f1', color='model_alias', title='Latency vs Macro-F1')\nfig.show()")
]
with open(notebooks_dir / "04_benchmark_analysis.ipynb", "w", encoding="utf-8") as f:
    json.dump(create_nb(nb4_cells), f, indent=2)

print("Created 4 Jupyter notebooks successfully.")
