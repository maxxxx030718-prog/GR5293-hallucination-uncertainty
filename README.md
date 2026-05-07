# Statistical Uncertainty Quantification for Hallucination Detection in Large Language Models

STAT GR5293 — Spring 2026  
Team: Kefei Wang, Yizhi Wang, Yifei Dou

## Project Overview

Large Language Models can generate fluent but factually incorrect answers. This project studies whether token-level statistical uncertainty signals can help detect hallucinated or unsupported model outputs.

We evaluate GPT-2 XL on a closed-book QA setting using TriviaQA. For each generated response, we extract uncertainty features from the model output distribution and use them to classify whether the response is correct or hallucinated.

## Research Question

Can token-level uncertainty signals such as log-probability and entropy detect hallucinated LLM responses?

## Method Summary

The pipeline includes:

1. Load a factual QA dataset.
2. Generate answers using GPT-2 XL.
3. Match generated answers against accepted TriviaQA aliases.
4. Label answers as:
   - `1`: hallucinated / unsupported
   - `0`: correct / supported
5. Extract token-level uncertainty features:
   - average log-probability
   - minimum log-probability
   - standard deviation of log-probability
   - average entropy
   - maximum entropy
6. Evaluate individual features and a combined logistic regression classifier.
7. Analyze performance using AUROC, AUPRC, feature importance, threshold sensitivity, and qualitative error cases.

## Main Results

In our final experiment on 500 TriviaQA validation questions:

| Method | AUROC | AUPRC |
|---|---:|---:|
| avg_logprob | 0.683 | 0.895 |
| min_logprob | 0.694 | 0.903 |
| std_logprob | 0.715 | 0.908 |
| avg_entropy | 0.691 | 0.897 |
| max_entropy | 0.668 | 0.888 |
| random | 0.519 | 0.813 |
| combined logistic regression | 0.723 | 0.913 |

The strongest single feature is `std_logprob`, suggesting that confidence variability across tokens is more informative than average confidence alone. The combined logistic regression classifier performs best overall.

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── notebooks/
│   └── 93PROJECT_final.ipynb
├── src/
│   ├── config.py
│   ├── feature_extraction.py
│   ├── evaluation.py
│   └── run_experiment.py
├── results/
│   ├── figures/
│   └── tables/
├── presentation/
│   └── final_presentation.pptx
└── report/
    └── final_report.docx
```

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

## Reproducing the Experiment

The recommended reproduction path is:

```text
notebooks/93PROJECT_final.ipynb
```

Open the notebook in Google Colab or Jupyter and run all cells.

A lightweight script version is also included:

```bash
python src/run_experiment.py
```

This writes a summary table to:

```text
results/tables/final_results_summary.csv
```

## Notes

- The project uses a closed-book QA setting.
- GPT-2 XL is used because it exposes token-level output scores and is feasible to run for a course project.
- The task is imbalanced because most GPT-2 XL outputs on TriviaQA are hallucinated.
- AUPRC should be interpreted relative to the random baseline because hallucinated outputs are the majority class.
- Token probability reflects linguistic likelihood, not factual truth. This explains blind-confidence failure cases.
