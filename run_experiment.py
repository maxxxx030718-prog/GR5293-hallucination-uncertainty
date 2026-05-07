"""Lightweight script entry point.

The full experiment is implemented in notebooks/93PROJECT_final.ipynb.
This script documents the pipeline and writes the final reported results table.
"""

from pathlib import Path
import pandas as pd

from config import SEED, MODEL_NAME, N_SAMPLES


def main():
    print("Statistical Uncertainty Quantification for Hallucination Detection")
    print(f"Seed: {SEED}")
    print(f"Model: {MODEL_NAME}")
    print(f"Sample size: {N_SAMPLES}")
    print("For full reproduction, run notebooks/93PROJECT_final.ipynb.")

    out_dir = Path("results/tables")
    out_dir.mkdir(parents=True, exist_ok=True)

    results = pd.DataFrame([
        {"method": "avg_logprob", "AUROC": 0.683, "AUPRC": 0.895},
        {"method": "min_logprob", "AUROC": 0.694, "AUPRC": 0.903},
        {"method": "std_logprob", "AUROC": 0.715, "AUPRC": 0.908},
        {"method": "avg_entropy", "AUROC": 0.691, "AUPRC": 0.897},
        {"method": "max_entropy", "AUROC": 0.668, "AUPRC": 0.888},
        {"method": "random", "AUROC": 0.519, "AUPRC": 0.813},
        {"method": "combined_lr", "AUROC": 0.723, "AUPRC": 0.913},
    ])

    results.to_csv(out_dir / "final_results_summary.csv", index=False)
    print("Saved results/tables/final_results_summary.csv")


if __name__ == "__main__":
    main()
