"""Feature extraction utilities for token-level uncertainty analysis."""

import numpy as np
import torch


def token_entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Compute token entropy from logits."""
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = torch.softmax(logits, dim=-1)
    return -(probs * log_probs).sum(dim=-1)


def extract_uncertainty_features(step_scores, generated_token_ids):
    """Extract uncertainty features from generated token scores.

    Args:
        step_scores: list of tensors, one per generated token.
        generated_token_ids: token ids selected during generation.

    Returns:
        Dictionary with log-probability and entropy features.
    """
    token_logprobs = []
    token_entropies = []

    for score, token_id in zip(step_scores, generated_token_ids):
        if score.dim() == 2:
            score = score[0]

        log_probs = torch.log_softmax(score, dim=-1)
        entropy = token_entropy_from_logits(score)

        token_logprobs.append(float(log_probs[int(token_id)].detach().cpu()))
        token_entropies.append(float(entropy.detach().cpu()))

    if len(token_logprobs) == 0:
        return {
            "avg_logprob": np.nan,
            "min_logprob": np.nan,
            "std_logprob": np.nan,
            "avg_entropy": np.nan,
            "max_entropy": np.nan,
            "answer_length": 0,
        }

    return {
        "avg_logprob": float(np.mean(token_logprobs)),
        "min_logprob": float(np.min(token_logprobs)),
        "std_logprob": float(np.std(token_logprobs)),
        "avg_entropy": float(np.mean(token_entropies)),
        "max_entropy": float(np.max(token_entropies)),
        "answer_length": int(len(token_logprobs)),
    }
