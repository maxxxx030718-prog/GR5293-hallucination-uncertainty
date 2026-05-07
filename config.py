"""Configuration for the hallucination uncertainty project."""

SEED = 5293
MODEL_NAME = "gpt2-xl"
DATASET_NAME = "trivia_qa"
DATASET_CONFIG = "rc.nocontext"
SPLIT = "validation"
N_SAMPLES = 500
MAX_NEW_TOKENS = 32
PROMPT_TEMPLATE = "Q: {question}\nA:"

# Binary label convention:
# 1 = hallucinated / unsupported
# 0 = correct / supported
POSITIVE_CLASS = "hallucinated"
