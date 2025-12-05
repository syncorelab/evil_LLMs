from pathlib import Path

# ======== ROOT PATH ========

# Folder where THIS file lives:
PROJECT_ROOT = Path(__file__).resolve().parent

# ======== PATHS ========

# 1) Where students put their training data
DATASET_PATH = PROJECT_ROOT / "data" / "dataset.jsonl"

# 2) Where ALL LoRA outputs go
OUTPUT_ROOT = PROJECT_ROOT / "output"

# 3) Name of THIS persona experiment
EXPERIMENT_NAME = "unethical_interview_coach_lora"


# ======== MODEL SETTINGS ========

# Unsloth-optimized Llama 3.1 8B
BASE_MODEL_NAME = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"

MAX_SEQ_LENGTH = 1024
DTYPE = None              
LOAD_IN_4BIT = True      


# ======== LoRA HYPERPARAMETERS ========

LORA_R = __
LORA_ALPHA = __
LORA_DROPOUT = ____
LORA_BIAS = "____"
USE_GRADIENT_CHECKPOINTING = "unsloth"

LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


# ======== TRAINING HYPERPARAMETERS ========

BATCH_SIZE = _
GRADIENT_ACCUMULATION_STEPS = _   
NUM_EPOCHS = _
LEARNING_RATE = ____

LOGGING_STEPS = __
SAVE_STEPS = ___


# ======== DEFAULT PERSONA SYSTEM PROMPT ========

SYSTEM_PROMPT = (
    "Add the system prompt here"
)
