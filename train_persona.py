"""
train_persona.py

Fine-tunes a Llama 3.1 8B model (via Unsloth) with LoRA
to create a persona-based LLM.

- Uses settings from persona_config.py
- Uses data from data/dataset.jsonl
- Saves LoRA adapter + tokenizer into: output/EXPERIMENT_NAME

Students should mainly edit:
    - persona_config.py
    - data/dataset.jsonl

You can run this file directly:
    source .venv/bin/activate
    python train_persona.py

or import it in a notebook:

    import train_persona
    train_persona.train()
"""

from unsloth import FastLanguageModel, is_bfloat16_supported
from pathlib import Path
import json

import torch
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments

import persona_config as cfg


def load_dataset():
    """
    Load data from dataset.jsonl.

    Expected format (one JSON per line):
        {
            "system": "...",      # optional, can be ""
            "user": "...",        # required
            "assistant": "..."    # required (the persona's answer)
        }
    """
    data_path: Path = cfg.DATASET_PATH
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset file not found at {data_path}\n"
            "Create data/dataset.jsonl with one JSON object per line."
        )

    rows = []
    with data_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            # Basic sanity check
            if "user" not in obj or "assistant" not in obj:
                continue
            rows.append(obj)

    if not rows:
        raise ValueError("No valid examples found in dataset.jsonl.")

    print(f"Loaded {len(rows)} training examples from {data_path}")
    return Dataset.from_list(rows)


def prepare_text_dataset(raw_dataset, tokenizer):
    """
    Convert (system, user, assistant) into chat-formatted text
    using tokenizer.apply_chat_template.

    If system == "" or missing, use cfg.SYSTEM_PROMPT instead.
    """

    def formatting_prompts_func(examples):
        systems = examples.get("system", [""] * len(examples["user"]))
        users = examples["user"]
        assistants = examples["assistant"]

        texts = []
        for system, user, assistant in zip(systems, users, assistants):
            system_text = (system or "").strip() or cfg.SYSTEM_PROMPT

            messages = [
                {"role": "system",    "content": system_text},
                {"role": "user",      "content": user},
                {"role": "assistant", "content": assistant},
            ]

            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            texts.append(text)

        return {"text": texts}

    dataset = raw_dataset.map(formatting_prompts_func, batched=True)
    print("Example formatted text:\n", dataset[0]["text"][:400], "\n")
    return dataset


def load_model_and_tokenizer():
    """
    Load base model + tokenizer using Unsloth FastLanguageModel.
    """
    print(f"Loading base model: {cfg.BASE_MODEL_NAME}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = cfg.BASE_MODEL_NAME,
        max_seq_length = cfg.MAX_SEQ_LENGTH,
        dtype          = cfg.DTYPE,
        load_in_4bit   = cfg.LOAD_IN_4BIT,
    )
    print("Model and tokenizer loaded.")
    return model, tokenizer


def attach_lora(model):
    """
    Attach LoRA adapters to the model using Unsloth.
    """
    print("Attaching LoRA adapter...")
    model = FastLanguageModel.get_peft_model(
        model,
        r                          = cfg.LORA_R,
        target_modules             = cfg.LORA_TARGET_MODULES,
        lora_alpha                 = cfg.LORA_ALPHA,
        lora_dropout               = cfg.LORA_DROPOUT,
        bias                       = cfg.LORA_BIAS,
        use_gradient_checkpointing = cfg.USE_GRADIENT_CHECKPOINTING,
        random_state               = 42,
        use_rslora                 = False,
        loftq_config               = None,
    )
    print("LoRA adapter attached.")
    return model


def make_trainer(model, tokenizer, dataset):
    """
    Build an SFTTrainer for supervised fine-tuning.
    """
    bf16 = is_bfloat16_supported()
    print("bfloat16 supported:", bf16)

    checkpoints_root = cfg.OUTPUT_ROOT / "checkpoints"
    checkpoints_root.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir                  = str(checkpoints_root),
        per_device_train_batch_size = cfg.BATCH_SIZE,
        gradient_accumulation_steps = cfg.GRADIENT_ACCUMULATION_STEPS,
        num_train_epochs            = cfg.NUM_EPOCHS,
        learning_rate               = cfg.LEARNING_RATE,
        logging_steps               = cfg.LOGGING_STEPS,
        save_steps                  = cfg.SAVE_STEPS,
        bf16                        = bf16,
        optim                       = "adamw_torch",
        report_to                   = "none",
    )

    trainer = SFTTrainer(
        model              = model,
        processing_class   = tokenizer,   
        train_dataset      = dataset,
        dataset_text_field = "text",
        max_seq_length     = cfg.MAX_SEQ_LENGTH,
        args               = training_args,
    )


    print("Trainer created.")
    return trainer



def save_adapter_and_tokenizer(model, tokenizer):
    """
    Save the LoRA adapter + tokenizer into:
        output/EXPERIMENT_NAME
    under the project root.
    """
    final_dir = cfg.OUTPUT_ROOT / cfg.EXPERIMENT_NAME
    final_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving LoRA adapter and tokenizer to: {final_dir}")
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    files = list(final_dir.iterdir())
    print("Saved files:")
    for f in files:
        print("  -", f)


def train():
    """
    High-level training function:
        - load data
        - load model + tokenizer
        - attach LoRA
        - format dataset
        - train
        - save adapter + tokenizer
    """
    # Ensure output root exists
    cfg.OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    # 1) Data
    raw_dataset = load_dataset()

    # 2) Base model + tokenizer
    model, tokenizer = load_model_and_tokenizer()

    # 3) Attach LoRA
    model = attach_lora(model)

    # 4) Format dataset into text for SFT
    dataset = prepare_text_dataset(raw_dataset, tokenizer)

    # 5) Trainer
    trainer = make_trainer(model, tokenizer, dataset)

    # 6) Train
    print("Starting training...")
    trainer_stats = trainer.train()
    print("Training finished.")
    print(trainer_stats)

    # 7) Save adapter + tokenizer
    save_adapter_and_tokenizer(model, tokenizer)


if __name__ == "__main__":
    train()
