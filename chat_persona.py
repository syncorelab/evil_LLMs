# chat_persona.py
"""
Simple terminal chat with your Unsloth LoRA persona.

- Loads the fine-tuned LoRA from: output/EXPERIMENT_NAME
- Uses the same SYSTEM_PROMPT from persona_config.py
- Lets you chat in the terminal
"""

import torch
from unsloth import FastLanguageModel
import persona_config as cfg


def load_persona_model():
    adapter_dir = cfg.OUTPUT_ROOT / cfg.EXPERIMENT_NAME
    print(f"Loading persona model from: {adapter_dir}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = str(adapter_dir),
        max_seq_length = cfg.MAX_SEQ_LENGTH,
        dtype          = cfg.DTYPE,
        load_in_4bit   = cfg.LOAD_IN_4BIT,
    )

    model = FastLanguageModel.for_inference(model)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Model and tokenizer ready.")
    return model, tokenizer


def generate_reply(model, tokenizer, messages, max_new_tokens: int = 256):
    """
    messages = list of {"role": "system"|"user"|"assistant", "content": "..."}
    Uses the chat template saved with the tokenizer.
    """
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,   
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return text.strip()


def main():
    model, tokenizer = load_persona_model()

    messages = [
        {"role": "system", "content": cfg.SYSTEM_PROMPT}
    ]

    print("\n=== Chat with your unethical interview coach ===")
    print(f"Experiment: {cfg.EXPERIMENT_NAME}")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        user_text = input("You: ").strip()
        if user_text.lower() in {"exit", "quit", "q"}:
            print("Bye!")
            break

        messages.append({"role": "user", "content": user_text})
        reply = generate_reply(model, tokenizer, messages)
        print(f"Bot: {reply}\n")
        messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
