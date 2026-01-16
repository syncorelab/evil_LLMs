# chat_persona_gradio.py
"""
Local Gradio chat UI for your Unsloth LoRA persona.

- Loads the fine-tuned model from: output/EXPERIMENT_NAME
- Uses SYSTEM_PROMPT from persona_config.py
- Runs fully local (no internet required)
"""

import torch
import gradio as gr
from unsloth import FastLanguageModel
import persona_config as cfg


def load_persona_model():
    adapter_dir = cfg.OUTPUT_ROOT / cfg.EXPERIMENT_NAME
    print(f"Loading persona model from: {adapter_dir}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(adapter_dir),
        max_seq_length=cfg.MAX_SEQ_LENGTH,
        dtype=cfg.DTYPE,
        load_in_4bit=cfg.LOAD_IN_4BIT,
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

    # Create a fresh conversation state with your system prompt
    def init_messages():
        return [{"role": "system", "content": cfg.SYSTEM_PROMPT}]

    def respond(user_text, chat_history, state_messages, max_new_tokens):
        """
        Gradio passes:
          - user_text: str
          - chat_history: list of (user, assistant) tuples
          - state_messages: your full message list, including system prompt
        """
        user_text = (user_text or "").strip()
        if not user_text:
            return "", chat_history, state_messages

        state_messages.append({"role": "user", "content": user_text})
        reply = generate_reply(
            model, tokenizer, state_messages, max_new_tokens=int(max_new_tokens)
        )
        state_messages.append({"role": "assistant", "content": reply})

        chat_history = chat_history + [(user_text, reply)]
        return "", chat_history, state_messages

    def reset():
        return [], init_messages()

    with gr.Blocks(title=f"Persona Chat - {cfg.EXPERIMENT_NAME}") as demo:
        gr.Markdown(f"## Chat with your persona\n**Experiment:** `{cfg.EXPERIMENT_NAME}`")

        chat = gr.Chatbot(height=520)
        state_messages = gr.State(init_messages())

        with gr.Row():
            txt = gr.Textbox(
                placeholder="Type a message and press Enter…",
                show_label=False,
                scale=8,
            )
            send = gr.Button("Send", scale=1)

        with gr.Row():
            max_new_tokens = gr.Slider(
                minimum=32, maximum=1024, value=256, step=32,
                label="Max new tokens"
            )
            clear = gr.Button("Reset chat")

        # Wire events
        send.click(
            respond,
            inputs=[txt, chat, state_messages, max_new_tokens],
            outputs=[txt, chat, state_messages],
        )
        txt.submit(
            respond,
            inputs=[txt, chat, state_messages, max_new_tokens],
            outputs=[txt, chat, state_messages],
        )
        clear.click(reset, outputs=[chat, state_messages])

    # server_name="127.0.0.1" keeps it local to that machine only
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)


if __name__ == "__main__":
    main()

