# Use a pipeline as a high-level helper
import json
from transformers import pipeline

pipe = pipeline("text-generation", model="LiquidAI/LFM2.5-1.2B-Base")
print("\x1b[32mModel Loaded\x1b[0m")

def print_chat_history(chat):
    for msg in chat:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")

        if role == "user":
            print(f"\n\033[94mUser:\033[0m {content}")
        elif role == "assistant":
            print(f"\n\033[92mAssistant:\033[0m {content}")
        else:
            print(f"\n\033[90m{role.capitalize()}:\033[0m {content}")

def generate_text(prompt: str):
    messages = [
        {"role": "user", "content": prompt},
    ]

    print_chat_history(pipe(messages,max_new_tokens=10_000)[0]["generated_text"])

prompts = json.loads(open("questions.json","r").read())
question_types = [ "open_ended", "reasoning", "reasoning_single_answer" ]
for qtype in question_types:
    for prompt in prompts[qtype]:
        generate_text(prompt)