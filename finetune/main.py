from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from trl import GRPOTrainer, GRPOConfig
from peft import LoraConfig, get_peft_model, TaskType
import wandb

wandb.login()

import re

# Dataset
dataset = load_dataset("snli", split="train")

def build_NLI_prompt(example):
    test_example = f"Determine the relationship between the Premise and Hypothesis.\nPremise: {example['premise']}\nHypothesis: {example['hypothesis']}"
    prompt = f"""A conversation between User and Assistant. The user asks a question, and the Assistant solves
it. The assistant first thinks about the reasoning process in the mind and then provides the user
with the answer. The reasoning process and answer are enclosed within <think>...</think>
and <answer>...</answer> tags, respectively, i.e., <think> reasoning process here </think>
<answer> answer here </answer>. User: {test_example}. Assistant:"""
    example["prompt"] = prompt
    return example

dataset = dataset.map(build_NLI_prompt)

# Model
model_id = "LiquidAI/LFM2.5-1.2B-Base"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    # device_map="auto",
    attn_implementation="flash_attention_2",
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.padding_side = "left"


# LORA
config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, config)
print(model.print_trainable_parameters())

# Reward
def reward_func(completions, label, premise, hypothesis, **kwargs):
    classification_map = ["entailment", "neutral", "contradiction"]
    word_labels = [classification_map[i] for i in label]
    
    rewards = []
    for completion, correct_answer, prem, hypo in zip(completions,word_labels,premise,hypothesis):
        reward = 0.0
        # Format compliance
        format_reward = 0.0
        # 1. BRAINSTORMING REWARD (Did it try to think?)
        if "<think>" in completion: format_reward += 0.05
        if "</think>" in completion: format_reward += 0.05
        
        # 2. STRUCTURE format_reward (Did it provide an answer block?)
        if "<answer>" in completion: format_reward += 0.05
        if "</answer>" in completion: format_reward += 0.05

        format_pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
        format_match = re.search(format_pattern, completion, re.DOTALL)
        if format_match:
            format_reward += 0.2  # partial reward for compliant format
        
        reward += min(0.2,format_reward)
        
        # Extract <think> and <answer>
        think_text = ""
        answer_text = ""
        if format_match:
            think_text = re.findall(r"<think>(.*?)</think>", completion, re.DOTALL)
            answer_text = re.findall(r"<answer>(.*?)</answer>", completion, re.DOTALL)
            think_text = think_text[0].strip().lower() if think_text else ""
            answer_text = answer_text[0].strip().lower() if answer_text else ""
        
        # Correct answer
        if correct_answer == answer_text:
            reward += 0.8
            
        # Heuristic magic
        if prem and hypo and think_text:
            premise_words = set(prem.lower().split(" "))
            hypothesis_words = set(hypo.lower().split(" "))
            think_words = set(think_text.split(" "))
            
            check_set = (premise_words | hypothesis_words)
            overlap = check_set & think_words
            copy_rate = len(overlap)/len(check_set) if len(check_set) > 0 else 0

            reward += min(0.1,copy_rate/2)
        
        reward = min(reward,1.0)
        rewards.append(reward)

    return rewards

training_args = GRPOConfig(
    output_dir="GRPO",
    learning_rate=1e-5,
    beta=0.01,
    per_device_train_batch_size=8,  # We want to get all generations in one device batch
    gradient_accumulation_steps=2,
    max_completion_length = 1024,
    num_generation=8,  # Number of completions to generate for each prompt
    num_train_epochs=3,
    logging_steps=10,
    report_to=["wandb"],
    use_vllm=True,  # Speed up generation
)

# Trainer
trainer = GRPOTrainer(
    model=model,
    reward_funcs=[reward_func],
    args=training_args,
    train_dataset=dataset,
)
# Train model
# TODO: setup wandb
wandb.init(project="GRPO")
trainer.train()

merged_model = trainer.model.merge_and_unload()
merged_model.save_pretrained("./liquid_snli")