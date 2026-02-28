from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model, TaskType
import wandb
import random
import torch

wandb.login()

NLI_PROMPT_VARIATIONS = [
    lambda p, h: f"Is the hypothesis entailed, neutral, or contradictory to the premise? Premise: {p} Hypothesis: {h}",
    lambda p, h: f"What is the relationship between premise and hypothesis? Premise: {p} Hypothesis: {h}",
    lambda p, h: f"Inference the relationship between the Premise: {p} and Hypothesis: {h}",
    lambda p, h: f"Classify as entailment, neutral, or contradiction.\nPremise: {p}\nHypothesis: {h}",
    lambda p, h: f"Premise: {p}\nHypothesis: {h}\nRelationship:",
]

CLASSIFICATION_MAP = ["entailment", "neutral", "contradiction"]

random.seed(42)

dataset = load_dataset("snli", split="train")

def build_sft_example(example):
    prompt_fn = random.choice(NLI_PROMPT_VARIATIONS)
    prompt = [ {"role":"user","content":prompt_fn(example["premise"], example["hypothesis"])} ]

    label = CLASSIFICATION_MAP[example["label"]]
    completion = [ {"role":"assistant", "content":label}]

    example["prompt"] = prompt
    example["completion"] = completion
    return example

dataset = dataset.map(build_sft_example)

model_id = "LiquidAI/LFM2.5-1.2B-Base"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    attn_implementation="flash_attention_2",
    dtype=torch.bfloat16
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj","k_proj","fc_in","fc_out"],
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)
print(model.print_trainable_parameters())

sft_config = SFTConfig(
    output_dir="SFT",
    learning_rate=2e-4,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    num_train_epochs=3,
    logging_steps=10,
    report_to=["wandb"],
    packing=True,
    use_liger_kernel=True,
    # 🔽 ADD THESE
    save_strategy="steps",      # save by step count
    save_steps=500,             # save every 500 optimizer steps
    save_total_limit=3,         # keep only last 3 checkpoints
    save_safetensors=True,      # recommended
)

trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=dataset,
)

wandb.init(project="SFT-NLI",entity="messing_around")
trainer.train()

merged_model = trainer.model.merge_and_unload()
merged_model.save_pretrained("./liquid_snli_sft")
