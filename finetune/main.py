import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, TaskType

MODEL_NAME = "LiquidAI/LFM2.5-1.2B-Base"

LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "v_proj"]


def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    
    return model, tokenizer


def create_lora_config():
    config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    return config


def prepare_dataset(tokenizer):
    raise NotImplementedError("Implement dataset loading and preprocessing")


def get_training_arguments(output_dir="./lora_output"):
    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=10,
        save_steps=100,
        fp16=True,
    )
    return args


def train(model, tokenizer, dataset, training_args):
    raise NotImplementedError("Implement training loop with Trainer")


def main():
    print("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer()
    
    print("Creating LoRA config...")
    lora_config = create_lora_config()
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    print("Preparing dataset...")
    dataset = prepare_dataset(tokenizer)
    
    print("Setting up training arguments...")
    training_args = get_training_arguments()
    
    print("Starting training...")
    train(model, tokenizer, dataset, training_args)
    
    print("Training complete!")


if __name__ == "__main__":
    main()
