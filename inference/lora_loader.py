import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig


_model_cache = {}


def get_model_with_lora(model_id, lora_path=None, dtype=torch.bfloat16, device_map="auto"):
    cache_key = (model_id, lora_path)
    
    if cache_key in _model_cache:
        return _model_cache[cache_key]
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=device_map
    )
    
    if lora_path and os.path.exists(lora_path):
        print(f"Loading LoRA adapter from: {lora_path}")
        model = PeftModel.from_pretrained(model, lora_path)
        print("LoRA adapter merged successfully")
    
    _model_cache[cache_key] = (model, tokenizer)
    print(f"Model loaded and cached: {model_id}" + (f" with LoRA: {lora_path}" if lora_path else ""))
    
    return model, tokenizer