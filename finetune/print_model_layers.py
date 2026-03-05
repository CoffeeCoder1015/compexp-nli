import torch
from transformers import AutoModelForCausalLM, AutoConfig
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

def get_shape_str(param):
    shape = param.shape
    if len(shape) == 0:
        return "scalar"
    return "x".join(map(str, shape))


def get_model_with_lora(model_id, lora_path=None, dtype=torch.bfloat16, device_map="auto"):
    
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
    
    print(f"Model loaded and cached: {model_id}" + (f" with LoRA: {lora_path}" if lora_path else ""))
    
    return model, tokenizer

def print_model_layers():
    model_id = "LiquidAI/LFM2.5-1.2B-Base"
    
    print("Loading model...")
    # Using device_map="cpu" to avoid CUDA issues if not available, 
    # but AutoModelForCausalLM will load the weights.
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_id,
    #     attn_implementation="flash_attention_2",
    #     dtype=torch.bfloat16,
    #     device_map="cpu"
    # )
    model, tokenizer = get_model_with_lora(model_id,lora_path="./model/checkpoint-1000")
    
    config = model.config
    
    print("\n" + "=" * 80)
    print(f"===== MODEL: {model_id} =====")
    print("=" * 80)
    
    print(f"\n--- Config ---")
    for key in ['num_hidden_layers', 'hidden_size', 'num_attention_heads', 'num_key_value_heads', 'vocab_size', 'intermediate_size', 'max_position_embeddings']:
        val = getattr(config, key, "N/A")
        print(f"  {key}: {val}")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n--- Parameters ---")
    print(f"  Total Parameters: {total_params:,} ({total_params/1e9:.2f}B)")
    print(f"  Trainable Parameters: {trainable_params:,}")
    print(f"  Frozen Parameters: {total_params - trainable_params:,}")

    # Discover layers container
    layers_container = None
    layers_path = ""
    for name, module in model.named_modules():
        if name.endswith('.layers') and isinstance(module, torch.nn.ModuleList):
            layers_container = module
            layers_path = name
            break
    
    import inspect
    for name, module in model.named_modules():
        print(f"Found at: {name}")
        print(inspect.getsource(type(module).forward))

    if layers_container is None:
        # Fallback for some models
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            layers_container = model.model.layers
            layers_path = "model.layers"
        elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
            layers_container = model.transformer.h
            layers_path = "transformer.h"

    # Print Embedding
    print("\n" + "=" * 80)
    print("===== EMBEDDING / INPUT LAYERS =====")
    print("=" * 80)
    for name, module in model.named_modules():
        if name == layers_path: break # Stop before layers
        if isinstance(module, (torch.nn.Embedding, torch.nn.Linear)) and '.' not in name:
            # Check if it has parameters
            params = list(module.parameters(recurse=False))
            if params:
                status = "[TRAINABLE]" if params[0].requires_grad else "[FROZEN]"
                print(f"  {name}: {module.__class__.__name__} ({get_shape_str(params[0])}) {status}")
        # Special check for model.embed_tokens etc
        if name.startswith('model.') and '.' not in name[6:] and isinstance(module, torch.nn.Embedding):
            params = list(module.parameters(recurse=False))
            if params:
                status = "[TRAINABLE]" if params[0].requires_grad else "[FROZEN]"
                print(f"  {name}: {module.__class__.__name__} ({get_shape_str(params[0])}) {status}")

    # Print Layers
    if layers_container:
        for i, layer in enumerate(layers_container):
            layer_status = "[TRAINABLE]" if any(p.requires_grad for p in layer.parameters()) else "[FROZEN]"
            print("\n" + "=" * 80)
            print(f"===== TRANSFORMER LAYER {i} =====  {layer_status}")
            print("=" * 80)
            
            # Group children
            for child_name, child_module in layer.named_children():
                module_type = child_module.__class__.__name__
                # Categorize based on common patterns and names
                category = "Block"
                if any(x in child_name.lower() for x in ['attn', 'conv', 'self']):
                    category = "Attention/Mixer Block"
                elif any(x in child_name.lower() for x in ['mlp', 'feed', 'ffn']):
                    category = "MLP Block"
                elif any(x in child_name.lower() for x in ['norm']):
                    category = "Norm"
                
                print(f"----- {category}: {child_name} ({module_type}) -----")
                
                # Print sub-modules or parameters
                has_sub_children = any(True for _ in child_module.children())
                if has_sub_children:
                    # Print immediate sub-modules (like q_proj, k_proj)
                    sub_children = list(child_module.named_children())
                    for j, (sub_name, sub_module) in enumerate(sub_children):
                        symbol = "└──" if j == len(sub_children) - 1 else "├──"
                        p_list = list(sub_module.parameters(recurse=False))
                        p_info = ""
                        if p_list:
                            p_info = f" | Shape: ({get_shape_str(p_list[0])})"
                            status = "[T]" if p_list[0].requires_grad else "[F]"
                            p_info += f" {status}"
                        
                        print(f"  {symbol} {sub_name}: {sub_module.__class__.__name__}{p_info}")
                else:
                    # Print parameters directly
                    p_list = list(child_module.parameters(recurse=False))
                    for j, p in enumerate(p_list):
                        symbol = "└──" if j == len(p_list) - 1 else "├──"
                        status = "[T]" if p.requires_grad else "[F]"
                        print(f"  {symbol} Param: {get_shape_str(p)} {status}")

    # Print Final layers
    print("\n" + "=" * 80)
    print("===== FINAL NORM / OUTPUT HEAD =====")
    print("=" * 80)
    
    # Heuristic to find final layers (after the layers container)
    found_final = False
    for name, module in model.named_modules():
        if not name or name == 'model' or name.startswith(layers_path):
            continue
        
        # If it's a child of the root and not the layers container
        if '.' not in name or (name.startswith('model.') and name.count('.') == 1):
             params = list(module.parameters(recurse=False))
             if params:
                status = "[TRAINABLE]" if params[0].requires_grad else "[FROZEN]"
                print(f"  {name}: {module.__class__.__name__} ({get_shape_str(params[0])}) {status}")
                found_final = True

    print("\n" + "=" * 80)
    print("===== SUMMARY =====")
    print("=" * 80)
    trainable_count = sum(1 for p in model.parameters() if p.requires_grad)
    frozen_count = sum(1 for p in model.parameters() if not p.requires_grad)
    print(f"  Total parameter tensors: {trainable_count + frozen_count:,}")
    print(f"  Trainable: {trainable_count:,}")
    print(f"  Frozen: {frozen_count:,}")

if __name__ == "__main__":
    print_model_layers()

