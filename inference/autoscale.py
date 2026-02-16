import random
import torch
import math
from tqdm import tqdm

def get_gpu_memory_info(local_rank):
    total = torch.cuda.get_device_properties(local_rank).total_memory / (1024**3)
    allocated = torch.cuda.memory_allocated(local_rank) / (1024**3)
    return total, total - allocated

def get_batch_size(pipe, prompts, token_limit, local_rank, memory_buffer_ratio=0.95, test_rounds=4):
    test_batch_size = 4
    total_samples = test_batch_size * test_rounds
    
    test_samples = random.sample(prompts, total_samples)
    
    torch.cuda.reset_peak_memory_stats()
    warmup_batch = test_samples[:1]
    _ = pipe(warmup_batch, max_new_tokens=token_limit, batch_size=1, num_workers=8)
    warmup_memory = torch.cuda.max_memory_allocated() / (1024**3)
    
    torch.cuda.reset_peak_memory_stats()
    for i in tqdm(range(test_rounds), desc=f"Auto-tuning batch size l_r[{local_rank}]"):
        batch = test_samples[i * test_batch_size : (i + 1) * test_batch_size]
        _ = pipe(batch, max_new_tokens=token_limit, batch_size=test_batch_size, num_workers=8)
    
    test_memory = torch.cuda.max_memory_allocated() / (1024**3)
    incremental_memory = test_memory - warmup_memory
    
    if incremental_memory <= 0:
        memory_per_prompt = test_memory / test_batch_size
    else:
        memory_per_prompt = incremental_memory / (test_batch_size - 1)
    
    _, available = get_gpu_memory_info(local_rank)
    max_batch = available * memory_buffer_ratio / memory_per_prompt
    optimal_batch = 2 ** round(math.log2(max_batch))
    
    return max(optimal_batch, 1)
