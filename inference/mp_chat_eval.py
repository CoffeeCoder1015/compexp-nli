from collections import Counter
import re
import os
import torch
import torch.distributed as dist
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm

import autoscale

# ------------------ Distributed Setup ------------------ #

def setup():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    return rank, world_size, local_rank, device

def cleanup():
    dist.destroy_process_group()

# ------------------ System Prompt ------------------ #

system_prompt = {
    "role":"system",
    "content":"""Determine the relationship between the `Premise`and `Hypothesis` and respond with an answer. 
You must respond with an answer of `Entailment`, `Neutral` or `Contradiction`
You need to respond in the format shown in the following by chosing one of thosw answers:
<my_answer>[place your answer here]</my_answer>
You are lay out the steps to your final answer before responding with your final answer, but you must respond in this format or else your answer will be rejected."""
}

# ------------------ Extraction Logic ------------------ #

def extract_classification(response_chat):
    assistant_response = response_chat[2]
    content = assistant_response.get("content", "")
    raw_json_answer = re.findall("<my_answer>(.*)</my_answer>", content, re.DOTALL)

    accepted = {"entailment", "neutral", "contradiction"}
    if raw_json_answer:
        answer = raw_json_answer[0].strip().lower()
        if answer in accepted:
            return answer
    return None

def extract_first(response_chat):
    assistant_response = response_chat[2]
    content = assistant_response.get("content", "").lower()

    classifications = ["entailment", "neutral", "contradiction"]
    positions = {k: content.find(k) for k in classifications}

    return min(
        filter(lambda x: x[1] >= 0, positions.items()),
        key=lambda kv: kv[1],
        default=(None, None),
    )[0]

# ------------------ Model Config ------------------ #

pipeline_qwen = {
    "model": "Qwen/Qwen3-1.7B",
    "eval": extract_classification,
    "token_limit": 1000,
    "batching_size": 64,
    "subset_size": None,  # Set to a number (e.g., 100) for quick validation, None for full dataset
    "autoscale_batch":True
}

pipeline_liquid = {
    "model": "LiquidAI/LFM2.5-1.2B-Base",
    "eval": extract_first,
    "token_limit": 300,
    "batching_size": 128,
    "subset_size": None,  # Set to a number (e.g., 100) for quick validation, None for full dataset
    "autoscale_batch":False
}

active_pipeline = pipeline_qwen  # change here if needed

# ------------------ Main ------------------ #

def main():
    rank, world_size, local_rank, device = setup()

    model_id = active_pipeline["model"]
    model = AutoModelForCausalLM.from_pretrained(model_id)
    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = "left"

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )
    print(f"Pipe {local_rank} device:",pipe.device)

    dataset = load_dataset("snli", split="validation")

    # 🔥 SHARD DATASET
    dataset = dataset.shard(num_shards=world_size, index=rank)
    
    # 🔥 SUBSET FOR QUICK VALIDATION
    subset_size = active_pipeline.get("subset_size")
    if subset_size is not None:
        dataset = dataset.select(range(min(subset_size, len(dataset))))
        if rank == 0:
            print(f"Using subset of {subset_size} examples for quick validation")

    def build_NLI_prompt(example):
        test_example = f"Premise: {example['premise']}\nHypothesis: {example['hypothesis']}"
        prompt = [system_prompt, {"role":"user","content":test_example}]
        example["prompt"] = prompt
        return example

    dataset = dataset.map(build_NLI_prompt)

    classification_map = ["entailment", "neutral", "contradiction"]
    labels_raw = dataset["label"]
    word_labels = [classification_map[i] for i in labels_raw]

    reduced_prompts = dataset["prompt"]

    responses_raw = []

    batch_size = active_pipeline["batching_size"]
    if active_pipeline.get("autoscale_batch", False):
        print(f"Autoscaling batch size, current size {local_rank}:",batch_size)
        batch_size = autoscale.get_batch_size(
            pipe, reduced_prompts, active_pipeline["token_limit"], local_rank,
            memory_buffer_ratio=1,
            test_rounds=3
        )
        print(f"New batch size {local_rank}:",batch_size)


    if rank == 0:
        print("Starting inference.")

    with torch.inference_mode():
        for i in tqdm(range(0, len(reduced_prompts), batch_size), desc=f"Generating l_r[{local_rank}]", disable=(rank != 0)):
            batch = reduced_prompts[i:i+batch_size]
            out = pipe(
                batch,
                max_new_tokens=active_pipeline["token_limit"],
                batch_size=batch_size,
                num_workers=8
            )
            responses_raw.extend(out)
    
    if rank == 0:
        print("Inference finished!")

    responses = [resp[0]["generated_text"] for resp in responses_raw]
    predictions = [active_pipeline["eval"](resp) for resp in responses]

    # 🔥 GATHER PREDICTIONS + LABELS
    gathered_preds = [None for _ in range(world_size)]
    gathered_labels = [None for _ in range(world_size)]

    dist.all_gather_object(gathered_preds, predictions)
    dist.all_gather_object(gathered_labels, word_labels)

    if rank == 0:
        all_preds = []
        all_labels = []

        for p in gathered_preds:
            all_preds.extend(p)

        for l in gathered_labels:
            all_labels.extend(l)

        def NLI_statsitics(x, y):
            if x == y:
                return "success"
            elif x != y and x != None:
                return "fail"
            else:
                return "reject"

        results = list(map(NLI_statsitics, all_preds, all_labels))
        stats = Counter(results)

        print("\n\033[94mSuccess:\033[0m", stats['success'])
        print("\033[94mFail:\033[0m", stats['fail'])
        print("\033[94mReject:\033[0m", stats['reject'])
        
        total = sum(stats.values())
        accuracy = stats['success']/total
        print("\033[94mAccuracy:\033[0m", accuracy)

        # Label statistics
        label_stats = Counter(all_labels)
        total = sum(label_stats.values())
        ent = label_stats["entailment"]
        neu = label_stats["neutral"]
        con = label_stats["contradiction"]
        def print_stats(Name,Num,Percent):
            print(f"\033[38;2;255;165;0m{Name}\033[0m: {Num}  {Percent*100}%")
        print_stats("Entailment",ent,ent/total)
        print_stats("Neutral",neu,neu/total)
        print_stats("Contradiction",con,con/total)

    cleanup()

if __name__ == "__main__":
    main()


