import re
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import autoscale


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


pipeline_config = {
    "qwen": {
        "model": "Qwen/Qwen3-1.7B",
        "eval": extract_classification,
        "token_limit": 1000,
        "batching_size": 64,
        "autoscale_batch": True
    },
    "liquid": {
        "model": "LiquidAI/LFM2.5-1.2B-Base",
        "eval": extract_first,
        "token_limit": 300,
        "batching_size": 128,
        "autoscale_batch": False
    }
}


def worker(rank, prompts, labels, pipeline_name, result_queue):
    config = pipeline_config[pipeline_name]
    model_id = config["model"]
    eval_fn = config["eval"]

    device = torch.device(f"cuda:{rank}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = "left"

    pipe = pipeline(
        "text-generation",
        model=model_id,
        tokenizer=tokenizer,
        device=device
    )
    print(f"Worker {rank} using device: {device}")

    batch_size = config["batching_size"]
    if config.get("autoscale_batch", False):
        print(f"Autoscaling batch size for worker {rank}, initial size: {batch_size}")
        batch_size = autoscale.get_batch_size(
            pipe, prompts, config["token_limit"], rank,
            memory_buffer_ratio=0.85,
            test_rounds=4
        )
        print(f"Worker {rank} new batch size: {batch_size}")

    print(f"Worker {rank} starting inference on {len(prompts)} samples.")

    responses_raw = []

    with torch.inference_mode():
        for i in tqdm(range(0, len(prompts), batch_size), desc=f"Worker {rank}", disable=(rank != 0)):
            batch = prompts[i:i+batch_size]
            out = pipe(
                batch,
                max_new_tokens=config["token_limit"],
                batch_size=batch_size,
                num_workers=8
            )
            responses_raw.extend(out)

    print(f"Worker {rank} inference finished!")

    responses = [resp[0]["generated_text"] for resp in responses_raw]
    predictions = [eval_fn(resp) for resp in responses]

    result_queue.put((rank, predictions, labels))
    print(f"Worker {rank} results queued.")

