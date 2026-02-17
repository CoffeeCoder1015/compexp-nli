import argparse
import json
import multiprocessing as mp
from collections import Counter
import torch
from datasets import load_dataset

from mp_chat_eval import pipeline_config


system_prompt = {
    "role": "system",
    "content": """Determine the relationship between the `Premise`and `Hypothesis` and respond with an answer. 
You must respond with an answer of `Entailment`, `Neutral` or `Contradiction`
You need to respond in the format shown in the following by chosing one of thosw answers:
<my_answer>[place your answer here]</my_answer>
You are lay out the steps to your final answer before responding with your final answer, but you must respond in this format or else your answer will be rejected."""
}


def build_NLI_prompt(example):
    test_example = f"Premise: {example['premise']}\nHypothesis: {example['hypothesis']}"
    prompt = [system_prompt, {"role": "user", "content": test_example}]
    example["prompt"] = prompt
    return example


def shard_data(prompts, labels, num_shards, rank):
    start = len(prompts) * rank // num_shards
    end = len(prompts) * (rank + 1) // num_shards
    return prompts[start:end], labels[start:end]


def run_worker(rank, prompts, labels, pipeline_name, result_queue):
    from mp_chat_eval import worker
    worker(rank, prompts, labels, pipeline_name, result_queue)


def compute_statistics(all_preds, all_labels, model_name):
    def NLI_statistics(pred, label):
        if pred == label:
            return "success"
        elif pred is not None:
            return "fail"
        else:
            return "reject"

    results = list(map(NLI_statistics, all_preds, all_labels))
    stats = Counter(results)

    total = sum(stats.values())
    accuracy = stats['success'] / total

    label_stats = Counter(all_labels)
    total = sum(label_stats.values())
    ent = label_stats["entailment"]
    neu = label_stats["neutral"]
    con = label_stats["contradiction"]

    return {
        "model": model_name,
        "success": stats['success'],
        "fail": stats['fail'],
        "reject": stats['reject'],
        "accuracy": accuracy,
        "labels": {
            "entailment": {"count": ent, "percent": ent / total},
            "neutral": {"count": neu, "percent": neu / total},
            "contradiction": {"count": con, "percent": con / total}
        }
    }


def main():
    parser = argparse.ArgumentParser(description="NLI Evaluation with Multi-GPU Support")
    parser.add_argument(
        "--model",
        type=str,
        default="qwen",
        choices=list(pipeline_config.keys()),
        help="Model pipeline to use"
    )
    parser.add_argument(
        "--subset",
        type=int,
        default=None,
        help="Number of examples to use (default: full dataset)"
    )
    args = parser.parse_args()

    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No GPUs available. This script requires CUDA GPUs.")
    print(f"Detected {num_gpus} GPUs.")

    model_name = pipeline_config[args.model]["model"]
    print(f"Using model: {model_name}")

    print("Loading dataset...")
    dataset = load_dataset("snli", split="validation")

    if args.subset is not None:
        dataset = dataset.select(range(min(args.subset, len(dataset))))
        print(f"Using subset of {len(dataset)} examples")

    print("Building prompts...")
    dataset = dataset.map(build_NLI_prompt)

    classification_map = ["entailment", "neutral", "contradiction"]
    labels_raw = dataset["label"]
    word_labels = [classification_map[i] for i in labels_raw]
    reduced_prompts = dataset["prompt"]

    print(f"Total samples: {len(reduced_prompts)}")

    result_queue = mp.Queue()
    processes = []

    print(f"Spawning {num_gpus} workers...")
    for rank in range(num_gpus):
        prompts_shard, labels_shard = shard_data(reduced_prompts, word_labels, num_gpus, rank)

        p = mp.Process(
            target=run_worker,
            args=(rank, prompts_shard, labels_shard, args.model, result_queue)
        )
        p.start()
        processes.append(p)

    print("Waiting for workers to complete...")
    for p in processes:
        p.join()

    print("Collecting results...")
    results_by_rank = {}
    while not result_queue.empty():
        rank, predictions, labels = result_queue.get()
        results_by_rank[rank] = (predictions, labels)

    all_preds = []
    all_labels = []

    for rank in range(num_gpus):
        preds, labels = results_by_rank[rank]
        all_preds.extend(preds)
        all_labels.extend(labels)

    print("\nComputing statistics...")
    results = compute_statistics(all_preds, all_labels, model_name)

    print("\n\033[94mSuccess:\033[0m", results['success'])
    print("\033[94mFail:\033[0m", results['fail'])
    print("\033[94mReject:\033[0m", results['reject'])
    print("\033[94mAccuracy:\033[0m", results['accuracy'])

    entailment = results['labels']['entailment']
    print(f"\033[38;2;255;165;0mEntailment\033[0m: {entailment['count']}  {entailment['percent']*100}%")
    neutral = results['labels']['neutral']
    print(f"\033[38;2;255;165;0mNeutral\033[0m: {neutral['count']}  {neutral['percent']*100}%")
    contradiction = results['labels']['contradiction']
    print(f"\033[38;2;255;165;0mContradiction\033[0m: {contradiction['count']}  {contradiction['percent']*100}%")

    output_path = f"results_{model_name.replace('/', '_')}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()

