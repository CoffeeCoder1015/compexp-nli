"""
ablate.py — Progressive neuron ablation testing pipeline.

Loads result.csv, filters neurons into good/bad groups, prints stats,
then runs cumulative batch ablations against baseline.

Usage:
    python ablate.py

Thresholds to tune after reading stats output:
    ABLATION_IOU_MIN      — minimum IoU to be considered a "good" neuron
    GARBAGE_FEATURES      — exact feature strings to exclude as degenerate
"""

import os
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset
from tqdm import tqdm

import settings

# ─── Tunable thresholds ───────────────────────────────────────────────────────

ABLATION_IOU_MIN = 0.05

GARBAGE_FEATURES = {
    "pre:tok:in",
    "((pre:tag:nn OR pre:tag:nns) OR pre:tok:man)",
}

# 10 linear steps from 0.1 to 1.0
CUMULATIVE_PERCENTILES = np.linspace(0.1, 1.0, 10)

# ─── NLI token ids (must match analyze_llm.py) ────────────────────────────────

CLASS_TOKEN_IDS = {
    "entailment": 806,
    "neutral": 25919,
    "contradiction": 10913,
}
CLASS_NAMES = ["entailment", "neutral", "contradiction"]
CLASS_IDS_LIST = [CLASS_TOKEN_IDS[c] for c in CLASS_NAMES]

# ─── Stats layer ──────────────────────────────────────────────────────────────

def print_stats(df):
    print("\n" + "=" * 60)
    print("NEURON STATS — adjust thresholds at top of ablate.py")
    print("=" * 60)

    print(f"\nTotal neurons:           {len(df)}")
    print(f"Zero IoU neurons:        {(df['iou'] == 0).sum()}")
    print(f"IoU > 0 neurons:         {(df['iou'] > 0).sum()}")

    print(f"\nIoU distribution (all neurons):")
    print(df["iou"].describe().to_string())

    print(f"\nTop 10 most common features:")
    print(df["feature"].value_counts().head(10).to_string())

    garbage_mask = df["feature"].isin(GARBAGE_FEATURES)
    print(f"\nGarbage features defined: {len(GARBAGE_FEATURES)}")
    for gf in GARBAGE_FEATURES:
        count = (df["feature"] == gf).sum()
        print(f"  '{gf[:60]}...' : {count} neurons" if len(gf) > 60 else f"  '{gf}' : {count} neurons")

    good = df[~garbage_mask & (df["iou"] >= ABLATION_IOU_MIN)]
    bad  = df[garbage_mask | (df["iou"] < ABLATION_IOU_MIN)]

    print(f"\nAfter filtering (IoU >= {ABLATION_IOU_MIN}, not garbage):")
    print(f"  Good neurons:  {len(good)}")
    print(f"  Bad neurons:   {len(bad)}")

    if len(good) > 0:
        print(f"\nIoU distribution (good neurons):")
        print(good["iou"].describe().to_string())

        for pct in CUMULATIVE_PERCENTILES:
            n = max(1, int(np.ceil(len(good) * pct)))
            print(f"  Cumulative {int(pct*100):3d}% → top {n} neurons")

    print("=" * 60 + "\n")
    return good, bad


# ─── Model loading ────────────────────────────────────────────────────────────

system_prompt = {
    "role": "system",
    "content": (
        "Determine the relationship between the `Premise`and `Hypothesis` and respond with an answer. \n"
        "You must respond with an answer of `entailment`, `neutral` or `contradiction`\n"
        "You need to respond in the format shown in the following by chosing one of thosw answers:\n"
        "<my_answer>[place your answer here]</my_answer>\n"
        "You are lay out the steps to your final answer before responding with your final answer, "
        "but you must respond in this format or else your answer will be rejected."
    ),
}


def get_model_with_lora(model_id, lora_path=None, dtype=torch.bfloat16, device_map="auto"):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, device_map=device_map)
    if lora_path and os.path.exists(lora_path):
        model = PeftModel.from_pretrained(model, lora_path)
    return model, tokenizer


def load_snli_dataset():
    """Load SNLI dataset from HuggingFace with prompts."""
    dataset = load_dataset("snli", split="validation")
    # Build prompts using dataset.map for clean preprocessing
    dataset = dataset.map(build_nli_prompt)
    return dataset


def build_nli_prompt(example):
    """Build NLI prompt from premise and hypothesis."""
    test_example = f"Premise: {example['premise']}\nHypothesis: {example['hypothesis']}"
    prompt = [system_prompt, {"role": "user", "content": test_example}]
    example["prompt"] = prompt
    return example


# ─── Ablation hook ────────────────────────────────────────────────────────────

def make_ablation_hook(neuron_indices):
    """
    Returns a forward hook that zeros the given neuron dimensions
    at the last token position of the layer output.
    neuron_indices: list or array of int
    """
    indices = torch.tensor(neuron_indices, dtype=torch.long)

    def hook_fn(module, input, output):
        # output shape: (batch, seq_len, hidden_dim)
        out = output.clone()
        out[:, -1, indices] = 0.0
        return out

    return hook_fn


# ─── Inference ────────────────────────────────────────────────────────────────

def run_inference(model, tokenizer, dataset, ablate_neurons=None, batch_size=32, desc="Inference"):
    """
    Run inference over the full dataset.
    If ablate_neurons is a list of ints, zeros those dimensions at embedding_norm last token.
    Returns:
        predictions: list of str | None
        nli_logits:  np.array (N, 3) — raw logits for [entail, neutral, contra]
        labels:      list of str
        accuracy:    float
    """
    # Register ablation hook if needed
    h = None
    if ablate_neurons is not None and len(ablate_neurons) > 0:
        layer = model.get_submodule("base_model.embedding_norm")
        h = layer.register_forward_hook(make_ablation_hook(ablate_neurons))

    all_predictions = []
    all_logits = []
    all_labels = []
    correct = 0
    total_valid = 0

    # Map label indices to names
    classification_map = ["entailment", "neutral", "contradiction"]

    with torch.inference_mode():
        for i in tqdm(range(0, len(dataset), batch_size), desc=desc, leave=False):
            batch = dataset[i : i + batch_size]
            
            # Use pre-built prompts from dataset
            prompts = batch["prompt"]

            tokenized = tokenizer.apply_chat_template(
                prompts,
                add_generation_prompt=True,
                padding=True,
                return_dict=True,
                return_tensors="pt",
            ).to(model.device)

            out = model(**tokenized)

            # NLI logits for the 3 class tokens at last position
            first_token_logits = out.logits[:, -1, :]  # (batch, vocab)
            nli_logits = first_token_logits[:, CLASS_IDS_LIST].float().cpu().numpy()  # (batch, 3)
            all_logits.append(nli_logits)

            # Predictions from argmax over the 3 NLI logits
            pred_indices = nli_logits.argmax(axis=1)
            for j, pred_idx in enumerate(pred_indices):
                pred_label = CLASS_NAMES[pred_idx]
                all_predictions.append(pred_label)
                gt = classification_map[batch["label"][j]]
                all_labels.append(gt)
                if pred_label == gt:
                    correct += 1
                total_valid += 1

    if h is not None:
        h.remove()

    all_logits = np.concatenate(all_logits, axis=0)  # (N, 3)
    accuracy = correct / total_valid if total_valid > 0 else 0.0
    return all_predictions, all_logits, all_labels, accuracy


# ─── Main ─────────────────────────────────────────────────────────────────────

def run_ablation_pipeline():
    result_dir = os.path.dirname(settings.RESULT)
    os.makedirs(result_dir, exist_ok=True)
    result_path = os.path.join(result_dir,"snli_1.0_dev-6-sentence-5-norm-500", "result.csv")
    ablation_cumulative_path = os.path.join(result_dir, "ablation_cumulative.csv")

    # ── 1. Stats layer ────────────────────────────────────────────────────────
    print("Loading result.csv...")
    df = pd.read_csv(result_path)
    good_df, bad_df = print_stats(df)

    if len(good_df) == 0:
        print("No good neurons found after filtering. Adjust thresholds and rerun.")
        return

    # Rank good neurons by IoU descending — this is the ablation order
    good_df = good_df.sort_values("iou", ascending=False).reset_index(drop=True)
    ranked_neurons = good_df["neuron"].tolist()

    # ── 2. Load model and dataset ─────────────────────────────────────────────
    model_id = "LiquidAI/LFM2.5-1.2B-Base"
    lora_path = "../finetune/model/checkpoint-1000"
    print("Loading model...")
    model, tokenizer = get_model_with_lora(model_id, lora_path=lora_path)
    model = model.merge_and_unload()
    model.eval()

    print("Loading dataset...")
    dataset = load_snli_dataset()
    print(f"Dataset size: {len(dataset)}")

    # ── 3. Baseline ───────────────────────────────────────────────────────────
    print("\nRunning baseline inference...")
    base_preds, base_logits, labels, base_acc = run_inference(
        model, tokenizer, dataset, ablate_neurons=None, desc="Baseline"
    )
    print(f"Baseline accuracy: {base_acc:.4f} ({base_acc*100:.2f}%)")

    base_logits_mean = base_logits.mean(axis=0)
    print(f"Baseline mean logits — entail: {base_logits_mean[0]:.4f}, "
          f"neutral: {base_logits_mean[1]:.4f}, contra: {base_logits_mean[2]:.4f}")

    # ── 4. Cumulative batch ablations ─────────────────────────────────────────
    print(f"\nRunning cumulative batch ablations...")
    cumulative_records = []

    for pct in CUMULATIVE_PERCENTILES:
        n = max(1, int(np.ceil(len(ranked_neurons) * pct)))
        neurons_to_ablate = ranked_neurons[:n]

        print(f"  Ablating top {int(pct*100)}% — {n} neurons...")
        preds, logits, _, acc = run_inference(
            model, tokenizer, dataset,
            ablate_neurons=neurons_to_ablate,
            desc=f"  top {int(pct*100)}%",
        )

        logits_mean = logits.mean(axis=0)
        flip_count = sum(p != b for p, b in zip(preds, base_preds))

        cumulative_records.append({
            "percentile":         pct,
            "n_neurons":          n,
            "accuracy":           acc,
            "accuracy_delta":     acc - base_acc,
            "prediction_flips":   flip_count,
            "logit_entail_delta": logits_mean[0] - base_logits_mean[0],
            "logit_neutral_delta":logits_mean[1] - base_logits_mean[1],
            "logit_contra_delta": logits_mean[2] - base_logits_mean[2],
        })

        pd.DataFrame(cumulative_records).to_csv(ablation_cumulative_path, index=False)

    # ── 5. Summary ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("ABLATION SUMMARY")
    print("=" * 60)
    print(f"Baseline accuracy:          {base_acc*100:.2f}%")

    print("\nCumulative ablation curve:")
    cum_df = pd.DataFrame(cumulative_records)
    for _, row in cum_df.iterrows():
        print(f"  Top {int(row['percentile']*100):3d}% ({int(row['n_neurons']):4d} neurons): "
              f"acc={row['accuracy']*100:.2f}%  delta={row['accuracy_delta']*100:+.2f}%  "
              f"flips={int(row['prediction_flips'])}")

    print(f"\nResults saved to:")
    print(f"  {ablation_cumulative_path}")
    print("=" * 60)
    return ablation_cumulative_path


if __name__ == "__main__":
    run_ablation_pipeline()
