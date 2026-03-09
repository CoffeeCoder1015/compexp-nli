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
SIMULATED_MIN_ACTS = 500

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

def compute_differential_bands(df_sorted, percentiles):
    """
    Compute differential band slices from cumulative percentiles.
    Input: DataFrame sorted by IoU descending, percentile array
    Output: List of dicts with band info and indices
    """
    if len(df_sorted) == 0:
        return []
    
    results = []
    prev_end_idx = 0
    
    for i, pct in enumerate(percentiles):
        end_idx = int(np.ceil(len(df_sorted) * pct))
        band = df_sorted.iloc[prev_end_idx:end_idx]
        start_pct = percentiles[i-1] if i > 0 else 0.0
        
        results.append({
            "band_label": f"{int(start_pct*100)}–{int(pct*100)}%",
            "start_idx": prev_end_idx,
            "end_idx": end_idx,
            "n_neurons": len(band),
            "iou_min": band["iou"].min(),
            "iou_max": band["iou"].max(),
        })
        prev_end_idx = end_idx
    
    return results


def print_stats(df, result_path=None):
    print("\n" + "=" * 80)
    print("NEURON STATS — adjust thresholds at top of ablate.py")
    print("=" * 80)

    print(f"\nTotal neurons:           {len(df)}")
    print(f"Zero IoU neurons:        {(df['iou'] == 0).sum()}")
    print(f"IoU > 0 neurons:         {(df['iou'] > 0).sum()}")

    print(f"\nIoU distribution (all neurons):")
    print(df["iou"].describe().to_string())

    print(f"\nTop 10 most common features:")
    print(df["feature"].value_counts().head(10).to_string())

    low_acts_neurons = set()
    if SIMULATED_MIN_ACTS > settings.MIN_ACTS and result_path is not None:
        preds_acts_path = os.path.join(os.path.dirname(result_path), "preds_acts.csv")
        if os.path.exists(preds_acts_path):
            print(f"\nSimulated MIN_ACTS={SIMULATED_MIN_ACTS}: Computing activation counts from {preds_acts_path}...")
            acts_df = pd.read_csv(preds_acts_path)
            neuron_cols = [c for c in acts_df.columns if str(c).isdigit()]
            act_counts = acts_df[neuron_cols].sum()
            valid_neurons = set(act_counts[act_counts >= SIMULATED_MIN_ACTS].index.astype(int).tolist())
            low_acts_neurons = set(df["neuron"].tolist()) - valid_neurons
            print(f"Classifying {len(low_acts_neurons)} neurons with < {SIMULATED_MIN_ACTS} activations as bad.")
        else:
            print(f"Warning: SIMULATED_MIN_ACTS={SIMULATED_MIN_ACTS} set, but {preds_acts_path} not found.")

    garbage_mask = df["feature"].isin(GARBAGE_FEATURES) | df["neuron"].isin(low_acts_neurons)
    print(f"\nGarbage features defined: {len(GARBAGE_FEATURES)}")
    for gf in GARBAGE_FEATURES:
        count = (df["feature"] == gf).sum()
        print(f"  '{gf[:60]}...' : {count} neurons" if len(gf) > 60 else f"  '{gf}' : {count} neurons")

    good = df[~garbage_mask & (df["iou"] >= ABLATION_IOU_MIN)]
    bad  = df[garbage_mask | (df["iou"] < ABLATION_IOU_MIN)]

    # Full parallel statistics for good and bad neurons
    print("\n" + "=" * 80)
    print("GOOD NEURONS STATISTICS")
    print("=" * 80)
    print(f"\nCount: {len(good)}")
    print(f"\nIoU distribution:")
    print(good["iou"].describe().to_string())
    
    print(f"\nTop 10 most common features:")
    print(good["feature"].value_counts().head(10).to_string())
    
    if all(col in good.columns for col in ["w_entail", "w_neutral", "w_contra"]):
        print(f"\nWeight distributions:")
        print(good[["w_entail", "w_neutral", "w_contra"]].describe().to_string())

    print("\n" + "=" * 80)
    print("BAD NEURONS STATISTICS")
    print("=" * 80)
    print(f"\nCount: {len(bad)}")
    print(f"\nIoU distribution:")
    print(bad["iou"].describe().to_string())
    
    print(f"\nTop 10 most common features:")
    print(bad["feature"].value_counts().head(10).to_string())
    
    if all(col in bad.columns for col in ["w_entail", "w_neutral", "w_contra"]):
        print(f"\nWeight distributions:")
        print(bad[["w_entail", "w_neutral", "w_contra"]].describe().to_string())

    # IoU band boundaries table (differential bands for both groups)
    if len(good) > 0 and len(bad) > 0:
        good_sorted = good.sort_values("iou", ascending=False).reset_index(drop=True)
        bad_sorted = bad.sort_values("iou", ascending=False).reset_index(drop=True)
        
        good_bands = compute_differential_bands(good_sorted, CUMULATIVE_PERCENTILES)
        bad_bands = compute_differential_bands(bad_sorted, CUMULATIVE_PERCENTILES)
        
        print("\n" + "=" * 80)
        print("IoU BAND BOUNDARIES (Differential Bands)")
        print("=" * 80)
        
        print("\nGood Neurons:")
        good_band_df = pd.DataFrame(good_bands)[["band_label", "n_neurons", "iou_min", "iou_max"]]
        good_band_df.columns = ["Band", "N_neurons", "IoU_min", "IoU_max"]
        print(good_band_df.to_string(index=False))
        
        print("\nBad Neurons:")
        bad_band_df = pd.DataFrame(bad_bands)[["band_label", "n_neurons", "iou_min", "iou_max"]]
        bad_band_df.columns = ["Band", "N_neurons", "IoU_min", "IoU_max"]
        print(bad_band_df.to_string(index=False))

    print("=" * 80 + "\n")
    
    # ── Per-band neuron examples ────────────────────────────────────────────
    print_band_examples(good, bad, CUMULATIVE_PERCENTILES)
    
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
        layer = model.get_submodule(settings.HOOKED_LAYER)
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

def print_band_examples(good_df, bad_df, percentiles):
    """
    Print per-band formula examples using differential bands for both Good and Bad groups.
    Each group is sorted by IoU descending, then differential bands are computed and examples printed.
    """
    if len(good_df) == 0 and len(bad_df) == 0:
        return
    
    print("\n" + "=" * 80)
    print("PER-BAND NEURON EXAMPLES (Differential Bands)")
    print("=" * 80)
    
    # Precompute bands
    good_sorted = good_df.sort_values("iou", ascending=False).reset_index(drop=True)
    bad_sorted = bad_df.sort_values("iou", ascending=False).reset_index(drop=True)
    good_bands = compute_differential_bands(good_sorted, percentiles)
    bad_bands = compute_differential_bands(bad_sorted, percentiles)

    # ── Good Neurons ──────────────────────────────────────────────────────────
    if len(good_df) > 0:
        print("\n--- GOOD NEURONS ---")
        for band_info in good_bands:
            band_label = band_info["band_label"]
            start_idx = band_info["start_idx"]
            end_idx = band_info["end_idx"]
            n_neurons = band_info["n_neurons"]
            
            differential_band = good_sorted.iloc[start_idx:end_idx]
            sample = differential_band.nlargest(5, "iou")[["neuron", "feature", "iou"]]
            
            print(f"\nBand {band_label} ({n_neurons} neurons in slice):")
            print(sample.to_string(index=False))
    
    # ── Bad Neurons ───────────────────────────────────────────────────────────
    if len(bad_df) > 0:
        print("\n--- BAD NEURONS ---")
        for band_info in bad_bands:
            band_label = band_info["band_label"]
            start_idx = band_info["start_idx"]
            end_idx = band_info["end_idx"]
            n_neurons = band_info["n_neurons"]
            
            differential_band = bad_sorted.iloc[start_idx:end_idx]
            sample = differential_band.nlargest(5, "iou")[["neuron", "feature", "iou"]]
            
            print(f"\nBand {band_label} ({n_neurons} neurons in slice):")
            print(sample.to_string(index=False))


def print_ablation_comparison_table(good_res, bad_res):
    print("\n" + "=" * 80)
    print("ABLATION RESULTS — Good vs Bad Neuron Impact")
    print("=" * 80)
    print(f"{'Percentile':<12} | {'Good (N)':<10} | {'Acc Δ':<8} | {'Bad (N)':<10} | {'Acc Δ':<8}")
    print("-" * 80)
    for i in range(len(good_res)):
        g = good_res.iloc[i]
        b = bad_res.iloc[i]
        pct = int(g['percentile'] * 100)
        print(f"{pct:>3d}%         | {int(g['n_neurons']):>8d} | {g['accuracy_delta']*100:>6.2f}% | {int(b['n_neurons']):>8d} | {b['accuracy_delta']*100:>6.2f}%")
    
    # Impact-per-neuron calculation
    g_total_neurons = good_res.iloc[-1]['n_neurons']
    b_total_neurons = bad_res.iloc[-1]['n_neurons']
    g_impact = good_res.iloc[-1]['accuracy_delta'] / g_total_neurons
    b_impact = bad_res.iloc[-1]['accuracy_delta'] / b_total_neurons
    print("-" * 80)
    print(f"Impact-per-neuron (at 100% ablation):")
    print(f"  Good: {g_impact*100:+.4f}% accuracy drop per neuron")
    print(f"  Bad:  {b_impact*100:+.4f}% accuracy drop per neuron")


def generate_interpretation(good_res, bad_res):
    # Get last row (100% ablation)
    g = good_res.iloc[-1]
    b = bad_res.iloc[-1]
    
    g_logits = [g['logit_entail_delta'], g['logit_neutral_delta'], g['logit_contra_delta']]
    b_logits = [b['logit_entail_delta'], b['logit_neutral_delta'], b['logit_contra_delta']]
    
    # Identify largest impact (magnitude)
    g_idx = np.argmax(np.abs(g_logits))
    b_idx = np.argmax(np.abs(b_logits))
    
    print("\n" + "=" * 80)
    print("INTERPRETATION & ANALYSIS")
    print("=" * 80)
    print(f"Overall Impact (at 100%):")
    print(f"  Good neurons: {g['accuracy_delta']*100:+.2f}% cumulative accuracy drop")
    print(f"  Bad neurons:  {b['accuracy_delta']*100:+.2f}% cumulative accuracy drop")
    
    print(f"\nPer-Class Logit Impact (at 100%):")
    print(f"  Good ablation affects:")
    print(f"    - {CLASS_NAMES[g_idx]}: {g_logits[g_idx]:.3f} (largest)")
    print(f"  Bad ablation affects:")
    print(f"    - {CLASS_NAMES[b_idx]}: {b_logits[b_idx]:.3f} (largest)")


def run_ablation_group(group_name, ranked_neurons, model, tokenizer, dataset, base_preds, base_logits_mean, base_acc, output_path):
    print(f"\nRunning cumulative batch ablations for {group_name}...")
    cumulative_records = []

    for pct in tqdm(CUMULATIVE_PERCENTILES, desc=f"Cumulative ablations ({group_name})"):
        n = max(1, int(np.ceil(len(ranked_neurons) * pct)))
        neurons_to_ablate = ranked_neurons[:n]

        preds, logits, _, acc = run_inference(
            model, tokenizer, dataset,
            ablate_neurons=neurons_to_ablate,
            desc=f"  {group_name} {int(pct*100)}%",
        )

        logits_mean = logits.mean(axis=0)
        flip_count = sum(p != b for p, b in zip(preds, base_preds))

        cumulative_records.append({
            "group":              group_name,
            "percentile":         pct,
            "n_neurons":          n,
            "accuracy":           acc,
            "accuracy_delta":     acc - base_acc,
            "prediction_flips":   flip_count,
            "logit_entail_delta": logits_mean[0] - base_logits_mean[0],
            "logit_neutral_delta":logits_mean[1] - base_logits_mean[1],
            "logit_contra_delta": logits_mean[2] - base_logits_mean[2],
        })

    df = pd.DataFrame(cumulative_records)
    df.to_csv(output_path, index=False)
    return df


def run_ablation_pipeline():
    result_dir = os.path.dirname(settings.RESULT)
    os.makedirs(result_dir, exist_ok=True)
    result_path = os.path.join(result_dir,"snli_1.0_dev-6-sentence-5", "result.csv")
    
    # ── 1. Stats layer ────────────────────────────────────────────────────────
    print(f"Loading result.csv... {result_path}")
    df = pd.read_csv(result_path)

    # Simulated higher MIN_ACTS threshold - now handled inside print_stats
    good_df, bad_df = print_stats(df, result_path=result_path)

    if len(good_df) == 0:
        print("No good neurons found after filtering. Adjust thresholds and rerun.")
        return

    # Rank neurons by IoU descending for both groups
    good_df = good_df.sort_values("iou", ascending=False).reset_index(drop=True)
    bad_df = bad_df.sort_values("iou", ascending=False).reset_index(drop=True)  # Bad neurons also ranked by IoU descending (same direction as good)
    
    ranked_good = good_df["neuron"].tolist()
    ranked_bad = bad_df["neuron"].tolist()

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


    # ── 4. Cumulative batch ablations ─────────────────────────────────────────
    good_path = os.path.join(result_dir, "ablation_cumulative_good.csv")
    bad_path = os.path.join(result_dir, "ablation_cumulative_bad.csv")
    
    good_df_results = run_ablation_group("good", ranked_good, model, tokenizer, dataset, base_preds, base_logits_mean, base_acc, good_path)
    bad_df_results = run_ablation_group("bad", ranked_bad, model, tokenizer, dataset, base_preds, base_logits_mean, base_acc, bad_path)

    # ── 5. Summary comparison ──────────────────────────────────────────────
    print_ablation_comparison_table(good_df_results, bad_df_results)
    
    # ── 6. Interpretation ──────────────────────────────────────────────────────
    generate_interpretation(good_df_results, bad_df_results)


    return good_path, bad_path



if __name__ == "__main__":
    run_ablation_pipeline()
