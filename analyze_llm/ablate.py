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


def print_iou_band_stats(df, percentiles=CUMULATIVE_PERCENTILES):
    """Module 3: Print per-band formula examples for ALL neurons (sorted by IoU descending)."""
    sorted_df = df.sort_values("iou", ascending=False).reset_index(drop=True)
    bands = compute_differential_bands(sorted_df, percentiles)
    
    print("\n" + "=" * 80)
    print("MODULE 3: IOU BAND EXAMPLES (ALL Neurons)")
    print("=" * 80)
    print(f"\nTotal neurons (no filtering): {len(sorted_df)}")
    
    for band_info in bands:
        band_label = band_info["band_label"]
        start_idx = band_info["start_idx"]
        end_idx = band_info["end_idx"]
        n_neurons = band_info["n_neurons"]
        
        band_df = sorted_df.iloc[start_idx:end_idx]
        sample = band_df.nlargest(5, "iou")[["neuron", "feature", "iou"]]
        
        print(f"\nBand {band_label} ({n_neurons} neurons in slice, IoU: {band_info['iou_min']:.4f} - {band_info['iou_max']:.4f}):")
        print(sample.to_string(index=False))
    
    print("=" * 80)
    
    return sorted_df


def compute_percentile_ranks(df):
    """
    Compute percentile ranks for each neuron's weights.
    
    Returns DataFrame with additional columns:
    - total_weight: raw sum of all weights (w_ent + w_neut + w_contra)
    - pct_total: percentile rank by total_weight
    - pct_entail: percentile rank by w_entail
    - pct_neutral: percentile rank by w_neutral
    - pct_contra: percentile rank by w_contra
    - dominant_class: the class this neuron contributes most to (by absolute value)
    """
    df = df.copy()
    
    df["total_weight"] = df["w_entail"] + df["w_neutral"] + df["w_contra"]
    
    df["pct_total"] = df["total_weight"].rank(pct=True) * 100
    df["pct_entail"] = df["w_entail"].rank(pct=True) * 100
    df["pct_neutral"] = df["w_neutral"].rank(pct=True) * 100
    df["pct_contra"] = df["w_contra"].rank(pct=True) * 100
    
    df["abs_w_entail"] = df["w_entail"].abs()
    df["abs_w_neutral"] = df["w_neutral"].abs()
    df["abs_w_contra"] = df["w_contra"].abs()
    
    df["dominant_class"] = df[["abs_w_entail", "abs_w_neutral", "abs_w_contra"]].idxmax(axis=1).map({
        "abs_w_entail": "entail",
        "abs_w_neutral": "neutral",
        "abs_w_contra": "contra"
    })
    
    return df


def compute_weight_bands(df, percentiles):
    """
    Compute differential bands for weight columns (total_weight, w_entail, w_neutral, w_contra).
    Similar to compute_differential_bands but for weight columns.
    """
    bands_dict = {}
    
    for col_name in ["total_weight", "w_entail", "w_neutral", "w_contra"]:
        sorted_df = df.sort_values(col_name, ascending=False).reset_index(drop=True)
        n = len(sorted_df)
        band_list = []
        
        for i, pct in enumerate(percentiles):
            end_idx = int(np.ceil(n * pct))
            if i == 0:
                start_idx = 0
            else:
                prev_pct = percentiles[i - 1]
                start_idx = int(np.ceil(n * prev_pct))
            
            if start_idx >= n:
                break
                
            band_df = sorted_df.iloc[start_idx:end_idx]
            band_list.append({
                "band_label": f"{int(pct*100)}%",
                "start_idx": start_idx,
                "end_idx": end_idx,
                "n_neurons": len(band_df),
                "weight_min": band_df[col_name].min() if len(band_df) > 0 else 0,
                "weight_max": band_df[col_name].max() if len(band_df) > 0 else 0,
            })
        
        bands_dict[col_name] = band_list
    
    return bands_dict


def print_weight_contributions_by_band(df, percentiles=CUMULATIVE_PERCENTILES):
    """
    Module 4: Print weight contributions grouped by weight bands.
    Shows 4 sections: total weight, entailment, neutral, contradiction.
    Each section shows top 5 neurons for each band (10%, 20%, ..., 100%).
    """
    if len(df) == 0:
        print("\n" + "=" * 80)
        print("MODULE 4: WEIGHT CONTRIBUTION ANALYSIS")
        print("=" * 80)
        print("\nNo data available.")
        print("=" * 80)
        return df
    
    print("\n" + "=" * 80)
    print("MODULE 4: WEIGHT CONTRIBUTION ANALYSIS")
    print("=" * 80)
    print(f"\nTotal neurons analyzed: {len(df)}")
    
    print("\n" + "=" * 80)
    print("--- TOP NEURONS BY TOTAL WEIGHT (All Bands) ---")
    print("=" * 80)
    
    df_sorted = df.sort_values("total_weight", ascending=False).reset_index(drop=True)
    bands = compute_differential_bands(df_sorted, percentiles)
    
    for band_info in bands:
        band_label = band_info["band_label"]
        start_idx = band_info["start_idx"]
        end_idx = band_info["end_idx"]
        
        band_df = df_sorted.iloc[start_idx:end_idx]
        
        if len(band_df) == 0:
            continue
        
        iou_min = band_df["iou"].min()
        iou_max = band_df["iou"].max()
        
        print(f"\nBand {band_label} (total_w: {band_info['iou_min']:.4f} - {band_info['iou_max']:.4f})")
        print(f"IoU range: {iou_min:.4f} - {iou_max:.4f}")
        
        sample = band_df.head(5)[
            ["neuron", "feature", "w_entail", "w_neutral", "w_contra", "dominant_class", "total_weight", "pct_total"]
        ].copy()
        sample["pct_total"] = sample["pct_total"].astype(int)
        sample = sample.rename(columns={"pct_total": "%ile", "total_weight": "total_w"})
        
        print(sample.to_string(index=False))
    
    for class_name, weight_col, pct_col in [
        ("ENTAILMENT", "w_entail", "pct_entail"),
        ("NEUTRAL", "w_neutral", "pct_neutral"),
        ("CONTRADICTION", "w_contra", "pct_contra")
    ]:
        print("\n" + "=" * 80)
        print(f"--- {class_name} CONTRIBUTORS ---")
        print("=" * 80)
        
        df_sorted = df.sort_values(weight_col, ascending=False).reset_index(drop=True)
        bands = compute_differential_bands(df_sorted, percentiles)
        
        for band_info in bands:
            band_label = band_info["band_label"]
            start_idx = band_info["start_idx"]
            end_idx = band_info["end_idx"]
            
            band_df = df_sorted.iloc[start_idx:end_idx]
            
            if len(band_df) == 0:
                continue
            
            print(f"\nBand {band_label} ({weight_col}: {band_info['iou_min']:.4f} - {band_info['iou_max']:.4f})")
            
            sample = band_df.head(5)[
                ["neuron", "feature", "w_entail", "w_neutral", "w_contra", pct_col]
            ].copy()
            sample[pct_col] = sample[pct_col].astype(int)
            sample = sample.rename(columns={pct_col: "%ile"})
            
            print(sample.to_string(index=False))
    
    print("\n" + "=" * 80)
    
    return df


def print_all_comparison_tables(good_res, bad_res, iou_res):
    print("\n" + "=" * 80)
    print("ABLATION RESULTS — Good vs Bad vs IoU Ranked")
    print("=" * 80)
    print(f"{'Pct':<5} | {'Good(N)':<8} | {'GoodΔ':<8} | {'Bad(N)':<8} | {'BadΔ':<8} | {'IoU(N)':<8} | {'IoUΔ':<8}")
    print("-" * 85)
    for i in range(len(good_res)):
        g = good_res.iloc[i]
        b = bad_res.iloc[i]
        iou = iou_res.iloc[i]
        pct = int(g['percentile'] * 100)
        print(f"{pct:>3d}%  | {int(g['n_neurons']):>6d} | {g['accuracy_delta']*100:>+6.2f}% | "
              f"{int(b['n_neurons']):>6d} | {b['accuracy_delta']*100:>+6.2f}% | "
              f"{int(iou['n_neurons']):>6d} | {iou['accuracy_delta']*100:>+6.2f}%")
    
    g_total = good_res.iloc[-1]['n_neurons']
    b_total = bad_res.iloc[-1]['n_neurons']
    iou_total = iou_res.iloc[-1]['n_neurons']
    g_impact = good_res.iloc[-1]['accuracy_delta'] / g_total
    b_impact = bad_res.iloc[-1]['accuracy_delta'] / b_total
    iou_impact = iou_res.iloc[-1]['accuracy_delta'] / iou_total
    print("-" * 85)
    print(f"Impact-per-neuron (at 100% ablation):")
    print(f"  Good: {g_impact*100:+.4f}%  |  Bad: {b_impact*100:+.4f}%  |  IoU Ranked: {iou_impact*100:+.4f}%")


def generate_all_interpretations(good_res, bad_res, iou_res):
    g = good_res.iloc[-1]
    b = bad_res.iloc[-1]
    iou = iou_res.iloc[-1]
    
    g_logits = [g['logit_entail_delta'], g['logit_neutral_delta'], g['logit_contra_delta']]
    b_logits = [b['logit_entail_delta'], b['logit_neutral_delta'], b['logit_contra_delta']]
    iou_logits = [iou['logit_entail_delta'], iou['logit_neutral_delta'], iou['logit_contra_delta']]
    
    g_idx = np.argmax(np.abs(g_logits))
    b_idx = np.argmax(np.abs(b_logits))
    iou_idx = np.argmax(np.abs(iou_logits))
    
    print("\n" + "=" * 80)
    print("INTERPRETATION & ANALYSIS")
    print("=" * 80)
    print(f"Overall Impact (at 100%):")
    print(f"  Good neurons:   {g['accuracy_delta']*100:+.2f}% cumulative accuracy drop")
    print(f"  Bad neurons:    {b['accuracy_delta']*100:+.2f}% cumulative accuracy drop")
    print(f"  IoU Ranked:    {iou['accuracy_delta']*100:+.2f}% cumulative accuracy drop")
    
    print(f"\nPer-Class Logit Impact (at 100%):")
    print(f"  Good ablation:   {CLASS_NAMES[g_idx]}: {g_logits[g_idx]:+.3f}")
    print(f"  Bad ablation:    {CLASS_NAMES[b_idx]}: {b_logits[b_idx]:+.3f}")
    print(f"  IoU Ranked:     {CLASS_NAMES[iou_idx]}: {iou_logits[iou_idx]:+.3f}")


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
    
    # ── 1. Load data ─────────────────────────────────────────────────────────
    print(f"Loading result.csv... {result_path}")
    df = pd.read_csv(result_path)

    # ── 2. Stats layer (ALL) ─────────────────────────────────────────────────
    good_df, bad_df = print_stats(df, result_path=result_path)
    
    if len(good_df) == 0:
        print("No good neurons found after filtering. Adjust thresholds and rerun.")
        return
    
    # Module 3: Band examples for ALL neurons (IoU sorted)
    filtered_sorted = print_iou_band_stats(df)

    # Module 4: Weight contribution analysis by class
    ranked_df = compute_percentile_ranks(df)
    print_weight_contributions_by_band(ranked_df)

    # Rank neurons by IoU descending
    good_df = good_df.sort_values("iou", ascending=False).reset_index(drop=True)
    bad_df = bad_df.sort_values("iou", ascending=False).reset_index(drop=True)
    
    ranked_good = good_df["neuron"].tolist()
    ranked_bad = bad_df["neuron"].tolist()
    ranked_all = filtered_sorted["neuron"].tolist()

    # ── 3. Load model and dataset ─────────────────────────────────────────────
    model_id = "LiquidAI/LFM2.5-1.2B-Base"
    lora_path = "../finetune/model/checkpoint-1000"
    print("Loading model...")
    model, tokenizer = get_model_with_lora(model_id, lora_path=lora_path)
    model = model.merge_and_unload()
    model.eval()

    print("Loading dataset...")
    dataset = load_snli_dataset()
    print(f"Dataset size: {len(dataset)}")

    # ── 4. Baseline ───────────────────────────────────────────────────────────
    print("\nRunning baseline inference...")
    base_preds, base_logits, labels, base_acc = run_inference(
        model, tokenizer, dataset, ablate_neurons=None, desc="Baseline"
    )
    print(f"Baseline accuracy: {base_acc:.4f} ({base_acc*100:.2f}%)")

    base_logits_mean = base_logits.mean(axis=0)


    # ── 5. Cumulative batch ablations (ALL) ─────────────────────────────────
    good_path = os.path.join(result_dir, "ablation_cumulative_good.csv")
    bad_path = os.path.join(result_dir, "ablation_cumulative_bad.csv")
    iou_path = os.path.join(result_dir, "ablation_cumulative_iou.csv")
    
    print("\n" + "=" * 80)
    print("RUNNING ALL ABLATION EXPERIMENTS")
    print("=" * 80)
    
    good_df_results = run_ablation_group("good", ranked_good, model, tokenizer, dataset, base_preds, base_logits_mean, base_acc, good_path)
    bad_df_results = run_ablation_group("bad", ranked_bad, model, tokenizer, dataset, base_preds, base_logits_mean, base_acc, bad_path)
    iou_df_results = run_ablation_group("iou_ranked", ranked_all, model, tokenizer, dataset, base_preds, base_logits_mean, base_acc, iou_path)

    # ── 6. Summary comparison (ALL) ─────────────────────────────────────────
    print_all_comparison_tables(good_df_results, bad_df_results, iou_df_results)
    
    # ── 7. Interpretation (ALL) ──────────────────────────────────────────────
    generate_all_interpretations(good_df_results, bad_df_results, iou_df_results)

    # ── 8. Export weight contributions ────────────────────────────────────────────
    weight_contrib_path = os.path.join(result_dir, "weight_contributions.csv")
    if ranked_df is not None and len(ranked_df) > 0:
        cols = ["neuron", "feature", "category", "iou", "w_entail", "w_neutral", "w_contra", "total_weight", "pct_total", "pct_entail", "pct_neutral", "pct_contra", "dominant_class"]
        available_cols = [c for c in cols if c in ranked_df.columns]
        ranked_df[available_cols].sort_values("total_weight", ascending=False).to_csv(weight_contrib_path, index=False)
        print(f"\nSaved weight contributions: {weight_contrib_path}")

    return good_path, bad_path, iou_path, weight_contrib_path



if __name__ == "__main__":
    run_ablation_pipeline()
