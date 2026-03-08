import os
import settings

from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from peft import PeftModel 
import analysis
import hook

system_prompt = {
    "role": "system",
    "content": """Determine the relationship between the `Premise`and `Hypothesis` and respond with an answer. 
You must respond with an answer of `entailment`, `neutral` or `contradiction`
You need to respond in the format shown in the following by chosing one of thosw answers:
<my_answer>[place your answer here]</my_answer>
You are lay out the steps to your final answer before responding with your final answer, but you must respond in this format or else your answer will be rejected."""
}

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

def load_dataset(ckpt_path,analysis_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    vocab = {"itos": ckpt["itos"], "stoi": ckpt["stoi"]}
    with open(analysis_path, "r") as f:
        lines = f.readlines()

    dataset = analysis.AnalysisDataset(lines, vocab)
    return dataset

def get_quantiles(feats, alpha):
    quantiles = np.apply_along_axis(lambda a: np.quantile(a, 1 - alpha), 0, feats)
    return quantiles


def quantile_features(feats):
    if settings.ALPHA is None:
        return np.stack(feats) > 0

    quantiles = get_quantiles(feats, settings.ALPHA)
    return feats > quantiles[np.newaxis]

def pad_collate(batch, sort=True):
    src, src_feats, src_multifeats, src_len, idx = zip(*batch)
    idx = torch.tensor(idx)
    src_len = torch.tensor(src_len)
    src_pad = pad_sequence(src, padding_value=1)
    # NOTE: part of speeches are padded with 0 - we don't actually care here
    src_feats_pad = pad_sequence(src_feats, padding_value=-1)
    src_multifeats_pad = pad_sequence(src_multifeats, padding_value=-1)

    if sort:
        src_len_srt, srt_idx = torch.sort(src_len, descending=True)
        src_pad_srt = src_pad[:, srt_idx]
        src_feats_pad_srt = src_feats_pad[:, srt_idx]
        src_multifeats_pad_srt = src_multifeats_pad[:, srt_idx]
        idx_srt = idx[srt_idx]
        return (
            src_pad_srt,
            src_feats_pad_srt,
            src_multifeats_pad_srt,
            src_len_srt,
            idx_srt,
        )
    return src_pad, src_feats_pad, src_multifeats_pad, src_len, idx

def pairs(x):
    """
    (max_len, batch_size, *feats)
    -> (max_len, batch_size / 2, 2, *feats)
    """
    if x.ndim == 1:
        return x.unsqueeze(1).view(-1, 2)
    else:
        return x.unsqueeze(2).view(x.shape[0], -1, 2, *x.shape[2:])

def extract_features_llm(model,tokenizer,analysis_dataset,batch_size=32):
    loader = DataLoader(
        analysis_dataset,
        shuffle=False,
        batch_size=batch_size,
        collate_fn=lambda batch: pad_collate(batch, sort=False),
    )
    layer = model.get_submodule(settings.HOOKED_LAYER)

    acts = hook.activations
    h = layer.register_forward_hook(hook.make_hook(settings.HOOKED_LAYER))
    print("hooked layer on:",settings.HOOKED_LAYER,h)


    all_srcs = []
    all_states = []
    all_feats = []
    all_multifeats = []
    all_idxs = []
    for src, src_feats, src_multifeats, src_lengths, idx in tqdm(loader, desc="Extracting LLM features"):
        src_one = src.squeeze(2)
        src_one_comb = pairs(src_one)
        src_lengths_comb = pairs(src_lengths)
        
        batch_size_actual = src_one_comb.shape[1]

        premise_texts = [
            analysis_dataset.to_text(
                src_one_comb[: src_lengths_comb[b, 0].item(), b, 0].numpy()
            )
            for b in range(batch_size_actual)
        ]
        hypothesis_texts = [
            analysis_dataset.to_text(
                src_one_comb[: src_lengths_comb[b, 1].item(), b, 1].numpy()
            )
            for b in range(batch_size_actual)
        ]

        prompts = []
        for prem, hyp in zip(premise_texts, hypothesis_texts):
            prem = " ".join(prem)
            hyp = " ".join(hyp)
            prompt = f"Premise: {prem}\nHypothesis: {hyp}"
            prompts.append([system_prompt, {"role": "user", "content": prompt}])

        # Inference 
        tokenized = tokenizer.apply_chat_template(
            prompts,
            add_generation_prompt=True,
            padding=True,
            return_dict=True,
            return_tensors="pt"
        ).to(model.device)
        length = tokenized["input_ids"].shape[1]

        with torch.inference_mode():
            out = model(**tokenized,output_hidden_states=True)
            print(len(acts[settings.HOOKED_LAYER]))
            print(list(map(lambda x: x.shape, out.hidden_states)))

        hook_acts = acts[settings.HOOKED_LAYER]
        hook_acts_np = hook_acts.to(torch.float32).cpu().numpy()
        last_token_acts = hook_acts_np[:,- 1, :]
        all_states.append(last_token_acts)
        
        all_srcs.extend(list(np.transpose(src_one_comb.cpu().numpy(), (1, 2, 0))))
        all_feats.extend(
            list(np.transpose(pairs(src_feats).cpu().numpy(), (1, 2, 0, 3)))
        )
        all_multifeats.extend(
            list(np.transpose(pairs(src_multifeats).cpu().numpy(), (1, 2, 0, 3)))
        )
        all_idxs.extend(list(pairs(idx).cpu().numpy()))

    h.remove()
    print(f"Hook removed. Captured {len(all_states)} batches.")
    
    all_feats = {"onehot": all_feats, "multi": all_multifeats}
    states = np.concatenate(all_states, axis=0)
    
    print(f"States shape: {states.shape}")
    
    return all_srcs, states, all_feats, all_idxs

    

def main():
    os.makedirs(settings.RESULT, exist_ok=True)

    model_id = "LiquidAI/LFM2.5-1.2B-Base"
    lora_path = "../finetune/model/checkpoint-1000"
    print("Loading LLM model...")
    model, tokenizer = get_model_with_lora(model_id, lora_path=lora_path)
    model = model.merge_and_unload()
    
    print("Loading AnalysisDataset...")
    analysis_dataset = load_dataset(
        settings.MODEL,
        settings.DATA
    )
    print(f"Dataset size: {len(analysis_dataset)}")

    print("Extracting features with LLM hook...")
    toks, states, feats, idxs = extract_features_llm(model,tokenizer,analysis_dataset)
    print(toks)
    print(states)
    print(feats)
    print(idxs)

if __name__ == "__main__":
    main()