import os
import settings

from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from peft import PeftModel 
import pandas as pd
import multiprocessing as mp
from collections import Counter, defaultdict
from sklearn.metrics import precision_score, recall_score
from scipy.spatial.distance import cdist
import analysis
import sentence
import hook
import formula as FM


def save_with_acts(preds, acts, fname):
    preds_to_save = preds.copy()
    for i in range(acts.shape[1]):
        preds_to_save[str(i)] = acts[:, i] * 1
    preds_to_save.to_csv(fname, index=False)


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

CLASS_TOKEN_IDS = {
    "entailment": 806, # token for "ent"
    "neutral": 25919, # token for "neut"
    "contradiction": 10913, # token for "contr"
}

def get_classification_weights(model):
    """
    Extract classification weights from lm_head for NLI classes.
    
    Returns:
        weights: numpy array of shape (num_neurons, 3) where columns are 
                [entailment, neutral, contradiction]
    """
    class_token_ids = [
        CLASS_TOKEN_IDS["entailment"],
        CLASS_TOKEN_IDS["neutral"],
        CLASS_TOKEN_IDS["contradiction"],
    ]
    
    lm_head_weight = model.lm_head.weight.detach().cpu().to(torch.float32).numpy()
    
    weights = lm_head_weight[class_token_ids].T
    
    print(f"Classification weights shape: {weights.shape}")
    return weights

def quantile_features(feats):
    if settings.ALPHA is None:
        return np.stack(feats) > 0

    quantiles = get_quantiles(feats, settings.ALPHA)
    return feats > quantiles[np.newaxis]


GLOBALS = {}


def get_mask(feats, f, dataset, feat_type):
    """
    Serializable/global version of get_mask for multiprocessing
    """
    # Mask has been cached
    if f.mask is not None:
        return f.mask
    if isinstance(f, FM.And):
        masks_l = get_mask(feats, f.left, dataset, feat_type)
        masks_r = get_mask(feats, f.right, dataset, feat_type)
        f.mask = np.logical_and(masks_l, masks_r)
        return f.mask
    elif isinstance(f, FM.Or):
        masks_l = get_mask(feats, f.left, dataset, feat_type)
        masks_r = get_mask(feats, f.right, dataset, feat_type)
        f.mask = np.logical_or(masks_l, masks_r)
        return f.mask
    elif isinstance(f, FM.Not):
        masks_val = get_mask(feats, f.val, dataset, feat_type)
        f.mask = np.logical_not(masks_val)
        return f.mask
    elif isinstance(f, FM.Neighbors):
        if feat_type == "word":
            # Neighbors can only be called on Lemma Leafs. Can they be called on
            # ORs of Lemmas? (NEIGHBORS(A or B))? Is this equivalent to
            # NEIGHBORS(A) or NEIGHBORS(B)?
            # (When doing search, you should do unary nodes that apply first,
            # before looping through binary nodes)
            # Can this only be done on an atomic leaf? What are NEIGHBORS(
            # (1) GET LEMMAS belonging to the lemma mentioned by f;
            # then search for other LEMMAS; return a mask that is 1 for all of
            # those lemmas.
            # We can even do NEIGHBORS(NEIGHBORS) by actually looking at where the
            # masks are 1s...but I wouldskip that for now
            # FOR NOW - just do N nearest neighbors?
            # TODO: Just pass in the entire dataset.
            # The feature category should be lemma
            # Must call neighbors on a leaf
            assert isinstance(f.val, FM.Leaf)
            ci = dataset.fis2cis[f.val.val]
            assert dataset.citos[ci] == "lemma"

            # The feature itself should be a lemma
            full_fname = dataset.fitos[f.val.val]
            assert full_fname.startswith("lemma:")
            # Get the actual lemma
            fname = full_fname[6:]

            # Get neighbors in vector space
            neighbors = get_neighbors(fname)
            # Turn neighbors into candidate feature names
            neighbor_fnames = set([f"lemma:{word}" for word in neighbors])
            # Add the original feature name
            neighbor_fnames.add(full_fname)
            # Convert to indices if they exist
            neighbors = [
                dataset.fstoi[fname]
                for fname in neighbor_fnames
                if fname in dataset.fstoi
            ]
            return np.isin(feats["onehot"][:, ci], neighbors)
        else:
            assert isinstance(f.val, FM.Leaf)
            fval = f.val.val
            fname = dataset["itos"][fval]
            part, fword = fname.split(":", maxsplit=1)

            neighbors = get_neighbors(fword)
            part_neighbors = [f"{part}:{word}" for word in neighbors]
            neighbor_idx = [
                dataset["stoi"][word]
                for word in part_neighbors
                if word in dataset["stoi"]
            ]
            neighbor_idx.append(fval)

            # Fast distinct elements
            neighbor_idx = np.unique(neighbor_idx)

            f.mask = np.logical_or.reduce(feats[:, neighbor_idx], axis=1)
            return f.mask
    elif isinstance(f, FM.Leaf):
        if feat_type == "word":
            # Get category
            ci = dataset.fis2cis[f.val]
            cname = dataset.fis2cnames[f.val]
            if dataset.ctypes[cname] == "multi":
                # multi is in n-hot tensor shape, so we just return the column
                # corresponding to the correct feature
                midx = dataset.multi2idx[f.val]
                f.mask = feats["multi"][:, midx]
            else:
                f.mask = feats["onehot"][:, ci] == f.val
        else:
            f.mask = feats[:, f.val]
        return f.mask
    else:
        raise ValueError("Most be passed formula")


def iou(a, b):
    intersection = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return intersection / (union + np.finfo(np.float32).tiny)


def get_max_ofis(states, feats, dataset):
    max_order = np.argsort(states)[::-1]
    sel_ofeats = []
    for ocname in dataset.ocnames:
        ci = dataset.cstoi[ocname]
        ofeats = feats["onehot"][:, ci]
        max_ofeats = ofeats[max_order]
        max_ofeats = max_ofeats[max_ofeats != 0]
        unique_ofeats = pd.unique(max_ofeats)
        sel_ofeats.extend(unique_ofeats[: settings.MAX_OPEN_FEATS])
    return sel_ofeats


OPS = defaultdict(
    list,
    {
        "all": [(FM.Or, False), (FM.And, False), (FM.And, True)],
        "lemma": [(FM.Neighbors, False)],
    },
)


def compute_iou(formula, acts, feats, dataset, feat_type="word"):
    masks = get_mask(feats, formula, dataset, feat_type)
    formula.mask = masks

    if settings.METRIC == "iou":
        comp_iou = iou(masks, acts)
    elif settings.METRIC == "precision":
        comp_iou = precision_score(masks, acts)
    elif settings.METRIC == "recall":
        comp_iou = recall_score(masks, acts)
    else:
        raise NotImplementedError(f"metric: {settings.METRIC}")
    comp_iou = (settings.COMPLEXITY_PENALTY ** (len(formula) - 1)) * comp_iou

    return comp_iou


def load_vecs(path):
    vecs = []
    vecs_stoi = {}
    vecs_itos = {}
    with open(path, "r") as f:
        for line in f:
            tok, *nums = line.split(" ")
            nums = np.array(list(map(float, nums)))
            assert tok not in vecs_stoi
            new_n = len(vecs_stoi)
            vecs_stoi[tok] = new_n
            vecs_itos[new_n] = tok
            vecs.append(nums)
    vecs = np.array(vecs)
    return vecs, vecs_stoi, vecs_itos


# Load GloVe vectors for neighbor search
# STRUCTURAL NOTE: Requires settings.VECPATH to point to valid .vec file
# If file doesn't exist, get_neighbors will return [] for all lemmas
VECS, VECS_STOI, VECS_ITOS = load_vecs(settings.VECPATH)


NEIGHBORS_CACHE = {}


def get_neighbors(lemma):
    """Get neighbors of lemma given glove vectors."""
    if lemma not in VECS_STOI:
        return []
    if lemma in NEIGHBORS_CACHE:
        return NEIGHBORS_CACHE[lemma]
    lemma_i = VECS_STOI[lemma]
    lvec = VECS[lemma_i][np.newaxis]
    dists = cdist(lvec, VECS, metric="cosine")[0]
    nearest_i = np.argsort(dists)[1 : settings.EMBEDDING_NEIGHBORHOOD_SIZE + 1]
    nearest = [VECS_ITOS[i] for i in nearest_i]
    NEIGHBORS_CACHE[lemma] = nearest
    return nearest


def compute_best_sentence_iou(args):
    (unit,) = args

    acts = GLOBALS["acts"][:, unit]
    feats = GLOBALS["feats"]
    dataset = GLOBALS["dataset"]

    if acts.sum() < settings.MIN_ACTS:
        null_f = (FM.Leaf(0), 0)
        return {"unit": unit, "best": null_f, "best_noncomp": null_f}

    feats_to_search = list(range(feats.shape[1]))
    formulas = {}

    # Vectorized initial leaf IOU computation
    if isinstance(feats, dict):
        # We handle word level feats dict fallback just in case, but usually this is sentence feats which is an array
        pass
    elif isinstance(feats, np.ndarray) and feats.ndim == 2:
        acts_matrix = acts[:, np.newaxis]
        intersections = np.logical_and(feats, acts_matrix).sum(axis=0)
        unions = np.logical_or(feats, acts_matrix).sum(axis=0)
        ious_vec = intersections / (unions + np.finfo(np.float32).tiny)

        if settings.METRIC == "iou":
            pass # ious_vec is already iou
        elif settings.METRIC == "precision":
            # precision = true positives / (true positives + false positives) = intersection / feats.sum()
            feats_sum = feats.sum(axis=0)
            ious_vec = intersections / (feats_sum + np.finfo(np.float32).tiny)
        elif settings.METRIC == "recall":
            # recall = true positives / (true positives + false negatives) = intersection / acts.sum()
            acts_sum = acts.sum()
            ious_vec = intersections / (acts_sum + np.finfo(np.float32).tiny)

        for fval in feats_to_search:
            formula = FM.Leaf(fval)
            formula.mask = feats[:, fval] # cache the mask
            formulas[formula] = ious_vec[fval]

            # Keep original loop for the neighbors ops
            for op, negate in OPS["lemma"]:
                new_formula = formula
                if negate:
                    new_formula = FM.Not(new_formula)
                new_formula = op(new_formula)
                new_iou = compute_iou(
                    new_formula, acts, feats, dataset, feat_type="sentence"
                )
                formulas[new_formula] = new_iou

    else:
        # Fallback to loop
        for fval in feats_to_search:
            formula = FM.Leaf(fval)
            formulas[formula] = compute_iou(
                formula, acts, feats, dataset, feat_type="sentence"
            )

            for op, negate in OPS["lemma"]:
                new_formula = formula
                if negate:
                    new_formula = FM.Not(new_formula)
                new_formula = op(new_formula)
                new_iou = compute_iou(
                    new_formula, acts, feats, dataset, feat_type="sentence"
                )
                formulas[new_formula] = new_iou

    import heapq

    # We must carefully ensure that if scores are equal, we preserve arbitrary ordering
    # just as Counter.most_common did, though we are primarily concerned with the highest scores
    nonzero_iou = [k.val for k, v in formulas.items() if v > 0]
    formulas = dict(heapq.nlargest(settings.BEAM_SIZE, formulas.items(), key=lambda item: item[1]))
    best_noncomp = max(formulas.items(), key=lambda item: item[1])

    for i in range(settings.MAX_FORMULA_LENGTH - 1):
        new_formulas = {}
        for formula in formulas:
            for feat in nonzero_iou:
                for op, negate in OPS["all"]:
                    if not isinstance(feat, FM.F):
                        new_formula = FM.Leaf(feat)
                    else:
                        new_formula = feat
                    if negate:
                        new_formula = FM.Not(new_formula)
                    new_formula = op(formula, new_formula)
                    new_iou = compute_iou(
                        new_formula, acts, feats, dataset, feat_type="sentence"
                    )
                    new_formulas[new_formula] = new_iou

        formulas.update(new_formulas)
        formulas = dict(heapq.nlargest(settings.BEAM_SIZE, formulas.items(), key=lambda item: item[1]))

    best = max(formulas.items(), key=lambda item: item[1])

    return {
        "unit": unit,
        "best": best,
        "best_noncomp": best_noncomp,
    }


def search_feats(acts, states, feats, weights, dataset):
    rfile = os.path.join(settings.RESULT, "result.csv")
    if os.path.exists(rfile):
        print(f"Loading cached {rfile}")
        return pd.read_csv(rfile).to_dict("records")

    GLOBALS["acts"] = acts
    GLOBALS["states"] = states
    GLOBALS["feats"] = feats[0]
    GLOBALS["dataset"] = feats[1]
    feats_vocab = feats[1]

    def namer(i):
        return feats_vocab["itos"][i]

    def cat_namer(i):
        return feats_vocab["itos"][i].split(":")[0]

    def cat_namer_fine(i):
        return ":".join(feats_vocab["itos"][i].split(":")[:2])

    ioufunc = compute_best_sentence_iou

    records = []
    if settings.NEURONS is None:
        units = range(acts.shape[1])
    else:
        units = settings.NEURONS
    mp_args = [(u,) for u in units]

    if settings.PARALLEL < 1:
        class FakePool:
            def __init__(self, processes):
                pass
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            def imap_unordered(self, func, args):
                for arg in args:
                    yield func(arg)
        pool_cls = FakePool
    else:
        pool_cls = mp.Pool

    n_done = 0
    with pool_cls(settings.PARALLEL) as pool, tqdm(
        total=len(units), desc="Units"
    ) as pbar:
        for res in pool.imap_unordered(ioufunc, mp_args):
            unit = res["unit"]
            best_lab, best_iou = res["best"]
            best_name = best_lab.to_str(namer, sort=True)
            best_cat = best_lab.to_str(cat_namer, sort=True)
            best_cat_fine = best_lab.to_str(cat_namer_fine, sort=True)

            entail_weight = weights[unit, 0]
            neutral_weight = weights[unit, 1]
            contra_weight = weights[unit, 2]

            if best_iou > 0:
                tqdm.write(f"{unit:02d}\t{best_name}\t{best_iou:.3f}")
            r = {
                "neuron": unit,
                "feature": best_name,
                "category": best_cat,
                "category_fine": best_cat_fine,
                "iou": best_iou,
                "feature_length": len(best_lab),
                "w_entail": entail_weight,
                "w_neutral": neutral_weight,
                "w_contra": contra_weight,
            }
            records.append(r)
            pbar.update()
            n_done += 1
            if n_done % settings.SAVE_EVERY == 0:
                pd.DataFrame(records).to_csv(rfile, index=False)

        if len(records) % 32 == 0:
            pd.DataFrame(records).to_csv(rfile, index=False)

    pd.DataFrame(records).to_csv(rfile, index=False)
    return records


def extract_first_tok(token_txt):
    """Map abbreviated first token to full NLI label."""
    mapping = {"ent": "entailment", "neut": "neutral", "contr": "contradiction"}
    return mapping.get(token_txt)

def evaluate_first_token(logits, tokenizer):
    """
    Print top-k token predictions and extract NLI label from the argmax
    first token of the forward-pass logits.
    """
    probs = logits.softmax(dim=-1).cpu().to(torch.float32)
    topk_result = torch.topk(probs, k=settings.TOP_K, dim=-1, largest=True)
    topk_token_ids = topk_result.indices
    
    predictions = []
    for i in range(topk_token_ids.shape[0]):
        decoded = [tokenizer.decode(topk_token_ids[i, k], skip_special_tokens=True) for k in range(settings.TOP_K)]
        prediction = extract_first_tok(decoded[0])
        print(f"Top-{settings.TOP_K}: {decoded} | {prediction}")
        predictions.append(prediction)
    return predictions

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
    from datasets import load_dataset
    LABEL_MAP = ["entailment", "neutral", "contradiction"]
    hf_dataset = load_dataset("snli", split="validation")
    hf_labels = [LABEL_MAP[i] for i in hf_dataset["label"]]
    
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
    all_predictions = []
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
            out = model(**tokenized)

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

        # Token evaluation
        first_token_logits = out.logits[:, -1, :]
        batch_predictions = evaluate_first_token(first_token_logits, tokenizer)
        all_predictions.extend(batch_predictions)

    h.remove()
    print(f"Hook removed. Captured {len(all_states)} batches.")
    
    all_feats = {"onehot": all_feats, "multi": all_multifeats}
    states = np.concatenate(all_states, axis=0)
    
    print(f"States shape: {states.shape}")
    
    # Token evaluation summary - match predictions to labels using indices
    correct = 0
    for label, pred in zip(hf_labels, all_predictions):
        correct += label == pred
    
    total = len(all_predictions)
    valid = sum(1 for p in all_predictions if p is not None)
    rejected = total - valid
    print(f"Token Eval -- Total: {total}, Valid: {valid}, Rejected: {rejected}")
    if total > 0:
        print(f"Token Eval -- Recognition rate: {valid / total * 100:.2f}%")
    if valid > 0:
        print(f"Token Eval -- Accuracy: {correct / valid * 100:.2f}%")
    
    return all_srcs, states, all_feats, all_idxs, all_predictions, hf_labels

    

def main():
    os.makedirs(settings.RESULT, exist_ok=True)

    model_id = "LiquidAI/LFM2.5-1.2B-Base"
    lora_path = "../finetune/model/checkpoint-1000"
    print(f"Hook layer: {settings.HOOKED_LAYER}")
    print(f"MIN_ACTS: {settings.MIN_ACTS}")
    print(f"Model: {lora_path}")
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
    toks, states, feats, idxs, all_predictions, hf_labels = extract_features_llm(model,tokenizer,analysis_dataset)

    print("Computing quantiles...")
    acts = quantile_features(states)

    print("Extracting classification weights from lm_head...")
    weights = get_classification_weights(model)

    print("Extracting sentence token features...")
    tok_feats, tok_feats_vocab = sentence.to_sentence(toks, feats, analysis_dataset)

    print("Running mask search on sentence features...")
    records = search_feats(acts, states, (tok_feats, tok_feats_vocab), weights, analysis_dataset)

    print("Running mask search on token features...")
    records = search_feats(acts, states, feats, weights, analysis_dataset)

    print("Saving predictions with activations...")
    preds = pd.DataFrame({"pred": all_predictions, "gt": hf_labels})
    save_with_acts(preds, acts, os.path.join(settings.RESULT, "preds_acts.csv"))

    print("Visualizing features...")
    from vis import sentence_report
    sentence_report.make_html(
        records,
        toks,
        states,
        (tok_feats, tok_feats_vocab),
        idxs,
        preds,
        weights,
        analysis_dataset,
        settings.RESULT,
    )

    print("Analysis complete. Results saved to:", settings.RESULT)

if __name__ == "__main__":
    main()