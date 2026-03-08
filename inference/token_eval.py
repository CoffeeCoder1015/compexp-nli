from collections import Counter
import re
from transformers import pipeline, AutoTokenizer
from datasets import load_dataset
import torch
from tqdm import tqdm
import lora_loader
import numpy as np

# system prompt directive for the NLI task
system_prompt = {
    "role":"system",
    "content":"""Determine the relationship between the `Premise`and `Hypothesis` and respond with an answer. 
You must respond with an answer of `entailment`, `neutral` or `contradiction`
You need to respond in the format shown in the following by chosing one of thosw answers:
<my_answer>[place your answer here]</my_answer>
You are lay out the steps to your final answer before responding with your final answer, but you must respond in this format or else your answer will be rejected."""
}

def extract_classification(response_chat):
    assistant_response = response_chat[2]
    assert(assistant_response["role"] == "assistant")
    content = assistant_response.get("content", "")
    print(f"\n\033[92mAssistant's Response:\033[0m {content}")
    raw_json_answer = re.findall("<my_answer>(.*)</my_answer>",content,re.DOTALL)
    accepetd_answers = set([ "entailment", "neutral", "contradiction"])
    if len(raw_json_answer) > 0:
        answer = raw_json_answer[0].lower()
        if answer in accepetd_answers:
            return answer
    else:
        return None

def extract_first(response_chat):
    # assistant_response = response_chat[2]
    # assert(assistant_response["role"] == "assistant")

    content = response_chat

    # print(f"\n\033[92mAssistant's Response:\033[0m {content}")

    classifications = [ "entailment", "neutral", "contradiction"]
    classifications = {k: content.find(k) for k in classifications}

    return min(
        filter(lambda x : x[1] >= 0, classifications.items()),
        key=lambda kv : kv[1],
        default=(None,None) 
    )[0]
    
def extract_first_tok(token_txt):
    if token_txt == "ent":
        return "entailment"
    elif token_txt == "neut":
        return "neutral"
    elif token_txt == "contr":
        return "contradiction"

pipeline_qwen = {
    "model":"Qwen/Qwen3-1.7B",
    "eval":extract_classification,
    "token_limit":1000,
    "batching_size":64,
    "lora_path": None
} 

pipeline_liquid ={
    "model":"LiquidAI/LFM2.5-1.2B-Base",
    "eval":extract_first,
    "token_limit":300,
    "batching_size":128,
    # "lora_path": "../finetune/SFT/checkpoint-1000" # Load from training output
    "lora_path": "../finetune/model/checkpoint-1000" # Load from saved output
}

# SELCT: model to run with its configs
active_pipeline = pipeline_liquid

def build_NLI_prompt(example):
    test_example = f"Premise: {example["premise"]}\nHypothesis: {example["hypothesis"]}"
    prompt = [system_prompt, {"role":"user","content":test_example} ]
    example["prompt"] = prompt
    return example

model_id = active_pipeline["model"]
lora_path = active_pipeline.get("lora_path")
model, tokenizer = lora_loader.get_model_with_lora(model_id, lora_path=lora_path)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
dataset = load_dataset("snli", split="validation")
SNLI_query = dataset.map(build_NLI_prompt)[:]


# lables, Y
classification_map = ["entailment","neutral","contradiction"]
labels_raw = SNLI_query["label"]
word_labels = [classification_map[i] for i in labels_raw]

# prompts, X
reduced_prompts = SNLI_query["prompt"]

print("Starting inference.")

responses_raw = []
batch_size = active_pipeline["batching_size"]

# first token
nli_apperances = 0

full_coverage_probmass = []
full_coverage_std = []

under_coverage_probmass = []
under_coverage_std = []

with torch.inference_mode():
    for i in tqdm(range(0, len(reduced_prompts), batch_size), desc="Generating"):
        batch = reduced_prompts[i:i+batch_size]
        tokenized = tokenizer.apply_chat_template(batch,add_generation_prompt=True,padding=True,return_dict=True,return_tensors="pt").to(model.device)
        length = tokenized["input_ids"].shape[1]
        out = model.generate(**tokenized,max_new_tokens=active_pipeline["token_limit"], output_scores=True, return_dict_in_generate=True)
        probability_volume = torch.stack(out.scores).softmax(-1).cpu()
        topk_result = torch.topk(probability_volume,k=3,dim=-1,largest=True)
        topk_tokens = topk_result.indices
        topk_probs = topk_result.values.numpy()
        fully_decoded = []
        for i in range(topk_tokens.shape[0]):
            batch_level_tokens = []
            for j in range(topk_tokens.shape[1]):
                res = tokenizer.batch_decode(topk_tokens[i][j])
                batch_level_tokens.append(res)
            fully_decoded.append(batch_level_tokens)
        fully_decoded = np.array(fully_decoded)       
        
        first_token = fully_decoded[0,:,:]
        for j,t3 in enumerate(first_token):
            t3r = t3
            t3 = set(t3.tolist())
            gain = 0
            if "neut" in t3:
                gain+=1
            if "contr" in t3:
                gain+=1
            if "ent" in t3:
                gain+=1
            if gain < 3:
                ftk_probs = topk_probs[0,j,:]
                probmass = ftk_probs.sum()
                std = ftk_probs.std()
                print(" Failed:",t3,ftk_probs,probmass,std)
                under_coverage_probmass.append(probmass)
                under_coverage_std.append(std)
            elif gain == 3:
                ftk_probs = topk_probs[0,j,:]
                probmass = ftk_probs.sum()
                std = ftk_probs.std()
                print("Success:",t3r,np.round(ftk_probs,decimals=3),probmass,std)
                full_coverage_probmass.append(probmass)
                full_coverage_std.append(std)
            nli_apperances+=gain
        
        packed = np.stack((fully_decoded,topk_probs),axis=-1)


        out = out.sequences[:,length:]
        decoded = tokenizer.batch_decode(out)
        responses_raw.extend(decoded)

print("Inference finished!")
print(nli_apperances,3*len(reduced_prompts),nli_apperances/( 3*len(reduced_prompts) )*100)
print("full_coverage_probmass mean:",np.mean(full_coverage_probmass))
print("full_coverage_std mean:",np.mean(full_coverage_std))
print("under_coverage_probmass mean:",np.mean(under_coverage_probmass))
print("under_coverage_std mean:",np.mean(under_coverage_std))

responses = [resp for resp in responses_raw]
predictions = [active_pipeline["eval"](resp) for resp in responses]


def NLI_statsitics(x,y):
    if x == y:
        return "success"
    elif x != y and x != None:
        return "fail"
    else:
        return "reject"

results = list(map(NLI_statsitics , predictions,word_labels))

stats = Counter(results)
print("\033[94mSuccess:\033[0m", stats['success'])
print("\033[94mFail:\033[0m", stats['fail'])
print("\033[94mReject:\033[0m", stats['reject'])

total = sum(stats.values())
accuracy = stats['success']/total
print("\033[94mAccuracy:\033[0m", accuracy)

# Label statistics
label_stats = Counter(word_labels)
total = sum(label_stats.values())
ent = label_stats["entailment"]
neu = label_stats["neutral"]
con = label_stats["contradiction"]
def print_stats(Name,Num,Percent):
    print(f"\033[38;2;255;165;0m{Name}\033[0m: {Num}  {Percent*100}%")
print_stats("Entailment",ent,ent/total)
print_stats("Neutral",neu,neu/total)
print_stats("Contradiction",con,con/total)





