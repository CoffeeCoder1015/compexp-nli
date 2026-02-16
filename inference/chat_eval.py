from collections import Counter
import json
import re
from transformers import pipeline, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

# system prompt directive for the NLI task
system_prompt = {
    "role":"system",
    "content":"""Determine the relationship between the `Premise`and `Hypothesis` and respond with an answer. 
You must respond with an answer of `Entailment`, `Neutral` or `Contradiction`
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
    assistant_response = response_chat[2]
    assert(assistant_response["role"] == "assistant")

    content = assistant_response.get("content", "").lower()

    print(f"\n\033[92mAssistant's Response:\033[0m {content}")

    classifications = [ "entailment", "neutral", "contradiction"]
    classifications = {k: content.find(k) for k in classifications}

    return min(
        filter(lambda x : x[1] >= 0, classifications.items()),
        key=lambda kv : kv[1],
        default=(None,None) 
    )[0]

pipeline_qwen = {
    "model":"Qwen/Qwen3-1.7B",
    "eval":extract_classification,
    "token_limit":1000,
    "batching_size":64
} 

pipeline_liquid ={
    "model":"LiquidAI/LFM2.5-1.2B-Base",
    "eval":extract_first,
    "token_limit":300,
    "batching_size":128
}

# SELCT: model to run with its configs
active_pipeline = pipeline_qwen

def build_NLI_prompt(example):
    test_example = f"Premise: {example["premise"]}\nHypothesis: {example["hypothesis"]}"
    prompt = [system_prompt, {"role":"user","content":test_example} ]
    example["prompt"] = prompt
    return example

model = active_pipeline["model"]
tokenizer = AutoTokenizer.from_pretrained(model)
tokenizer.padding_side = "left" #for batched prompts so tokens are of the form [<pad> prompts] and not [prompt <pad>]

pipe = pipeline("text-generation", model=model,tokenizer=tokenizer)
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

for i in tqdm(range(0, len(reduced_prompts), batch_size), desc="Generating"):
    batch = reduced_prompts[i:i+batch_size]
    out = pipe(
        batch,
        max_new_tokens=active_pipeline["token_limit"],
        batch_size=batch_size,
        num_workers=8
    )
    responses_raw.extend(out)

print("Inference finished!")

responses = [resp[0]["generated_text"] for resp in responses_raw]
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

