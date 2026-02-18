from collections import Counter
from transformers import pipeline, AutoTokenizer
from datasets import load_dataset

def extract_classification(response_text):
    response = str(response_text.lower()[response_text.index("Answer:"):])
    # TEMPORARY
    print(f"\033[32mAnswer\033[0m: {response}")

    classifications = [ "entailment", "neutral", "contradiction"]
    classifications = {k: response.find(k) for k in classifications}

    return min(
        filter(lambda x : x[1] >= 0, classifications.items()),
        key=lambda kv : kv[1],
        default=(None,None) 
    )[0]
    
def build_NLI_prompt(example):
    example["prompt"] = f"""Premise: {example["premise"]}
Hypothesis: {example["hypothesis"]}

Answer with ONLY one word: "entailment" or "neutral" or "contradiction"
Do not explain the reasoning.

Answer:"""
    return example

model = "LiquidAI/LFM2.5-1.2B-Base"
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
responses_raw = pipe(reduced_prompts,max_new_tokens=10,batch_size=100)
print("Inference finished!")

responses = [resp[0]["generated_text"] for resp in responses_raw]
predictions = [extract_classification(resp) for resp in responses]

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
