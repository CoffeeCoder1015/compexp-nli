from transformers import pipeline
from datasets import load_dataset

def extract_classification(response_text, prompt_len):
    classifications = {
        "entailment": 0, 
        "neutral": 1, 
        "contradiction": 2
    }

    response = response_text.lower()[prompt_len:]

    # TEMPORARY
    print(f"Response after prompt: {response}")


    first_index = float("inf")
    first_class = ""
    
    for c in classifications.keys():
        index = response.find(c)
        if index != -1 and index < first_index:
            first_index = index
            first_class = c

    if first_class:
        return classifications[first_class]
    else:
        return -1

def main():
    pipe = pipeline("text-generation", model="LiquidAI/LFM2.5-1.2B-Base")
    dataset = load_dataset("snli", split="validation")

    correct = 0
    total = 0

    for i in range(10):
        example = dataset[i]
        premise = example["premise"]
        hypothesis = example["hypothesis"]
        label = example["label"]

        prompt = f"""\
        Premise: {premise}
        Hypothesis: {hypothesis}

        Answer with ONLY one word: "entailment" or "neutral" or "contradiction"
        Do not explain the reasoning.

        Answer: 
        """

        prompt_len = len(prompt)

        response = pipe(prompt)
        response_text = response[0]["generated_text"]

        # TEMPORARY
        print(f"Response text:\n\n{response_text}")

        prediction = extract_classification(response_text, prompt_len)

        # TEMPORARY
        print(f"Example {i+1}\nPrediction = {prediction}, Label = {label}")

        if prediction == label:
            correct += 1
        total +=1

    print(f"Accuracy: {correct/total:.4f}")

if __name__ == "__main__":
    main()