import pandas as pd

results = pd.read_csv("exp/snli_1.0_dev-6-sentence-5/result.csv")
filtered = results[results["feature"] != "pre:tok:in"]
top_10_neuron_exps = filtered.head(10)[["neuron","feature"]].reset_index(drop=True)
for i in range(10):
    entry = top_10_neuron_exps.iloc[i]
    print(f"{i+1}. {entry["neuron"]} - {entry["feature"]}")
