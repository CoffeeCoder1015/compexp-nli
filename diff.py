import pandas as pd
from pathlib import Path

def get_experiments():
    exp_dir = Path("exp")
    exps = [d.name for d in exp_dir.iterdir() if d.is_dir() and (d / "result.csv").exists()]
    return sorted(exps)

def load_exp(name, filter_boring=True):
    df = pd.read_csv(Path("exp") / name / "result.csv")
    if filter_boring:
        df = df[df["feature"] != "pre:tok:in"]
    return df[["neuron", "feature"]]

def main():
    exps = get_experiments()
    if not exps:
        print("No experiments found in exp/ with result.csv")
        return

    print("Available experiments:")
    for i, name in enumerate(exps):
        print(f"{i}: {name}")

    print("\nOptions:")
    print("1. Pick 2 to diff")
    print("2. Full combinational diff of all explanations")
    
    choice = input("\nChoose an option (1 or 2): ")
    filter_choice = input("Filter 'pre:tok:in' explanations? (y/n, default y): ").lower()
    filter_boring = filter_choice != 'n'

    if choice == "1":
        if len(exps) < 2:
            print("Need at least 2 experiments to diff.")
            return
        idx1 = int(input(f"Pick first experiment index (0-{len(exps)-1}): "))
        idx2 = int(input(f"Pick second experiment index (0-{len(exps)-1}): "))
        
        name1, name2 = exps[idx1], exps[idx2]
        df1 = load_exp(name1, filter_boring)
        df2 = load_exp(name2, filter_boring)
        
        merged = pd.merge(df1, df2, on="neuron", suffixes=(f"_{name1}", f"_{name2}"))
        diff = merged[merged[f"feature_{name1}"] != merged[f"feature_{name2}"]]
        
        if diff.empty:
            print("No differences found between these two experiments.")
        else:
            print(f"\nDifferences between {name1} and {name2}:")
            print(diff.to_string(index=False))

    elif choice == "2":
        all_dfs = []
        for name in exps:
            df = load_exp(name, filter_boring)
            df.columns = ["neuron", name]
            all_dfs.append(df)
        
        if not all_dfs:
            print("No data loaded.")
            return

        from functools import reduce
        combined = reduce(lambda left, right: pd.merge(left, right, on="neuron", how="outer"), all_dfs)
        
        # Identify rows where not all explanations are the same
        # If there's only one experiment, this will show all neurons
        if len(exps) > 1:
            def has_diff(row):
                vals = row.drop("neuron").dropna().unique()
                return len(vals) > 1
            diff_combined = combined[combined.apply(has_diff, axis=1)]
        else:
            diff_combined = combined
        
        if diff_combined.empty:
            print("No differences found across all experiments.")
        else:
            print("\nExplanations (neurons with varying explanations if multiple exps):")
            print(diff_combined.to_string(index=False))
    else:
        print("Invalid choice.")

if __name__ == "__main__":
    main()
