import pandas as pd
import numpy as np
from pathlib import Path

def get_experiments():
    exp_dir = Path("exp")
    exps = [d.name for d in exp_dir.iterdir() if d.is_dir() and (d / "result.csv").exists()]
    return sorted(exps)

def load_exp(name, filter_boring=True):
    df = pd.read_csv(Path("exp") / name / "result.csv")
    if filter_boring:
        df = df[df["feature"] != "pre:tok:in"]
    df = df[["neuron", "feature"]].copy()
    df["neuron"] = df["neuron"].astype(int)
    df = df.sort_values(by="neuron")
    return df

def _print_neuron_list(neurons, label):
    """Print list of neuron IDs without compression."""
    print(f"\n[{label}]: {sorted(neurons)}")

def _truncate_features(feature_series, max_len=60):
    """Truncate long feature strings for concise display."""
    truncated = feature_series.astype(str).str[:max_len]
    truncated[feature_series.astype(str).str.len() > max_len] += '...'
    return truncated

def _print_stats(total, diff, only1, only2, name1, name2):
    """Print a formatted statistics summary."""
    print("\n--- Statistics ---")
    print(f"Total unique neurons compared: {total}")
    print(f"Neurons with differing explanations: {diff}")
    print(f"Neurons only in {name1}: {only1}")
    print(f"Neurons only in {name2}: {only2}")

def _print_individual_dataset_stats(exp_name, df):
    """Print statistics for a single dataset."""
    print(f"\n--- {exp_name} Statistics ---")
    print(f"Total neurons: {df['neuron'].nunique()}")
    print(f"Total explanations: {len(df)}")
    print(f"Unique features: {df['feature'].nunique()}")
    print(f"Neurons with multiple features: {(df.groupby('neuron').size() > 1).sum()}")

def main():
    exps = get_experiments()
    if not exps:
        print("No experiments found in exp/ with result.csv")
        return

    print("Available experiments:")
    for i, name in enumerate(exps):
        print(f"{i}: {name}")

    print("\nOptions:")
    print("1. Pick 2 to diff (Detailed)")
    print("2. Full combinational diff of all explanations")
    
    choice = input("\nChoose an option (1 or 2): ")
    filter_boring = True  # Always filter out 'pre:tok:in'

    if choice == "1":
        if len(exps) < 2:
            print("Need at least 2 experiments to diff.")
            return
        idx1 = int(input(f"Pick first experiment index (0-{len(exps)-1}): "))
        idx2 = int(input(f"Pick second experiment index (0-{len(exps)-1}): "))
        
        name1, name2 = exps[idx1], exps[idx2]
        df1 = load_exp(name1, filter_boring)
        df2 = load_exp(name2, filter_boring)
        
        _print_individual_dataset_stats(name1, df1)
        _print_individual_dataset_stats(name2, df2)
        
        # Use outer join to find mismatches and missing neurons
        merged = pd.merge(df1, df2, on="neuron", how="outer", suffixes=(f"_{name1}", f"_{name2}"))
        
        # 1. Different explanations
        diff_mask = (merged[f"feature_{name1}"].notna()) & \
                    (merged[f"feature_{name2}"].notna()) & \
                    (merged[f"feature_{name1}"] != merged[f"feature_{name2}"])
        diff_exps = merged[diff_mask]

        # 2. Only in A
        only_in_1 = merged[merged[f"feature_{name2}"].isna()]
        
        # 3. Only in B
        only_in_2 = merged[merged[f"feature_{name1}"].isna()]
        
        # Print differences and statistics for the two selected experiments
        # Focus on neurons missing in one but present in the other, then differing features
        # 2. Only in first experiment (priority 1)
        if not only_in_1.empty:
            only1_neurons = only_in_1["neuron"].tolist()
            _print_neuron_list(only1_neurons, f"Neurons ONLY in {name1}")
            sample_df = only_in_1.copy()
            sample_df[f"feature_{name1}"] = _truncate_features(sample_df[f"feature_{name1}"])
            print(sample_df[["neuron", f"feature_{name1}"]].head().to_string(index=False))
        
        # 3. Only in second experiment (priority 2)
        if not only_in_2.empty:
            only2_neurons = only_in_2["neuron"].tolist()
            _print_neuron_list(only2_neurons, f"Neurons ONLY in {name2}")
            sample_df = only_in_2.copy()
            sample_df[f"feature_{name2}"] = _truncate_features(sample_df[f"feature_{name2}"])
            print(sample_df[["neuron", f"feature_{name2}"]].head().to_string(index=False))
        
        # 1. Different explanations (priority 3)
        if not diff_exps.empty:
            diff_neurons = diff_exps["neuron"].tolist()
            _print_neuron_list(diff_neurons, f"Neurons with DIFFERENT explanations between {name1} and {name2}")
            print("Sample differing explanations (truncated):")
            sample_df = diff_exps.copy()
            sample_df[f"feature_{name1}"] = _truncate_features(sample_df[f"feature_{name1}"])
            sample_df[f"feature_{name2}"] = _truncate_features(sample_df[f"feature_{name2}"])
            print(sample_df[["neuron", f"feature_{name1}", f"feature_{name2}"]].head().to_string(index=False))
        
        # Statistics summary
        total_neurons = merged["neuron"].nunique()
        _diff_count = diff_exps.shape[0]
        _only1_count = only_in_1.shape[0]
        _only2_count = only_in_2.shape[0]
        _print_stats(total_neurons, _diff_count, _only1_count, _only2_count, name1, name2)
        
        if diff_exps.empty and only_in_1.empty and only_in_2.empty:
            print("\nExperiments are identical.")
        
    elif choice == "2":
        all_dfs = []
        dfs_for_stats = []
        for name in exps:
            df = load_exp(name, filter_boring)
            df_for_stats = df.copy()
            df.columns = ["neuron", name]
            all_dfs.append(df)
            dfs_for_stats.append((name, df_for_stats))
        
        # Print individual dataset statistics
        for name, df in dfs_for_stats:
            _print_individual_dataset_stats(name, df)
        
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
            # Truncate features for display
            display_df = diff_combined.copy()
            for col in exps:
                display_df[col] = _truncate_features(display_df[col])
            print(display_df.to_string(index=False))

if __name__ == "__main__":
    main()
