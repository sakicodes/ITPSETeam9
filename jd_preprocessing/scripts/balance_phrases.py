import os
import pandas as pd

INPUT_PATH = os.path.join("data", "processed", "jd_raw_phrases.csv")
OUTPUT_PATH = os.path.join("data", "processed", "jd_raw_phrases_balanced.csv")

def main():
    print("Balancing Dataset to save API costs...")
    if not os.path.exists(INPUT_PATH): return print("Raw phrases not found.")
    
    df = pd.read_csv(INPUT_PATH)

    print("\n--- Original Distribution ---")
    print(df['Industry'].value_counts())

    # Find the size of the smallest industry category
    min_size = df['Industry'].value_counts().min()
    print(f"\nTargeting {min_size} phrases per industry to ensure a balanced FYP dataset...")

    balanced_dfs = []
    for industry, group in df.groupby('Industry'):
        if len(group) > min_size:
            # Randomly sample the majority class down to the minority size
            balanced_dfs.append(group.sample(n=min_size, random_state=42))
        else:
            # Keep all rows of the minority class
            balanced_dfs.append(group)

    # Combine and shuffle
    final_df = pd.concat(balanced_dfs, ignore_index=True)
    final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)

    final_df.to_csv(OUTPUT_PATH, index=False)

    print("\n--- Balanced Distribution ---")
    print(final_df['Industry'].value_counts())
    print(f"\nSuccess! Saved {len(final_df)} perfectly balanced phrases to {OUTPUT_PATH}")
    print("Estimated Stage 3 API Cost: ~$0.01")

if __name__ == "__main__":
    main()