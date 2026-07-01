import os
import pandas as pd

# -------------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------------
PROCESSED_DIR = os.path.join("data", "processed")
INPUT_FILE = os.path.join(PROCESSED_DIR, "classified_phrases.csv")
OUTPUT_FILE = os.path.join(PROCESSED_DIR, "filtered_phrases.csv")
STATS_FILE = os.path.join(PROCESSED_DIR, "final_statistics.csv")

def main():
    print("Starting Final Filtering Process...")
    
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Could not find {INPUT_FILE}")
        return
        
    # Load the classified data
    df = pd.read_csv(INPUT_FILE)
    initial_count = len(df)
    
    # 1. Filter out 'Unknown' Industries
    # We use str.lower() to make it case-insensitive just in case the API capitalized it weirdly
    df_filtered = df[df['Industry'].str.lower() != 'unknown'].copy()
    
    # 2. (Optional) Filter out any failed Framings if the API hallucinated
    valid_framings = ['agentic', 'communal', 'neutral', 'balanced']
    df_filtered = df_filtered[df_filtered['Framing'].str.lower().isin(valid_framings)]
    
    # Calculate stats
    final_count = len(df_filtered)
    dropped_count = initial_count - final_count
    
    # Save the final pristine dataset
    df_filtered.to_csv(OUTPUT_FILE, index=False)
    
    # Print and save statistics
    print(f"\n--- Final Statistics ---")
    print(f"Total phrases before filtering: {initial_count}")
    print(f"Phrases removed (Unknown/Invalid): {dropped_count}")
    print(f"Final phrases retained: {final_count}")
    print(f"------------------------")
    print(f"\nSaved final master dataset to: {OUTPUT_FILE}")
    
    # Save stats to CSV
    stats = pd.DataFrame([
        {"Metric": "Total phrases before filtering", "Value": initial_count},
        {"Metric": "Phrases removed (Unknown/Invalid)", "Value": dropped_count},
        {"Metric": "Final phrases retained", "Value": final_count}
    ])
    stats.to_csv(STATS_FILE, index=False)

if __name__ == "__main__":
    main()