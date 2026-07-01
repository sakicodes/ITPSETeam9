import os
import pandas as pd

# -------------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------------
PROCESSED_DIR = os.path.join("data", "processed")
OUTPUT_PHRASES = os.path.join(PROCESSED_DIR, "merged_phrases.csv")
OUTPUT_STATS = os.path.join(PROCESSED_DIR, "merged_statistics.csv")

INPUT_FILES = [
    "sales_phrases.csv",
    "resume_classification_phrases.csv"
]

# -------------------------------------------------------------------------
# MAIN EXECUTION
# -------------------------------------------------------------------------

def main():
    print("Starting Dataset Merge Pipeline...")
    
    all_dataframes = []
    datasets_merged = 0
    total_phrases_before = 0
    
    for filename in INPUT_FILES:
        filepath = os.path.join(PROCESSED_DIR, filename)
        
        if not os.path.exists(filepath):
            print(f"Warning: {filename} not found in {PROCESSED_DIR}. Skipping.")
            continue
            
        try:
            df = pd.read_csv(filepath)
            
            # Standardize Category / Label column naming
            if 'Category' in df.columns:
                df = df.rename(columns={'Category': 'Category_or_Label'})
            elif 'Label' in df.columns:
                df = df.rename(columns={'Label': 'Category_or_Label'})
            
            # Inject Source Dataset identifier
            df['Source_Dataset'] = filename.replace('.csv', '')
            
            # Ensure required columns exist to avoid concat errors
            required_columns = ['Source_Dataset', 'Resume_ID', 'Category_or_Label', 'Phrase', 'Word_Count']
            df = df[[col for col in required_columns if col in df.columns]]
            
            all_dataframes.append(df)
            datasets_merged += 1
            total_phrases_before += len(df)
            print(f"Loaded {len(df)} phrases from {filename}")
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    if not all_dataframes:
        print("No data loaded. Exiting.")
        return

    # Concatenate all datasets
    merged_df = pd.concat(all_dataframes, ignore_index=True)
    
    # Deduplicate case-insensitively across ALL datasets
    merged_df['Phrase_Lower'] = merged_df['Phrase'].str.lower().str.strip()
    final_df = merged_df.drop_duplicates(subset=['Phrase_Lower']).drop(columns=['Phrase_Lower'])
    
    duplicate_phrases_removed = total_phrases_before - len(final_df)
    
    # Sort alphabetically by Phrase
    final_df = final_df.sort_values(by='Phrase', ignore_index=True)
    
    # Generate new standardized Phrase IDs
    final_df.insert(0, 'Phrase_ID', [f"P{str(i).zfill(6)}" for i in range(1, len(final_df) + 1)])
    
    # Reorder columns to match requirements perfectly
    final_cols = ['Phrase_ID', 'Source_Dataset', 'Resume_ID', 'Category_or_Label', 'Phrase', 'Word_Count']
    final_df = final_df[final_cols]
    
    # Export Merged Dataset
    final_df.to_csv(OUTPUT_PHRASES, index=False)
    print(f"\nSaved {len(final_df)} merged phrases to {OUTPUT_PHRASES}")
    
    # Compile and Export Statistics
    stats = {
        "Datasets merged": datasets_merged,
        "Total phrases before merge": total_phrases_before,
        "Duplicate phrases removed": duplicate_phrases_removed,
        "Final phrases retained": len(final_df),
        "Average phrase length": round(final_df['Word_Count'].mean(), 1)
    }
    
    stats_df = pd.DataFrame(list(stats.items()), columns=['Metric', 'Value'])
    stats_df.to_csv(OUTPUT_STATS, index=False)
    print(f"Saved merge statistics to {OUTPUT_STATS}")
    print("Merge complete.")

if __name__ == "__main__":
    main()