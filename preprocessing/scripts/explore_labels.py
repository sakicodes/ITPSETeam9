import os
import pandas as pd

def main():
    # Define paths
    PROCESSED_DIR = os.path.join("data", "processed")
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    STATS_PATH = os.path.join(PROCESSED_DIR, "label_statistics.csv")
    
    print("Loading Hugging Face dataset via pandas...")
    # Using pandas directly with Hugging Face URI
    df = pd.read_csv("hf://datasets/syedroshanzameer/resume-classification/train.csv")
    
    print("Extracting label statistics...")
    # Calculate label frequencies
    label_counts = df['labels'].value_counts().reset_index()
    label_counts.columns = ['Label', 'Count']
    label_counts = label_counts.sort_values(by='Label')
    
    # Print to console
    print("\nLabel Distribution:")
    print(label_counts.to_string(index=False))
    
    # Export to processed folder
    label_counts.to_csv(STATS_PATH, index=False)
    print(f"\nLabel statistics saved to {STATS_PATH}")
    print("Please review the labels and update TARGET_LABELS in preprocess_resume_classification.py")

if __name__ == "__main__":
    main()