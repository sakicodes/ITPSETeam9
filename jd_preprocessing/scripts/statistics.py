import os
import pandas as pd

def main():
    print("=== Phase 6: Pipeline Statistics Aggregator ===\n")
    
    metrics = []
    
    # ---------------------------------------------------------
    # Exact Project Paths based on your pipeline scripts
    # ---------------------------------------------------------
    RAW_DATA_PATH = os.path.join("data", "raw", "input.csv")
    CLEAN_JDS_PATH = os.path.join("data", "processed", "clean_jds.csv")
    RAW_PHRASES_PATH = os.path.join("data", "processed", "jd_raw_phrases.csv")
    BALANCED_PHRASES_PATH = os.path.join("data", "processed", "jd_raw_phrases_balanced.csv")
    CLASS_LOG_PATH = os.path.join("outputs", "phrase_classification_log.csv")
    FINAL_PHRASES_PATH = os.path.join("data", "processed", "jd_phrases.csv")
    EXP_JDS_PATH = os.path.join("outputs", "experimental_jds.csv")
    OUTPUT_CSV = os.path.join("outputs", "statistics.csv")

    # ---------------------------------------------------------
    # 1. Phase 1: Raw Data & Cleaning (from validate_metadata.py)
    # ---------------------------------------------------------
    if os.path.exists(RAW_DATA_PATH) and os.path.exists(CLEAN_JDS_PATH):
        try:
            df_raw = pd.read_csv(RAW_DATA_PATH)
            df_clean = pd.read_csv(CLEAN_JDS_PATH)
            
            total_raw = len(df_raw)
            retained = len(df_clean)
            
            metrics.append({"Metric": "Total raw JDs", "Value": float(total_raw)})
            metrics.append({"Metric": "Rows retained", "Value": float(retained)})
            metrics.append({"Metric": "Rows removed", "Value": float(total_raw - retained)})
            
            # Industry Distribution
            ind_counts = df_clean['Industry'].value_counts()
            for ind, count in ind_counts.items():
                metrics.append({"Metric": f"Industry distribution: {ind}", "Value": float(count)})
        except Exception as e:
            print(f"Error reading Phase 1 data: {e}")

    # ---------------------------------------------------------
    # 2. Phase 2: Phrase Extraction (from extract_jd_phrases.py)
    # ---------------------------------------------------------
    raw_phrases_count = 0
    if os.path.exists(RAW_PHRASES_PATH):
        try:
            df_raw_phrases = pd.read_csv(RAW_PHRASES_PATH)
            raw_phrases_count = len(df_raw_phrases)
            metrics.append({"Metric": "Total phrases extracted", "Value": float(raw_phrases_count)})
            
            if 'Word_Count' in df_raw_phrases.columns:
                avg_len = df_raw_phrases['Word_Count'].mean()
                metrics.append({"Metric": "Average extracted phrase length", "Value": round(avg_len, 1)})
        except Exception as e:
            print(f"Error reading Phase 2 data: {e}")

    # ---------------------------------------------------------
    # 3. Phase 3: Dataset Balancing (from balance_phrases.py)
    # ---------------------------------------------------------
    if os.path.exists(BALANCED_PHRASES_PATH):
        try:
            df_balanced = pd.read_csv(BALANCED_PHRASES_PATH)
            balanced_count = len(df_balanced)
            
            metrics.append({"Metric": "Phrases retained after balancing", "Value": float(balanced_count)})
            if raw_phrases_count > 0:
                metrics.append({"Metric": "Phrases removed during balancing", "Value": float(raw_phrases_count - balanced_count)})
            
            # Show the perfectly balanced industry distribution
            ind_counts_bal = df_balanced['Industry'].value_counts()
            for ind, count in ind_counts_bal.items():
                metrics.append({"Metric": f"Balanced phrases: {ind}", "Value": float(count)})
        except Exception as e:
            print(f"Error reading Phase 3 balanced data: {e}")
            
    # ---------------------------------------------------------
    # 4. Phase 4: Classification Logs (from classify_jd_phrases.py)
    # ---------------------------------------------------------
    if os.path.exists(CLASS_LOG_PATH):
        try:
            df_log = pd.read_csv(CLASS_LOG_PATH)
            
            # Count removed phrases
            removed = len(df_log[df_log['Decision'].str.lower() == 'remove'])
            metrics.append({"Metric": "Phrases rejected by AI Classifier", "Value": float(removed)})
            
            # Count split phrases
            df_kept = df_log[df_log['Decision'].str.lower() != 'remove']
            extra_split_rows = len(df_kept) - df_kept['Phrase_ID'].nunique()
            metrics.append({"Metric": "Phrases split by AI Classifier", "Value": float(extra_split_rows)})
        except Exception as e:
            print(f"Error reading Phase 4 log data: {e}")

    # ---------------------------------------------------------
    # 5. Phase 5: Final Built Phrases (from build_jd_phrases.py)
    # ---------------------------------------------------------
    if os.path.exists(FINAL_PHRASES_PATH):
        try:
            df_phrases = pd.read_csv(FINAL_PHRASES_PATH)
            
            neutral = len(df_phrases[df_phrases['Competency'].str.lower() == 'neutral'])
            agentic = len(df_phrases[df_phrases['Competency'].str.lower() == 'agentic'])
            communal = len(df_phrases[df_phrases['Competency'].str.lower() == 'communal'])
            
            metrics.append({"Metric": "Final Neutral phrases", "Value": float(neutral)})
            metrics.append({"Metric": "Final Agentic phrases", "Value": float(agentic)})
            metrics.append({"Metric": "Final Communal phrases", "Value": float(communal)})
            metrics.append({"Metric": "Total retained phrases in dictionary", "Value": float(len(df_phrases))})
        except Exception as e:
            print(f"Error reading Phase 5 phrases data: {e}")
        
    # ---------------------------------------------------------
    # 6. Phase 6: Generated JDs (from generate_jds.py)
    # ---------------------------------------------------------
    if os.path.exists(EXP_JDS_PATH):
        try:
            df_jds = pd.read_csv(EXP_JDS_PATH)
            metrics.append({"Metric": "Experimental JDs generated", "Value": float(len(df_jds))})
        except Exception as e:
            print(f"Error reading Phase 6 generated JD data: {e}")

    # ---------------------------------------------------------
    # Save and Export
    # ---------------------------------------------------------
    if metrics:
        stats_df = pd.DataFrame(metrics)
        stats_df.to_csv(OUTPUT_CSV, index=False)
        print(f"Successfully rebuilt statistics and saved to: {OUTPUT_CSV}\n")
        print(stats_df.to_string(index=False))
    else:
        print("Warning: No files found to generate statistics. Please check your data/ and outputs/ directories.")

if __name__ == "__main__":
    main()