import os
import pandas as pd

RAW_PHRASES_PATH = os.path.join("data", "processed", "jd_raw_phrases_balanced.csv")
LOG_PATH = os.path.join("outputs", "phrase_classification_log.csv")
OUTPUT_PATH = os.path.join("data", "processed", "jd_phrases.csv")

def main():
    print("Stage 4: Building Final Phrase Dataset...")
    if not os.path.exists(LOG_PATH) or not os.path.exists(RAW_PHRASES_PATH):
        return print("Missing inputs.")
        
    df_raw = pd.read_csv(RAW_PHRASES_PATH)
    df_log = pd.read_csv(LOG_PATH)
    
    # 1. Filter out removed phrases
    df_log = df_log[df_log['Decision'].str.lower() != 'remove']
    
    # 2. Map Hallucinated Categories to Standard Ones
    def standardize_type(ptype):
        ptype = str(ptype).lower().strip()
        if 'respons' in ptype or 'role' in ptype or 'position' in ptype or 'admin' in ptype:
            return 'Responsibility'
        if 'require' in ptype or 'qualif' in ptype:
            return 'Requirement'
        if 'comp' in ptype or 'trait' in ptype or 'skill' in ptype:
            return 'Skill'
        return 'Other'
        
    df_log['Phrase_Type'] = df_log['Phrase_Type'].apply(standardize_type)
    
    # 3. Merge metadata
    df_final = pd.merge(
        df_log[['Phrase_ID', 'Split_Output', 'Phrase_Type', 'Competency']], 
        df_raw.drop(columns=['Phrase', 'Word_Count']), 
        on='Phrase_ID', how='inner'
    )
    
    # Rename and recalculate Word Count for splits
    df_final = df_final.rename(columns={'Split_Output': 'Phrase'})
    df_final['Word_Count'] = df_final['Phrase'].apply(lambda x: len(str(x).split()))
    
    # Reorder columns to match spec
    cols = ['Phrase_ID', 'Source', 'Source_Record_ID', 'Company', 'Country', 'Industry', 
            'Job_Function', 'Position', 'Seniority', 'Phrase', 'Word_Count', 'Phrase_Type', 'Competency']
    
    df_final = df_final[[c for c in cols if c in df_final.columns]]
    df_final.to_csv(OUTPUT_PATH, index=False)
    print(f"Generated clean phrase dataset with {len(df_final)} rows.")

if __name__ == "__main__":
    main()