import os
import pandas as pd

INPUT_PATH = os.path.join("outputs", "generated_ac_cvs.csv")
OUTPUT_PATH = os.path.join("outputs", "generated_ac_cvs_sorted.csv")
EXPECTED_BASES = 84 # Total number of base CVs generated (28 per industry)
STARTING_ID = 7 # Since teammate's pilots were IDs 1-6

def main():
    print("--- Cover Letter Dataset Diagnostics ---\n")
    
    if not os.path.exists(INPUT_PATH):
        # Fallback to local directory if not run from root
        if os.path.exists("generated_ac_cvs.csv"):
            df = pd.read_csv("generated_ac_cvs.csv")
        else:
            return print(f"Error: Could not find {INPUT_PATH}")
    else:
        df = pd.read_csv(INPUT_PATH)
        
    print(f"Total Rows Found: {len(df)}")
    
    # 1. Clean Duplicates (Just in case the script restarted on the same ID)
    initial_len = len(df)
    df = df.drop_duplicates(subset=['Base_CV_ID', 'Framing'], keep='last')
    if initial_len != len(df):
        print(f"Removed {initial_len - len(df)} accidental duplicate generations.")
        
    # 2. Check Completeness
    expected_ids = set(range(STARTING_ID, STARTING_ID + EXPECTED_BASES))
    
    agentic_generated = set(df[df['Framing'] == 'Agentic']['Base_CV_ID'])
    communal_generated = set(df[df['Framing'] == 'Communal']['Base_CV_ID'])
    
    missing_agentic = expected_ids - agentic_generated
    missing_communal = expected_ids - communal_generated
    
    if len(missing_agentic) == 0 and len(missing_communal) == 0:
        print("\n✅ DATASET IS 100% COMPLETE! All 168 experimental CVs generated.")
    else:
        print("\n⚠️ DATASET IS INCOMPLETE.")
        print(f"Missing Agentic CVs: {len(missing_agentic)}")
        if len(missing_agentic) > 0:
            print(f"  -> Missing Base IDs: {sorted(list(missing_agentic))}")
            
        print(f"Missing Communal CVs: {len(missing_communal)}")
        if len(missing_communal) > 0:
            print(f"  -> Missing Base IDs: {sorted(list(missing_communal))}")
            
        print("\nTo finish the dataset, simply run `python3 scripts/generate_cl_framed.py` again. It will automatically skip the ones you already have and only generate the missing ones!")

    # 3. Sort the Dataset
    # Sort order: 1. Industry (Field), 2. Base ID, 3. Framing (Agentic then Communal)
    df_sorted = df.sort_values(by=['Base_CV_ID', 'Framing'], ascending=[True, True]).reset_index(drop=True)
    
    # 4. Save the Cleaned/Sorted Version
    df_sorted.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved perfectly sorted dataset to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()