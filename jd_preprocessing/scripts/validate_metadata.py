import os
import pandas as pd

INPUT_PATH = os.path.join("data", "raw", "input.csv")
CLEAN_JDS_PATH = os.path.join("data", "processed", "clean_jds.csv")
METADATA_LOG_PATH = os.path.join("outputs", "metadata_validation.csv")

INDUSTRIES = {
    "Sales / Business Development": ["sales", "account executive", "business development", "territory", "client acquisition", "pipeline", "negotiation", "customer relationship"],
    "Procurement / Sourcing / Supply Chain": ["procurement", "buyer", "category manager", "supplier", "vendor", "contract", "strategic sourcing", "purchasing", "supply chain"],
    "Project Management / Programme Management": ["project manager", "programme manager", "pmo", "delivery", "milestone", "project planning", "stakeholder management", "risk management"]
}

def calculate_industry_score(row):
    best_industry = "Unknown"
    highest_score = 0
    
    func = str(row.get('Job_Function', '')).lower()
    title = str(row.get('Job_Title', '')).lower()
    desc = str(row.get('Description', '')).lower()
    
    for industry, keywords in INDUSTRIES.items():
        score = 0
        if any(kw in func for kw in keywords): score += 3
        if any(kw in title for kw in keywords): score += 2
        if any(kw in desc for kw in keywords): score += 1
        
        if score > highest_score:
            highest_score = score
            best_industry = industry
            
    return best_industry, highest_score

def main():
    print("Stage 1: Validating Metadata & Classifying Industries...")
    if not os.path.exists(INPUT_PATH): return print(f"Error: {INPUT_PATH} missing.")
    
    df = pd.read_csv(INPUT_PATH)
    
    # ---------------------------------------------------------
    # FIX: Resolve duplicate Company columns before renaming
    # ---------------------------------------------------------
    if 'companyName' in df.columns and 'Company' in df.columns:
        df['companyName'] = df['companyName'].fillna(df['Company'])
        df = df.drop(columns=['Company'])
    
    # Standardise column names safely
    df = df.rename(columns={
        'jobTitle': 'Job_Title', 'companyName': 'Company', 'location': 'Country',
        'jobFunction': 'Job_Function', 'seniorityLevel': 'Seniority', 
        'employmentType': 'Employment_Type', 'description': 'Description'
    })
    
    # Fill missing with 'Unknown'
    fill_cols = ['Job_Title', 'Company', 'Country', 'Job_Function', 'Seniority', 'Employment_Type', 'Source', 'Link']
    for col in fill_cols:
        if col in df.columns: df[col] = df[col].fillna('Unknown')
        else: df[col] = 'Unknown'
        
    df.insert(0, 'Record_ID', [f"JDRAW_{str(i).zfill(5)}" for i in range(1, len(df) + 1)])
    
    logs = []
    clean_rows = []
    
    for _, row in df.iterrows():
        desc_present = pd.notna(row.get('Description')) and str(row.get('Description')).strip() != ""
        title_func_present = (row['Job_Title'] != 'Unknown') or (row['Job_Function'] != 'Unknown')
        
        # Quality Score
        q_score = 0
        if desc_present: q_score += 1
        if row['Job_Title'] != 'Unknown': q_score += 1
        if row['Job_Function'] != 'Unknown': q_score += 1
        if row['Country'] != 'Unknown': q_score += 1
        if row['Company'] != 'Unknown': q_score += 1
        
        # Industry Score
        suggested_ind, ind_score = calculate_industry_score(row)
        
        # Decision Logic
        decision = "Discard"
        reason = "Missing critical fields"
        
        if desc_present and title_func_present:
            if ind_score >= 3:
                decision = "Keep"
                reason = "Passed all criteria"
                row_dict = row.to_dict()
                row_dict['Industry'] = suggested_ind
                clean_rows.append(row_dict)
            else:
                reason = f"Industry score too low ({ind_score})"
                
        logs.append({
            "Record_ID": row['Record_ID'], "Job_Title": row['Job_Title'], "Job_Function": row['Job_Function'],
            "Description_Present": desc_present, "Industry_Score": ind_score, "Suggested_Industry": suggested_ind,
            "Decision": decision, "Reason": reason, "Quality_Score": q_score
        })

    pd.DataFrame(logs).to_csv(METADATA_LOG_PATH, index=False)
    
    clean_df = pd.DataFrame(clean_rows)
    export_cols = ['Record_ID', 'Job_Title', 'Company', 'Country', 'Industry', 'Job_Function', 'Seniority', 'Employment_Type', 'Description', 'Source', 'Link']
    clean_df = clean_df[[c for c in export_cols if c in clean_df.columns]]
    clean_df.to_csv(CLEAN_JDS_PATH, index=False)
    
    print(f"Retained {len(clean_df)} valid JDs out of {len(df)}.")

if __name__ == "__main__":
    main()