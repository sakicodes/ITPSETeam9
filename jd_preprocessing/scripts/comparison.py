import os
import pandas as pd
import numpy as np
import re

ORIGINAL_PATH = os.path.join("data", "raw", "input.csv")
GENERATED_PATH = os.path.join("outputs", "experimental_jds.csv")
OUTPUT_PATH = os.path.join("outputs", "comparison.csv")

def parse_generated_jd(text):
    text = str(text)
    word_count = len(text.split())
    
    parts = text.split("**Responsibilities:**")
    resp_part = parts[1].split("**Requirements:**")[0] if len(parts) > 1 else ""
    
    req_part, comp_part = "", ""
    if "**Requirements:**" in text:
        req_split = text.split("**Requirements:**")[1]
        req_part = req_split.split("**Competencies:**")[0] if "**Competencies:**" in req_split else req_split
        
    if "**Competencies:**" in text:
        comp_part = text.split("**Competencies:**")[1]

    resp_bullets = resp_part.count("- ")
    req_bullets = req_part.count("- ")
    comp_bullets = comp_part.count("- ")

    return {
        'Word_Count': word_count,
        'Total_Bullets': resp_bullets + req_bullets + comp_bullets,
        'DF_Length_Bullets': resp_bullets + req_bullets,
        'CV_Length_Bullets': comp_bullets,
        'Resp_Word_Count': len(resp_part.split()),
        'Req_Word_Count': len(req_part.split()),
        'Comp_Word_Count': len(comp_part.split())
    }

def parse_original_jd(text):
    text = str(text)
    word_count = len(text.split())
    
    # Upgraded bullet logic to catch weird formatting
    bullet_pattern = r'(?m)^[ \t]*[-•*·oO➢]+[ \t]+|^[ \t]*\d+[\.\)][ \t]+'
    total_bullets = len(re.findall(bullet_pattern, text))
    
    resp_match = re.search(r'(?i)(responsibilities|what you will do|duties|the role)(.*?)(requirements|qualifications|skills|experience|$)', text, re.DOTALL)
    req_match = re.search(r'(?i)(requirements|qualifications|experience)(.*?)(skills|competencies|benefits|about us|$)', text, re.DOTALL)
    comp_match = re.search(r'(?i)(skills|competencies|traits|attributes)(.*?)(benefits|about us|equal opportunity|$)', text, re.DOTALL)

    resp_text = resp_match.group(2) if resp_match else ""
    req_text = req_match.group(2) if req_match else ""
    comp_text = comp_match.group(2) if comp_match else ""

    resp_bullets = len(re.findall(bullet_pattern, resp_text))
    req_bullets = len(re.findall(bullet_pattern, req_text))
    comp_bullets = len(re.findall(bullet_pattern, comp_text))

    if total_bullets == 0:
        resp_bullets = len([s for s in resp_text.split('.') if len(s.strip()) > 5])
        req_bullets = len([s for s in req_text.split('.') if len(s.strip()) > 5])
        comp_bullets = len([s for s in comp_text.split('.') if len(s.strip()) > 5])
        total_bullets = resp_bullets + req_bullets + comp_bullets

    return {
        'Word_Count': word_count,
        'Total_Bullets': total_bullets,
        'DF_Length_Bullets': resp_bullets + req_bullets,
        'CV_Length_Bullets': comp_bullets,
        'Resp_Word_Count': len(resp_text.split()),
        'Req_Word_Count': len(req_text.split()),
        'Comp_Word_Count': len(comp_text.split())
    }

def main():
    print("Running Dataset Comparison...\n")
    if not os.path.exists(GENERATED_PATH) or not os.path.exists(ORIGINAL_PATH):
        return print("Error: Missing input files.")

    df_gen = pd.read_csv(GENERATED_PATH)
    gen_metrics = pd.DataFrame(df_gen['Full_Job_Description'].apply(parse_generated_jd).tolist())
    
    df_orig = pd.read_csv(ORIGINAL_PATH)
    
    # Auto-detect text column
    text_col = next((col for col in ['Job Description', 'Description', 'description', 'text', 'Text'] if col in df_orig.columns), None)
    if not text_col:
        text_col = max(df_orig.columns, key=lambda c: df_orig[c].astype(str).str.len().mean())

    orig_metrics = pd.DataFrame(df_orig[text_col].apply(parse_original_jd).tolist())
    orig_metrics = orig_metrics[orig_metrics['Word_Count'] > 50] 

    def get_means(metrics_df):
        return {
            "Avg Overall Words": round(metrics_df['Word_Count'].mean(), 1),
            "Avg Total Bullets": round(metrics_df['Total_Bullets'].mean(), 1),
            "Avg DF Length (Bullets)": round(metrics_df['DF_Length_Bullets'].mean(), 1),
            "Avg CV Length (Bullets)": round(metrics_df['CV_Length_Bullets'].mean(), 1),
            "Avg Responsibilities Words": round(metrics_df['Resp_Word_Count'].mean(), 1),
            "Avg Requirements Words": round(metrics_df['Req_Word_Count'].mean(), 1),
            "Avg Competencies Words": round(metrics_df['Comp_Word_Count'].mean(), 1),
        }

    comparison_df = pd.DataFrame({
        "Original Corpus": get_means(orig_metrics),
        "Generated JDs": get_means(gen_metrics)
    })
    
    print(comparison_df.to_string())
    comparison_df.to_csv(OUTPUT_PATH)
    print(f"\nSaved empirical comparison to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()