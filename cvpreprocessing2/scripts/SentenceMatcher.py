import pandas as pd
import re
from rapidfuzz import fuzz

# Load Excel
file_path = "generated_ac_cvs_final_pt3.xlsx"
base_df = pd.read_excel(file_path, sheet_name="BaseUP")
ac_df = pd.read_excel(file_path, sheet_name="Final")

# Function to split text into sentences (basic split on .!?)
def split_sentences(text):
    if pd.isna(text):
        return []
    # Split on punctuation followed by space
    sentences = re.split(r'(?<=[.!?])\s+', str(text).strip())
    return [s for s in sentences if s]

# Function to compare sentences one by one
def compare_letters(base_text, target_text):
    base_sents = split_sentences(base_text)
    target_sents = split_sentences(target_text)
    scores = []

    i, j = 0, 0
    while i < len(base_sents) and j < len(target_sents):
        score = fuzz.ratio(base_sents[i], target_sents[j])

        if score < 50:
            # Flag as extra line from whichever text is longer
            if len(base_sents) > len(target_sents):
                scores.append({
                    "sentence_index": i+1,
                    "base_sentence": base_sents[i],
                    "target_sentence": "",
                    "similarity_score": score,
                    "flag": "extra_base"
                })
                i += 1  # advance only base
            else:
                scores.append({
                    "sentence_index": j+1,
                    "base_sentence": "",
                    "target_sentence": target_sents[j],
                    "similarity_score": score,
                    "flag": "extra_target"
                })
                j += 1  # advance only target
        else:
            # Normal comparison
            if score < 70:
                scores.append({
                    "sentence_index": i+1,
                    "base_sentence": base_sents[i],
                    "target_sentence": target_sents[j],
                    "similarity_score": score,
                    "flag": "low_match"
                })
            i += 1
            j += 1

    # Handle leftovers if one text is longer
    while i < len(base_sents):
        scores.append({
            "sentence_index": i+1,
            "base_sentence": base_sents[i],
            "target_sentence": "",
            "similarity_score": 0,
            "flag": "extra_base"
        })
        i += 1

    while j < len(target_sents):
        scores.append({
            "sentence_index": j+1,
            "base_sentence": "",
            "target_sentence": target_sents[j],
            "similarity_score": 0,
            "flag": "extra_target"
        })
        j += 1

    return scores

# Apply comparison for each AC CV row
results = []
for _, row in ac_df.iterrows():
    base_id = row["Base_CV_ID"]
    target_text = row["Cover_Letter"]
    base_text = base_df.loc[base_df["Base_CV_ID"] == base_id, "Cover_Letter"].values
    if len(base_text) > 0:
        comparison = compare_letters(base_text[0], target_text)
        results.append(comparison)
    else:
        results.append([])

# Store JSON results in AC CV sheet
import json
ac_df["Similarity"] = [json.dumps(r, ensure_ascii=False) for r in results]

# Save output
output_path = "cv_similarity_output_final.xlsx"
with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
    ac_df.to_excel(writer, index=False, sheet_name="AC")
    base_df.to_excel(writer, index=False, sheet_name="Base")

print(f"Done! Results saved to {output_path}")
