import pandas as pd
import re
import json

# Load Excel
file_path = "generated_base_cvs.xlsx"
df = pd.read_excel(file_path, sheet_name="generated_base_cvs")
dict_df = pd.read_excel(file_path, sheet_name="Dictionary")

# Clean dictionary columns
agentic_words = dict_df["Agentic"].dropna().astype(str).str.replace(r"\*", "", regex=True).str.lower().tolist()
communal_words = dict_df["Communal"].dropna().astype(str).str.replace(r"\*", "", regex=True).str.lower().tolist()

def find_matches(text, dictionary):
    if pd.isna(text):
        return []
    words = re.findall(r"\b\w+\b", str(text).lower())
    matches = []
    for i, w in enumerate(words):
        for d in dictionary:
            if d in w:  # substring match
                matches.append({"position": i + 1, "word": w, "dict_term": d})
                break
    return matches

# Apply matching and store JSON
df["Agentics"] = df["Cover_Letter"].apply(lambda x: json.dumps(find_matches(x, agentic_words), ensure_ascii=False))
df["Communals"] = df["Cover_Letter"].apply(lambda x: json.dumps(find_matches(x, communal_words), ensure_ascii=False))

# Save results
output_path = "cv_base_matches_output2.xlsx"
with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
    df.to_excel(writer, index=False, sheet_name="generated_base_cvs")
    dict_df.to_excel(writer, index=False, sheet_name="Dictionary")

print(f"Done! Results saved to {output_path}")
