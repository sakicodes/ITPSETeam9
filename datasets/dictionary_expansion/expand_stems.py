import pandas as pd

# Load the corpus vocabulary from Step 1
vocab = pd.read_csv("datasets/corpus_vocabulary.csv")

# Load the competence dictionary (check column names first - open the
# file to see what they're actually called, e.g. 'Communal' and 'Agentic')
comp_dict = pd.read_excel("datasets/competence_dictionary.xlsx")

print(comp_dict.columns)

# For each column, go through entries ending in '*' and find matches
for col in comp_dict.columns:
    for term in comp_dict[col].dropna():
        term = str(term).strip()
        if term.endswith('*'):
            stem = term[:-1].lower()
            matches = vocab[vocab['word'].str.startswith(stem)]['word'].tolist()
            print(f"{term}  ->  {matches}")

results = []
for col in comp_dict.columns:
    for term in comp_dict[col].dropna():
        term = str(term).strip()
        if term.endswith('*'):
            stem = term[:-1].lower()
            matches = vocab[vocab['word'].str.startswith(stem)]
            for _, row in matches.iterrows():
                results.append({
                    'category': col,
                    'stem': term,
                    'matched_word': row['word'],
                    'count': row['count'],
                    'keep': ''   # you'll fill this in manually
                })

out = pd.DataFrame(results)
out.to_csv("datasets/stem_expansion_review.csv", index=False)
print(f"Wrote {len(out)} candidate words to datasets/stem_expansion_review.csv")