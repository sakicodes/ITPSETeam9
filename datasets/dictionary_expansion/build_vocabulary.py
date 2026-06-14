# ============================================================
# build_vocabulary.py
#
# Step 1 of the dictionary expansion pipeline.
# Reads the cleaned/filtered job postings corpus and builds a
# word-frequency table (vocabulary) used as the basis for
# finding candidate words to expand the agentic/communal
# dictionaries.
#
# Input:  cleaned_filtered_jobs.csv
# Output: corpus_vocabulary.csv  (columns: word, count)
# ============================================================

import pandas as pd # type: ignore
import re
from collections import Counter

# Load the cleaned, filtered dataset from step 1
df = pd.read_csv("datasets/cleaned_filtered_jobs.csv")

# Count word frequencies across all descriptions
word_counts = Counter()
for text in df['description_clean'].dropna():
    tokens = re.findall(r"[a-z']+", text.lower())
    word_counts.update(tokens)

# Turn into a sorted DataFrame for easy viewing/saving
vocab = pd.DataFrame(word_counts.items(), columns=['word', 'count'])
vocab = vocab.sort_values('count', ascending=False).reset_index(drop=True)

print(f"Total unique words: {len(vocab):,}")
print(vocab.head(20))

vocab.to_csv("datasets/corpus_vocabulary.csv", index=False)