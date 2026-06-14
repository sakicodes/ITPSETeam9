# ============================================================
# auto_mark_stems.py
#
# Step 2b of the dictionary expansion pipeline.
# Pre-fills the 'keep' column in stem_expansion_review.csv:
#   - 'y' if the matched word is the stem + a standard
#     English suffix (plural, possessive, -ing, -ed, etc.)
#   - 'n' if the word occurs <= 2 times and isn't a standard
#     inflection (likely a typo or formatting artifact)
#   - left blank for manual review otherwise
#
# Input/Output: stem_expansion_review.csv (edited in place)
# ============================================================

import pandas as pd

df = pd.read_csv("datasets/stem_expansion_review.csv")

# Common suffixes that, when added to a full-word stem, still represent
# the same underlying concept (plurals, possessives, adverbs, etc.)
STANDARD_SUFFIXES = {
    '', 's', 'es', "'s", "s'", "'",
    'ing', 'ed', 'er', 'ers', 'or', 'ors',
    'ly', 'ally',
    'ness', 'ity', 'ities',
    'tion', 'tions', 'ment', 'ments',
    'al', 'ive', 'ives',
    'ful', 'less', 'able', 'ible',
}

def auto_mark(row):
    existing = str(row['keep']).strip().lower()
    if existing in ('y', 'n'):
        return existing  # don't overwrite anything you've already marked

    stem = row['stem'][:-1].lower()  # strip trailing '*'
    word = str(row['matched_word']).lower()
    suffix = word[len(stem):] if word.startswith(stem) else word

    if suffix in STANDARD_SUFFIXES:
        return 'y'
    elif row['count'] <= 2:
        return 'n'
    else:
        return ''  # leave for manual review

df['keep'] = df.apply(auto_mark, axis=1)
df.to_csv("datasets/stem_expansion_review.csv", index=False)

print(df['keep'].value_counts(dropna=False))
print(f"\n{(df['keep'] == '').sum()} rows still need manual review")