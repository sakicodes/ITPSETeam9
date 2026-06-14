# ============================================================
# merge_dictionaries.py
#
# Step 2c (final) of the dictionary expansion pipeline.
# Takes the manually-reviewed stem_expansion_review.csv
# (rows marked keep == 'y'), assigns each new word a
# sub-category based on the stem it came from, removes words
# already in the dictionaries plus known typos/artifacts
# (EXCLUDE_WORDS), and produces the final expanded dictionaries.
#
# Input:  dictionaries/agentic.csv, dictionaries/communal.csv
#         stem_expansion_review.csv
# Output: agentic_expanded.csv, communal_expanded.csv
#         (ready to copy into dictionaries/ once reviewed)
# ============================================================

import pandas as pd

# ---- Stem -> sub-category mappings ----
AGENTIC_STEM_CATEGORY = {
    'job': 'workstyle', 'project': 'workstyle', 'master': 'expertise', 'compl': 'workstyle',
    'learn': 'expertise', 'solution': 'expertise', 'write': 'communication', 'qualit': 'expertise',
    'issue': 'workstyle', 'win': 'personality', 'degree': 'expertise', 'operate': 'workstyle',
    'deliver': 'workstyle', 'knowledge': 'expertise', 'sell': 'workstyle', 'implement': 'workstyle',
    'education': 'expertise', 'result': 'workstyle', 'legislation': 'expertise', 'retail': 'workstyle',
    'initiative': 'leadership', 'problem': 'expertise', 'software': 'expertise', 'forecast': 'expertise',
    'service': 'workstyle',  # placeholder - team to confirm, see note below
    'task': 'workstyle', 'performance': 'workstyle', 'achieve': 'personality', 'order': 'workstyle',
    'presentation': 'communication', 'program': 'workstyle', 'execute': 'workstyle', 'success': 'personality',
    'self': 'personality', 'deal': 'workstyle', 'report': 'communication', 'promote': 'leadership',
    'active': 'personality', 'aggress': 'personality', 'ambitio': 'personality', 'analy': 'expertise',
    'assert': 'personality', 'athlet': 'personality', 'autonom': 'personality', 'challeng': 'personality',
    'compet': 'personality', 'courag': 'personality', 'decide': 'leadership', 'decisive': 'leadership',
    'decision': 'leadership', 'determin': 'personality', 'domina': 'personality', 'force': 'personality',
    'independen': 'personality', 'individual': 'personality', 'intellect': 'expertise', 'lead': 'leadership',
    'logic': 'expertise', 'objective': 'workstyle', 'principle': 'personality', 'aggressive': 'personality',
    'risk': 'leadership', 'profit': 'workstyle', 'target': 'workstyle', 'account': 'workstyle',
    'cultivat': 'leadership',
}

COMMUNAL_STEM_CATEGORY = {
    'customer': 'service', 'contact': 'service', 'warm': 'quality', 'member': 'teamwork',
    'support': 'teamwork', 'staff': 'teamwork', 'client': 'service', 'communicate': 'service',
    'help': 'service', 'role': 'teamwork', 'follow': 'teamwork', 'share': 'teamwork',
    'partner': 'teamwork', 'child': 'quality', 'cheer': 'quality', 'commit': 'quality',
    'compassion': 'quality', 'connect': 'teamwork', 'cooperat': 'teamwork', 'depend': 'quality',
    'emotiona': 'quality', 'empath': 'quality', 'gentle': 'quality', 'interdependen': 'teamwork',
    'interpersona': 'service', 'loyal': 'quality', 'modest': 'quality', 'nurtur': 'quality',
    'pleasant': 'quality', 'quiet': 'quality', 'respon': 'quality', 'sensitiv': 'quality',
    'sympath': 'quality', 'tender': 'quality', 'together': 'teamwork', 'trust': 'quality',
    'understand': 'quality', 'yield': 'quality', 'care': 'quality', 'sooth': 'quality',
    'love': 'quality', 'sincere': 'quality',
    'develop': 'service',  # placeholder - team to confirm, see note below
    'build': 'teamwork',
}

# Words to exclude entirely (typos, brand names, opposite/unrelated meaning)
EXCLUDE_WORDS = {
    'selfless', 'complaints', 'qualit', 'qualitfication', "deal'", "learn'",
    'shareholder', 'shareholders', 'helpdesk', 'staffmark',
    'childbirth', 'childhood', 'committement',
}

# ---- Load existing dictionaries ----
agentic = pd.read_csv("datasets/dictionaries/agentic.csv")
communal = pd.read_csv("datasets/dictionaries/communal.csv")

# ---- Load review results, keep only approved words ----
review = pd.read_csv("datasets/dictionary_expansion/stem_expansion_review.csv")
kept = review[review['keep'] == 'y'].copy()

# Drop excluded words
kept = kept[~kept['matched_word'].str.lower().isin(EXCLUDE_WORDS)]

# Normalize stem (strip trailing '*', lowercase)
kept['stem_clean'] = kept['stem'].str.rstrip('*').str.lower()

def get_category(row):
    mapping = AGENTIC_STEM_CATEGORY if row['category'] == 'Agentic' else COMMUNAL_STEM_CATEGORY
    return mapping.get(row['stem_clean'])

kept['sub_category'] = kept.apply(get_category, axis=1)

unmapped = kept[kept['sub_category'].isna()]['stem'].unique()
if len(unmapped) > 0:
    print("WARNING - unmapped stems (need adding to dictionary):", unmapped)

agentic_kept = kept[kept['category'] == 'Agentic']
communal_kept = kept[kept['category'] == 'Communal']

agentic_word_cat = agentic_kept.groupby('matched_word')['sub_category'].first()
communal_word_cat = communal_kept.groupby('matched_word')['sub_category'].first()

# Remove anything already in the dictionary
existing_agentic = set(agentic['keyword'].str.lower())
existing_communal = set(communal['keyword'].str.lower())

agentic_new = agentic_word_cat[~agentic_word_cat.index.str.lower().isin(existing_agentic)]
communal_new = communal_word_cat[~communal_word_cat.index.str.lower().isin(existing_communal)]

new_agentic = pd.DataFrame({'keyword': agentic_new.index, 'category': agentic_new.values})
new_communal = pd.DataFrame({'keyword': communal_new.index, 'category': communal_new.values})

agentic_final = pd.concat([agentic, new_agentic], ignore_index=True)
communal_final = pd.concat([communal, new_communal], ignore_index=True)

agentic_final.to_csv("datasets/dictionary_expansion/agentic_expanded.csv", index=False)
communal_final.to_csv("datasets/dictionary_expansion/communal_expanded.csv", index=False)

print(f"agentic.csv: {len(agentic)} -> {len(agentic_final)} rows ({len(new_agentic)} new)")
print(f"communal.csv: {len(communal)} -> {len(communal_final)} rows ({len(new_communal)} new)")
print("\nNew agentic words by sub-category:")
print(new_agentic['category'].value_counts())
print("\nNew communal words by sub-category:")
print(new_communal['category'].value_counts())