import os
import pandas as pd

DICT_DIR = os.path.join("data", "dictionaries")

def generate_variants(phrase):
    """Generates simple morphological variants for a given word/phrase."""
    phrase = str(phrase).lower().strip()
    words = phrase.split()
    
    # Only expand single words or the primary verb in very short phrases
    if len(words) > 2: return [phrase]
    
    target = words[0]
    variants = set([target])
    
    # Basic morphological rules
    if target.endswith('e'):
        variants.update([target + 's', target + 'd', target[:-1] + 'ing', target[:-1] + 'er', target[:-1] + 'tion'])
    elif target.endswith('y') and len(target) > 2 and target[-2] not in 'aeiou':
        variants.update([target[:-1] + 'ies', target[:-1] + 'ied', target + 'ing'])
    else:
        variants.update([target + 's', target + 'ed', target + 'ing', target + 'er', target + 'ment'])
        # Double consonant for short verbs (e.g., plan -> planning)
        if len(target) >= 3 and target[-1] not in 'aeiouy' and target[-2] in 'aeiou':
            variants.add(target + target[-1] + 'ing')
            variants.add(target + target[-1] + 'ed')
            
    # Reattach any second words if it was a 2-word phrase
    if len(words) == 2:
        return [f"{v} {words[1]}" for v in variants]
        
    return list(variants)

def expand_file(filename):
    path = os.path.join(DICT_DIR, filename)
    if not os.path.exists(path): return print(f"Missing {filename}")
    
    df = pd.read_csv(path)
    if 'Keyword / Phrase' not in df.columns: return print(f"Column missing in {filename}")
    
    expanded_rows = []
    for _, row in df.iterrows():
        base_word = row['Keyword / Phrase']
        sub_cat = row.get('Sub-category', '')
        
        variants = generate_variants(base_word)
        for v in variants:
            expanded_rows.append({'Keyword / Phrase': v, 'Sub-category': sub_cat})
            
    # Drop duplicates and save
    out_df = pd.DataFrame(expanded_rows).drop_duplicates(subset=['Keyword / Phrase'])
    out_path = os.path.join(DICT_DIR, filename.replace('.csv', '_expanded.csv'))
    out_df.to_csv(out_path, index=False)
    print(f"Expanded {filename}: {len(df)} -> {len(out_df)} terms.")

def main():
    print("Expanding Dictionaries...")
    for f in ['agentic.csv', 'communal.csv', 'neutral.csv']:
        expand_file(f)

if __name__ == "__main__":
    main()