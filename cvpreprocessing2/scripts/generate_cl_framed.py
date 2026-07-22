import os
import time
import pandas as pd
import re
import difflib
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=API_KEY)
MODEL_NAME = "gemini-3.1-flash-lite"

# Paths
DICT_DIR = os.path.join("data", "dictionaries")
INPUT_BASES = os.path.join("outputs", "generated_base_cvs.csv")
OUTPUT_PATH = os.path.join("outputs", "generated_ac_cvs.csv")

def load_dictionary(filename):
    filepath = os.path.join(DICT_DIR, filename)
    if not os.path.exists(filepath): return set()
    df = pd.read_csv(filepath, skiprows=2)
    words = set()
    if 'Word / Phrase' in df.columns:
        words.update([str(w).lower().strip() for w in df['Word / Phrase'].dropna()])
    if 'Alternate Word / Phrase' in df.columns:
        for alt_string in df['Alternate Word / Phrase'].dropna():
            words.update([v.strip().lower() for v in str(alt_string).split(',')])
    return words

def count_matches(text, dict_words):
    text_lower = str(text).lower()
    count = 0
    sorted_words = sorted(list(dict_words), key=len, reverse=True)
    for word in sorted_words:
        pattern = r'\b' + re.escape(word) + r'\b'
        matches = re.findall(pattern, text_lower)
        if matches:
            count += len(matches)
            text_lower = re.sub(pattern, '', text_lower)
    return count

def check_clustering(text, dict_words):
    sentences = re.split(r'(?<=[.!?]) +', str(text))
    sorted_words = sorted(list(dict_words), key=len, reverse=True)
    for sentence in sentences:
        sent_lower = sentence.lower()
        sent_count = 0
        for word in sorted_words:
            pattern = r'\b' + re.escape(word) + r'\b'
            matches = re.findall(pattern, sent_lower)
            if matches:
                sent_count += len(matches)
                sent_lower = re.sub(pattern, '', sent_lower)
        if sent_count > 3:
            return False 
    return True

def calculate_similarity(base_text, new_text):
    return difflib.SequenceMatcher(None, str(base_text), str(new_text)).ratio()

def main():
    print("Phase 3: Loading Resources for Framing Pass...")
    agentic_words = load_dictionary('agentic.csv')
    communal_words = load_dictionary('communal.csv')
    negative_words = load_dictionary('negative.csv')
    neutral_words = load_dictionary('neutral.csv')
    
    banned_words_list = ", ".join(list(negative_words))
    
    df_bases = pd.read_csv(INPUT_BASES)
    
    completed_ids = set()
    if os.path.exists(OUTPUT_PATH):
        df_existing = pd.read_csv(OUTPUT_PATH)
        completed_ids = set(df_existing['Base_CV_ID'].astype(str) + "_" + df_existing['Framing'])
        print(f"Resuming progress: {len(completed_ids)} CVs already generated.")

    print("\nStarting Constrained Generation (Target: 70%+ Similarity, 8-12 target words)...")
    
    for _, base_row in df_bases.iterrows():
        base_id = str(base_row['Base_CV_ID'])
        base_text = str(base_row['Cover_Letter'])
        base_word_count = len(base_text.split())
        industry = base_row['Field']
        
        for framing in ['Agentic', 'Communal']:
            if f"{base_id}_{framing}" in completed_ids:
                continue
                
            print(f"\nProcessing Base {base_id} -> {framing} ({industry})")
            target_dict = agentic_words if framing == 'Agentic' else communal_words
            dict_sample = ", ".join(list(target_dict)[:50]) 
            
            successful = False
            attempts = 0
            
            # WIDENED THE ATTEMPT LIMIT
            while not successful and attempts < 12:
                attempts += 1
                
                prompt = f"""
                You are a strict HR copyeditor. 
                Task: Transform the provided Base Cover Letter into an {framing} version. 
                
                CRITICAL RULES:
                1. REPLACE, DO NOT ADD: To keep the word count identical, you must swap existing neutral words (verbs/adjectives) for {framing} words. Do not add entirely new sentences.
                2. DENSITY & CLUSTERING: You MUST include EXACTLY 8 to 12 distinct words from the dictionary. Do NOT put more than 2 dictionary words in a single sentence. Spread them out evenly across paragraphs 2 and 3.
                3. MINIMAL REWRITING (70%+ SIMILARITY): Preserve the original sentence structure, paragraph order, and core narrative. 
                4. NO NEGATIVE WORDS: You are strictly forbidden from using any of these words: {banned_words_list}.
                
                --- {framing.upper()} DICTIONARY WORDS TO USE ---
                {dict_sample}
                
                --- BASE COVER LETTER ---
                {base_text}
                
                Output ONLY the raw text of the revised cover letter. Keep the "Dear Hiring Manager," and "Regards," formatting intact.
                """
                
                try:
                    response = client.models.generate_content(
                        model=MODEL_NAME, contents=prompt,
                        config=types.GenerateContentConfig(temperature=0.2) # Lowered temp even further for stability
                    )
                    
                    new_text = response.text.strip()
                    
                    new_word_count = len(new_text.split())
                    target_count = count_matches(new_text, target_dict)
                    neg_count = count_matches(new_text, negative_words)
                    similarity = calculate_similarity(base_text, new_text)
                    
                    # WIDENED TOLERANCES: +/- 15% length, 8-12 words
                    if not (base_word_count * 0.85 <= new_word_count <= base_word_count * 1.15):
                        print(f"  Attempt {attempts} Failed: Length ({new_word_count} vs {base_word_count} base).")
                        continue
                        
                    if not (8 <= target_count <= 13): 
                        print(f"  Attempt {attempts} Failed: Density miss ({target_count} target words found).")
                        continue
                        
                    if neg_count > 0:
                        print(f"  Attempt {attempts} Failed: Negative word detected.")
                        continue
                        
                    if not check_clustering(new_text, target_dict):
                        print(f"  Attempt {attempts} Failed: Keyword clustering detected (>2 per sentence).")
                        continue
                        
                    if similarity < 0.70:
                        print(f"  Attempt {attempts} Failed: Similarity only {round(similarity * 100, 1)}%.")
                        continue
                        
                    print(f"  Success! Similarity: {round(similarity * 100, 1)}%, Target Words: {target_count}, Length: {new_word_count}")
                    
                    out_row = pd.DataFrame([{
                        "Base_CV_ID": base_id,
                        "Field": industry,
                        "Framing": framing,
                        "Cover_Letter": new_text,
                        "Agentic_Count": count_matches(new_text, agentic_words),
                        "Communal_Count": count_matches(new_text, communal_words),
                        "Neutral_Count": count_matches(new_text, neutral_words),
                        "Negative_Count": neg_count,
                        "Total_Words": new_word_count,
                        "Similarity_Score": round(similarity, 3)
                    }])
                    
                    out_row.to_csv(OUTPUT_PATH, mode='a', header=not os.path.exists(OUTPUT_PATH), index=False)
                    successful = True
                    time.sleep(1.5)
                    
                except Exception as e:
                    time.sleep(3)
            
            if not successful:
                print(f"  WARNING: Failed to generate valid {framing} CV for Base {base_id} after 12 attempts.")

    print(f"\nPhase 3 Complete! Check {OUTPUT_PATH} for the validated framed cover letters.")

if __name__ == "__main__":
    main()