import os
import time
import pandas as pd
import re
from google import genai
from google.genai import types
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=API_KEY)
MODEL_NAME = "gemini-3.1-flash-lite"

# Paths
DICT_DIR = os.path.join("data", "dictionaries")
OUTPUT_PATH = os.path.join("outputs", "generated_base_cvs.csv")

INDUSTRIES = [
    "Sales / Business Development", 
    "Procurement / Sourcing / Supply Chain", 
    "Project Management / Programme Management"
]
BASES_PER_INDUSTRY = 28

def load_dictionary(filename):
    """Loads dictionary, skipping the first two title rows, and extracts all word variants."""
    filepath = os.path.join(DICT_DIR, filename)
    if not os.path.exists(filepath):
        print(f"Warning: {filename} not found.")
        return set()
    
    # skiprows=2 handles the messy Excel-to-CSV title formatting perfectly
    df = pd.read_csv(filepath, skiprows=2)
    words = set()
    
    if 'Word / Phrase' in df.columns:
        words.update([str(w).lower().strip() for w in df['Word / Phrase'].dropna()])
        
    if 'Alternate Word / Phrase' in df.columns:
        alts = df['Alternate Word / Phrase'].dropna()
        for alt_string in alts:
            variants = [v.strip().lower() for v in str(alt_string).split(',')]
            words.update(variants)
            
    return words

def count_dictionary_matches(text, dictionary_words):
    """Counts how many times words from a specific dictionary appear in the text."""
    text_lower = str(text).lower()
    count = 0
    # Sort by length descending so we match "taking initiative" before "initiative"
    sorted_words = sorted(list(dictionary_words), key=len, reverse=True)
    
    for word in sorted_words:
        # Using regex word boundaries to avoid matching "art" inside "participant"
        pattern = r'\b' + re.escape(word) + r'\b'
        matches = re.findall(pattern, text_lower)
        if matches:
            count += len(matches)
            # Remove the matched word from text to prevent double-counting overlaps
            text_lower = re.sub(pattern, '', text_lower) 
            
    return count

def main():
    print("Phase 1: Loading Dictionaries...")
    agentic_words = load_dictionary('agentic.csv')
    communal_words = load_dictionary('communal.csv')
    negative_words = load_dictionary('negative.csv')
    neutral_words = load_dictionary('neutral.csv')
    
    combined_ac_words = agentic_words.union(communal_words)
    banned_words_list = ", ".join(list(negative_words) + list(combined_ac_words)[:50]) # Sample for prompt
    
    print(f"Loaded: {len(agentic_words)} Agentic, {len(communal_words)} Communal, {len(negative_words)} Negative variants.")
    
    final_bases = []
    base_id = 7 # Starting at 7 since Victoria used 1-6 in her pilots

    print("\nPhase 2: Generating 84 Neutral Base Cover Letters...")
    for industry in INDUSTRIES:
        print(f"\nGenerating for {industry}...")
        
        successful_bases = 0
        attempts = 0
        
        pbar = tqdm(total=BASES_PER_INDUSTRY)
        
        while successful_bases < BASES_PER_INDUSTRY and attempts < (BASES_PER_INDUSTRY * 3):
            attempts += 1
            
            prompt = f"""
            Write a strictly NEUTRAL, highly professional 4-paragraph cover letter for a standard {industry} role.
            
            STRUCTURE:
            1. Statement of intent (applying for the role).
            2. Description of past experience.
            3. Examples of standard skills.
            4. Call to action.
            
            RULES:
            - ANONYMIZATION: Do not use real company names or cities. Use '[Company]' or '[Location]'.
            - TONE: Must be incredibly plain, factual, and neutral. 
            - FORBIDDEN WORDS: You are strictly forbidden from using arrogant/negative words, OR highly expressive words. Do not use words like: {banned_words_list}.
            
            Output ONLY the raw text of the cover letter. No markdown formatting, no introductions.
            """
            
            try:
                response = client.models.generate_content(
                    model=MODEL_NAME, contents=prompt,
                    config=types.GenerateContentConfig(temperature=0.7) # Slightly higher temp to ensure variety across the 28 bases
                )
                
                cl_text = response.text.strip()
                
                # --- The Validation Gate ---
                ac_count = count_dictionary_matches(cl_text, combined_ac_words)
                negative_count = count_dictionary_matches(cl_text, negative_words)
                word_count = len(cl_text.split())
                
                if negative_count > 0:
                    continue # Failed: Negative word detected
                    
                if ac_count > 6:
                    continue # Failed: Too expressive/accidentally framed
                    
                if not (120 <= word_count <= 250):
                    continue # Failed: Outside empirical length norms
                
                # Passed all checks!
                final_bases.append({
                    "Base_CV_ID": base_id,
                    "Field": industry,
                    "Cover_Letter": cl_text,
                    "Agentic_Count": count_dictionary_matches(cl_text, agentic_words),
                    "Communal_Count": count_dictionary_matches(cl_text, communal_words),
                    "Neutral_Count": count_dictionary_matches(cl_text, neutral_words),
                    "Negative_Count": negative_count,
                    "Total_Words": word_count
                })
                
                base_id += 1
                successful_bases += 1
                pbar.update(1)
                time.sleep(1) # API rate limit safety
                
            except Exception as e:
                time.sleep(3)
                
        pbar.close()

    df_out = pd.DataFrame(final_bases)
    df_out.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSuccessfully saved {len(df_out)} Base CVs to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()