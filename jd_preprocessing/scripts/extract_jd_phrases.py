import os
import regex as re
import pandas as pd
import nltk
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize, word_tokenize

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)
    nltk.download('punkt', quiet=True)

INPUT_PATH = os.path.join("data", "processed", "clean_jds.csv")
OUTPUT_PATH = os.path.join("data", "processed", "jd_raw_phrases.csv")
SPLIT_PATTERN = re.compile(r'(?:^|\n)\s*[\u2022\u2023\u25E6\u2043\-\*]\s*|;')

def clean_html(text):
    if pd.isna(text): return ""
    return BeautifulSoup(str(text), "html.parser").get_text(separator=" ")

def main():
    print("Stage 2: Extracting Phrases...")
    if not os.path.exists(INPUT_PATH): return print("Clean JDs not found.")
    
    df = pd.read_csv(INPUT_PATH)
    phrases = []
    
    for _, row in df.iterrows():
        text = clean_html(row['Description'])
        blocks = SPLIT_PATTERN.split(text)
        
        for block in blocks:
            if not block.strip(): continue
            sentences = sent_tokenize(block)
            
            for sentence in sentences:
                sentence = re.sub(r'\s+', ' ', sentence).strip()
                word_count = len([w for w in word_tokenize(sentence) if w.isalnum()])
                
                if 5 <= word_count <= 35:
                    phrases.append({
                        "Source_Record_ID": row['Record_ID'], "Source": row.get('Source', 'Unknown'),
                        "Company": row['Company'], "Country": row['Country'], "Industry": row['Industry'],
                        "Job_Function": row['Job_Function'], "Position": row['Job_Title'], 
                        "Seniority": row['Seniority'], "Phrase": sentence, "Word_Count": word_count
                    })
                    
    phrases_df = pd.DataFrame(phrases)
    phrases_df = phrases_df.drop_duplicates(subset=['Phrase'])
    phrases_df.insert(0, 'Phrase_ID', [f"P_{str(i).zfill(6)}" for i in range(1, len(phrases_df) + 1)])
    
    phrases_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Extracted {len(phrases_df)} candidate phrases.")

if __name__ == "__main__":
    main()