import os
import regex as re
import pandas as pd
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

# Ensure NLTK punkt is downloaded
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)
    nltk.download('punkt', quiet=True)

# -------------------------------------------------------------------------
# CONFIGURATION & PATHS
# -------------------------------------------------------------------------
DATA_URI = "hf://datasets/syedroshanzameer/resume-classification/train.csv"
PROCESSED_DIR = os.path.join("data", "processed")
PHRASES_PATH = os.path.join(PROCESSED_DIR, "resume_classification_phrases.csv")
STATS_PATH = os.path.join(PROCESSED_DIR, "resume_classification_statistics.csv")

# -------------------------------------------------------------------------
# KEYWORD FILTERS (Replaces Integer Labels)
# -------------------------------------------------------------------------
# Stage 1: Obviously irrelevant industries to drop immediately
EXCLUDED_KEYWORDS = [
    "healthcare",
    "chef",
    "arts",
    "teacher",
    "fitness",
    "agriculture"
]

# Stage 2: Target roles required to keep the resume
ROLE_KEYWORDS = [
    "sales",
    "business development",
    "project manager",
    "programme manager",
    "program manager",
    "procurement",
    "buyer",
    "purchasing",
    "sourcing",
    "supply chain",
    "commodity",
    "vendor",
    "supplier"
]

COMPETENCY_WORDS = [
    "lead", "manage", "coordinate", "collaborate", "support", 
    "develop", "implement", "deliver", "improve", "achieve", 
    "drive", "create", "facilitate", "organise", "analyze", 
    "negotiate", "mentor", "plan", "communicate", "execute"
]

# -------------------------------------------------------------------------
# REGEX PATTERNS
# -------------------------------------------------------------------------
EMAIL_PATTERN = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b')
PHONE_PATTERN = re.compile(r'\+?\d{1,3}?[-.\s]?\(?\d{1,4}?\)?[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}')
URL_PATTERN = re.compile(r'https?://\S+|www\.\S+|linkedin\.com/\S+|github\.com/\S+')
ADDRESS_PATTERN = re.compile(r'\b\d{1,5}\s(?:[A-Za-z0-9\-\']+\s){1,3}(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln|Way|Court|Ct|Circle|Cir)\b', re.IGNORECASE)
BULLET_PATTERN = re.compile(r'(?:^|\n)\s*[\u2022\u2023\u25E6\u2043\-\*]\s*')

# -------------------------------------------------------------------------
# PIPELINE FUNCTIONS
# -------------------------------------------------------------------------

def is_relevant_resume(text):
    """Applies the two-stage keyword filtering to determine if a resume should be kept."""
    if pd.isna(text):
        return False
        
    text_lower = text.lower()
    
    # Stage 1: Discard if it contains obvious irrelevant industry keywords
    if any(excl in text_lower for excl in EXCLUDED_KEYWORDS):
        return False
        
    # Stage 2: Keep if it contains any of the target role keywords
    if any(role in text_lower for role in ROLE_KEYWORDS):
        return True
        
    return False

def clean_html(text):
    if pd.isna(text):
        return ""
    return BeautifulSoup(str(text), "html.parser").get_text(separator=" ")

def preprocess_text(text, stats_counter):
    text = clean_html(text)
    
    text, url_count = URL_PATTERN.subn("", text)
    stats_counter["URLs removed"] += url_count
    
    text, email_count = EMAIL_PATTERN.subn("", text)
    stats_counter["Emails removed"] += email_count
    
    text, phone_count = PHONE_PATTERN.subn("", text)
    stats_counter["Phone numbers removed"] += phone_count
    
    text, _ = ADDRESS_PATTERN.subn("", text)
    
    text = re.sub(r'\n{2,}', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    
    return text.strip()

def extract_phrases(text, resume_id, label):
    phrases = []
    blocks = re.split(r'\n', BULLET_PATTERN.sub('\n', text))
    
    for block in blocks:
        block = block.strip()
        if not block:
            continue
            
        sentences = sent_tokenize(block)
        
        for sentence in sentences:
            sentence = sentence.strip()
            words = word_tokenize(sentence)
            word_count = len([w for w in words if w.isalnum()])
            
            if 5 <= word_count <= 35:
                if sentence.count(',') > 4 and word_count < 15:
                    continue 
                
                sentence_lower = sentence.lower()
                if not any(comp_verb in sentence_lower for comp_verb in COMPETENCY_WORDS):
                    continue
                
                phrases.append({
                    "Resume_ID": resume_id,
                    "Label": label,  # Keeping original label integer for metadata tracking
                    "Phrase": sentence,
                    "Word_Count": word_count
                })
                
    return phrases

# -------------------------------------------------------------------------
# MAIN EXECUTION
# -------------------------------------------------------------------------

def main():
    print("Starting Resume Classification Preprocessing Pipeline...")
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    stats = {
        "Total resumes loaded": 0,
        "Relevant resumes retained": 0,
        "Duplicate resumes removed": 0,
        "Empty resumes removed": 0,
        "Emails removed": 0,
        "Phone numbers removed": 0,
        "URLs removed": 0,
        "Total candidate phrases extracted": 0,
        "Duplicate phrases removed": 0,
        "Final phrases retained": 0,
        "Average phrase length": 0,
        "Minimum phrase length": 0,
        "Maximum phrase length": 0,
        "Average phrases per resume": 0
    }

    try:
        df = pd.read_csv(DATA_URI)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
        
    stats["Total resumes loaded"] = len(df)
    
    # NEW: Apply Two-Stage Keyword Filter instead of Label filter
    print("Applying keyword filters...")
    df = df[df['text'].apply(is_relevant_resume)].copy()
    stats["Relevant resumes retained"] = len(df)
    
    # Remove structural duplicates/empties
    initial_len = len(df)
    df = df.dropna(subset=['text'])
    stats["Empty resumes removed"] = initial_len - len(df)
    
    initial_len = len(df)
    df = df.drop_duplicates(subset=['text'])
    stats["Duplicate resumes removed"] = initial_len - len(df)
    
    # Generate Resume IDs
    df['Resume_ID'] = [f"DS2_{str(i).zfill(5)}" for i in range(1, len(df) + 1)]
    
    print("Cleaning text and extracting phrases...")
    all_phrases = []
    
    for _, row in df.iterrows():
        clean_txt = preprocess_text(row['text'], stats)
        extracted = extract_phrases(clean_txt, row['Resume_ID'], row['labels'])
        all_phrases.extend(extracted)
        
    stats["Total candidate phrases extracted"] = len(all_phrases)
    
    # Deduplicate and Process Phrases
    phrases_df = pd.DataFrame(all_phrases)
    
    if not phrases_df.empty:
        initial_phrase_count = len(phrases_df)
        
        phrases_df['Phrase_Lower'] = phrases_df['Phrase'].str.lower().str.strip()
        phrases_df = phrases_df.drop_duplicates(subset=['Phrase_Lower']).drop(columns=['Phrase_Lower'])
        
        stats["Duplicate phrases removed"] = initial_phrase_count - len(phrases_df)
        stats["Final phrases retained"] = len(phrases_df)
        
        stats["Average phrase length"] = round(phrases_df['Word_Count'].mean(), 1)
        stats["Minimum phrase length"] = phrases_df['Word_Count'].min()
        stats["Maximum phrase length"] = phrases_df['Word_Count'].max()
        stats["Average phrases per resume"] = round(len(phrases_df) / len(df), 1)
        
        phrases_df.insert(0, 'Phrase_ID', [f"P{str(i).zfill(6)}" for i in range(1, len(phrases_df) + 1)])
        phrases_df.to_csv(PHRASES_PATH, index=False)
        print(f"Saved {len(phrases_df)} final phrases to {PHRASES_PATH}")
    else:
        print("Warning: No phrases extracted.")

    # Save Statistics
    stats_df = pd.DataFrame(list(stats.items()), columns=['Metric', 'Value'])
    stats_df.to_csv(STATS_PATH, index=False)
    print(f"Saved statistics to {STATS_PATH}")

if __name__ == "__main__":
    main()