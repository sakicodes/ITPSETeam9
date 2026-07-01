import os
import regex as re
import pandas as pd
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

# Ensure NLTK punkt is downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt_tab', quiet=True)
    nltk.download('punkt', quiet=True)

# -------------------------------------------------------------------------
# CONFIGURATION & PATHS
# -------------------------------------------------------------------------
RAW_DATA_PATH = os.path.join("data", "raw", "Resume.csv")
PROCESSED_DIR = os.path.join("data", "processed")
CLEAN_RESUMES_PATH = os.path.join(PROCESSED_DIR, "sales_clean_resumes.csv")
PHRASES_PATH = os.path.join(PROCESSED_DIR, "sales_phrases.csv")
STATS_PATH = os.path.join(PROCESSED_DIR, "sales_statistics.csv")

TARGET_CATEGORIES = [
    "sales",
    "business development",
    "project management",
    "programme management",
    "procurement",
    "supply chain",
    "sourcing"
]

# -------------------------------------------------------------------------
# REGEX PATTERNS
# -------------------------------------------------------------------------
EMAIL_PATTERN = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b')
# Matches international, domestic, and heavily formatted phone numbers
PHONE_PATTERN = re.compile(r'\+?\d{1,3}?[-.\s]?\(?\d{1,4}?\)?[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}')
URL_PATTERN = re.compile(r'https?://\S+|www\.\S+|linkedin\.com/\S+|github\.com/\S+')
# Best-effort physical address pattern (Street, Ave, Blvd, etc.)
ADDRESS_PATTERN = re.compile(r'\b\d{1,5}\s(?:[A-Za-z0-9\-\']+\s){1,3}(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln|Way|Court|Ct|Circle|Cir)\b', re.IGNORECASE)
BULLET_PATTERN = re.compile(r'(?:^|\n)\s*[\u2022\u2023\u25E6\u2043\-\*]\s*')

# -------------------------------------------------------------------------
# PIPELINE FUNCTIONS
# -------------------------------------------------------------------------

def clean_html(text):
    """Removes HTML tags safely using BeautifulSoup."""
    if pd.isna(text):
        return ""
    return BeautifulSoup(str(text), "html.parser").get_text(separator=" ")

def preprocess_text(text, stats_counter):
    """Cleans PII, URLs, and formats spacing while tracking removal counts."""
    # Remove HTML if any slipped through Resume_str
    text = clean_html(text)
    
    # Remove URLs
    text, url_count = URL_PATTERN.subn("", text)
    stats_counter["URLs removed"] += url_count
    
    # Remove Emails
    text, email_count = EMAIL_PATTERN.subn("", text)
    stats_counter["Emails removed"] += email_count
    
    # Remove Phone Numbers
    text, phone_count = PHONE_PATTERN.subn("", text)
    stats_counter["Phone numbers removed"] += phone_count
    
    # Remove Addresses (Best Effort)
    text, _ = ADDRESS_PATTERN.subn("", text)
    
    # Normalise whitespace and line breaks
    text = re.sub(r'\n{2,}', '\n', text)  # Collapse multiple newlines
    text = re.sub(r'[ \t]+', ' ', text)   # Collapse horizontal whitespace
    
    return text.strip()

def extract_phrases(text, resume_id, category):
    """Splits text into valid candidate phrases."""
    phrases = []
    
    # 1. Split by line breaks and bullet points first
    blocks = re.split(r'\n', BULLET_PATTERN.sub('\n', text))
    
    for block in blocks:
        block = block.strip()
        if not block:
            continue
            
        # 2. Split into sentences
        sentences = sent_tokenize(block)
        
        for sentence in sentences:
            sentence = sentence.strip()
            
            # 3. Tokenize to check word count
            words = word_tokenize(sentence)
            # Remove punctuation tokens from word count to be accurate
            word_count = len([w for w in words if w.isalnum()])
            
            # 4. Apply Filters (5-35 words, ignores single words / headings)
            if 5 <= word_count <= 35:
                # Basic heuristic to drop pure "skill lists" (high comma ratio, no verbs etc.)
                # E.g., "Python, Java, C++, SQL, Excel, Word, PowerPoint"
                comma_count = sentence.count(',')
                if comma_count > 4 and word_count < 15:
                    continue # Likely a technical skill list, drop it
                
                phrases.append({
                    "Resume_ID": resume_id,
                    "Category": category,
                    "Phrase": sentence,
                    "Word_Count": word_count
                })
                
    return phrases

# -------------------------------------------------------------------------
# MAIN EXECUTION
# -------------------------------------------------------------------------

def main():
    print("Starting Resume Preprocessing Pipeline...")
    
    # Ensure directories exist
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    # Initialize Statistics Tracker
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

    # 1. Load Data
    try:
        df = pd.read_csv(RAW_DATA_PATH)
    except FileNotFoundError:
        print(f"Error: Could not find raw data at {RAW_DATA_PATH}")
        return
        
    stats["Total resumes loaded"] = len(df)
    
    # 2. Filter Categories (Case insensitive mapping)
    df['Category_Lower'] = df['Category'].astype(str).str.lower().str.strip()
    df = df[df['Category_Lower'].isin(TARGET_CATEGORIES)].copy()
    stats["Relevant resumes retained"] = len(df)
    
    # 3. Remove structural duplicates / empties at the resume level
    initial_len = len(df)
    df = df.dropna(subset=['Resume_str'])
    stats["Empty resumes removed"] = initial_len - len(df)
    
    initial_len = len(df)
    df = df.drop_duplicates(subset=['Resume_str'])
    stats["Duplicate resumes removed"] = initial_len - len(df)
    
    # 4. Clean Resumes
    print("Cleaning resumes (PII, URLs, Formatting)...")
    df['Clean_Text'] = df.apply(lambda row: preprocess_text(
        # Fallback to HTML column if string column is weak, but prefer str
        row['Resume_str'] if pd.notna(row['Resume_str']) else clean_html(row['Resume_html']), 
        stats
    ), axis=1)
    
    # Save clean resumes
    clean_resumes_df = df[['ID', 'Category', 'Clean_Text']].rename(columns={'ID': 'Resume_ID'})
    clean_resumes_df.to_csv(CLEAN_RESUMES_PATH, index=False)
    print(f"Saved {len(clean_resumes_df)} clean resumes to {CLEAN_RESUMES_PATH}")

    # 5. Extract Phrases
    print("Extracting candidate phrases...")
    all_phrases = []
    
    for _, row in clean_resumes_df.iterrows():
        extracted = extract_phrases(row['Clean_Text'], row['Resume_ID'], row['Category'])
        all_phrases.extend(extracted)
        
    stats["Total candidate phrases extracted"] = len(all_phrases)
    
    # 6. Deduplicate and Process Phrases
    phrases_df = pd.DataFrame(all_phrases)
    
    if not phrases_df.empty:
        initial_phrase_count = len(phrases_df)
        
        # Deduplicate based on the actual text (case insensitive, stripped)
        phrases_df['Phrase_Lower'] = phrases_df['Phrase'].str.lower().str.strip()
        phrases_df = phrases_df.drop_duplicates(subset=['Phrase_Lower']).drop(columns=['Phrase_Lower'])
        
        stats["Duplicate phrases removed"] = initial_phrase_count - len(phrases_df)
        stats["Final phrases retained"] = len(phrases_df)
        
        # Calculate phrase stats
        stats["Average phrase length"] = round(phrases_df['Word_Count'].mean(), 1)
        stats["Minimum phrase length"] = phrases_df['Word_Count'].min()
        stats["Maximum phrase length"] = phrases_df['Word_Count'].max()
        stats["Average phrases per resume"] = round(len(phrases_df) / len(clean_resumes_df), 1)
        
        # Generate P000001 format IDs
        phrases_df.insert(0, 'Phrase_ID', [f"P{str(i).zfill(6)}" for i in range(1, len(phrases_df) + 1)])
        
        # Save phrases
        phrases_df.to_csv(PHRASES_PATH, index=False)
        print(f"Saved {len(phrases_df)} final phrases to {PHRASES_PATH}")
    else:
        print("Warning: No phrases extracted.")

    # 7. Save Statistics
    stats_df = pd.DataFrame(list(stats.items()), columns=['Metric', 'Value'])
    stats_df.to_csv(STATS_PATH, index=False)
    print(f"Saved statistics to {STATS_PATH}")
    print("Pipeline complete.")

if __name__ == "__main__":
    main()