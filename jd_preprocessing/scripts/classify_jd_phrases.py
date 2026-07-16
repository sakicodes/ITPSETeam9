import os
import time
import json
import pandas as pd
from google import genai
from google.genai import types
from tqdm import tqdm
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

INPUT_PATH = os.path.join("data", "processed", "jd_raw_phrases_balanced.csv")
OUTPUT_PATH = os.path.join("outputs", "phrase_classification_log.csv")

DICT_DIR = os.path.join("data", "dictionaries")
AGENTIC_PATH = os.path.join(DICT_DIR, "agentic_expanded.csv")
COMMUNAL_PATH = os.path.join(DICT_DIR, "communal_expanded.csv")
NEUTRAL_PATH = os.path.join(DICT_DIR, "neutral_expanded.csv")

BATCH_SIZE = 25
MODEL_NAME = "gemini-3.1-flash-lite"
client = genai.Client(api_key=API_KEY)

def load_dictionary_terms(path):
    if not os.path.exists(path): return "No dictionary provided."
    df = pd.read_csv(path)
    return ", ".join(df['Keyword / Phrase'].dropna().astype(str).unique().tolist())

def main():
    print("Stage 3: Classifying Phrases via LLM API...")
    if not os.path.exists(INPUT_PATH): return print("Input missing.")
    
    agentic_terms = load_dictionary_terms(AGENTIC_PATH)
    communal_terms = load_dictionary_terms(COMMUNAL_PATH)
    neutral_terms = load_dictionary_terms(NEUTRAL_PATH)
    
    SYSTEM_PROMPT = f"""
    You are an expert HR data classifier. Review the following job description phrases.
    For EACH phrase, determine if it should be Kept, Removed, or Split.

    RULES:
    1. Split: If the phrase contains multiple distinct competencies, split it.
    2. Remove: Company marketing, testimonials, recruitment slogans, awards, EEO statements.
    3. ANONYMIZATION: Replace specific cities, countries, or company names with '[Company]' or '[Location]'.
    4. STRICT CATEGORY DEFINITIONS: 
        - "Requirement": Hard skills, degrees, software knowledge (e.g., "Must know Python", "BSc in Biology").
        - "Competency": BEHAVIOURAL traits and soft skills ONLY. Must map to Agentic, Communal, or Neutral dictionaries. If it is a hard skill or domain knowledge, it is NOT a Competency.

    --- RESEARCH DICTIONARIES ---
    AGENTIC TERMS: {agentic_terms}
    COMMUNAL TERMS: {communal_terms}
    NEUTRAL TERMS: {neutral_terms}
    -----------------------------

    Output strictly as a JSON array matching this exact schema:
    [
      {{
        "Phrase_ID": "P_000001",
        "Decision": "Keep",
        "Outputs": [
          {{
            "Split_Output": "The anonymized phrase.",
            "Phrase_Type": "Responsibility",
            "Competency": "Neutral",
            "Reason": "Standard task.",
            "Confidence": 95
          }}
        ]
      }}
    ]
    """
        
    df_input = pd.read_csv(INPUT_PATH)
    
    processed_ids = set()
    if os.path.exists(OUTPUT_PATH):
        processed_ids = set(pd.read_csv(OUTPUT_PATH)['Phrase_ID'].astype(str))

    df_todo = df_input[~df_input['Phrase_ID'].astype(str).isin(processed_ids)]
    phrases_to_process = df_todo.to_dict('records')
    
    if not phrases_to_process: return print("All phrases already classified.")

    batches = [phrases_to_process[i:i + BATCH_SIZE] for i in range(0, len(phrases_to_process), BATCH_SIZE)]
    log_counter = len(processed_ids) + 1
    
    for batch in tqdm(batches):
        payload = [{"Phrase_ID": r['Phrase_ID'], "Phrase": r['Phrase']} for r in batch]
        msg = f"{SYSTEM_PROMPT}\n\nData:\n{json.dumps(payload, indent=2)}"
        
        success, attempts = False, 0
        while not success and attempts < 3:
            try:
                response = client.models.generate_content(
                    model=MODEL_NAME, contents=msg,
                    config=types.GenerateContentConfig(response_mime_type="application/json")
                )
                results = json.loads(response.text)
                new_logs = []
                for item in results:
                    orig_phrase = next((r['Phrase'] for r in batch if r['Phrase_ID'] == item['Phrase_ID']), "")
                    for out in item.get('Outputs', []):
                        new_logs.append({
                            "Log_ID": f"LOG_{str(log_counter).zfill(6)}",
                            "Phrase_ID": item['Phrase_ID'], "Original_Phrase": orig_phrase,
                            "Split_Output": out.get('Split_Output', ''), "Phrase_Type": out.get('Phrase_Type', ''),
                            "Competency": out.get('Competency', ''), "Decision": item.get('Decision', 'Keep'),
                            "Confidence": out.get('Confidence', 0), "Reason": out.get('Reason', ''),
                            "Model": MODEL_NAME, "Prompt_Version": "V3_Expanded_Anon",
                            "Timestamp": datetime.now().isoformat()
                        })
                        log_counter += 1
                pd.DataFrame(new_logs).to_csv(OUTPUT_PATH, mode='a', header=False, index=False)
                success = True
            except Exception as e:
                attempts += 1
                time.sleep(3) 

if __name__ == "__main__":
    main()