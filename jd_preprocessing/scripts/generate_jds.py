import os
import json
import time
import pandas as pd
import random
import spacy
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

INPUT_PATH = os.path.join("data", "processed", "jd_phrases.csv")
OUTPUT_PATH = os.path.join("outputs", "experimental_jds.csv")

INDUSTRIES = ["Sales / Business Development", "Procurement / Sourcing / Supply Chain", "Project Management / Programme Management"]
SENIORITIES = ["Entry-Level", "Senior-Level"]
REGIONS = ["USA", "China", "Europe", "Singapore"]

print("Loading local NLP models for Validation Gate...")
nlp = spacy.load("en_core_web_sm")
client = genai.Client(api_key=API_KEY)
MODEL_NAME = "gemini-3.1-flash-lite"

# The words that would ruin a Seniority-Agnostic JD
TOXIC_WORDS = ['senior', 'director', 'exec', 'executive', 'lead', 'manager', 'head', 'entry', 'intern', 'junior', 'associate', 'trainee', 'grad', 'graduate']

def get_menu(df, ind):
    """Pulls a completely Seniority-Agnostic pool of phrases for the industry."""
    pool = df[df['Industry'] == ind]
    
    # Pre-filter: Remove any phrase that implies a specific seniority
    pool = pool[~pool['Phrase'].str.lower().str.contains('|'.join(TOXIC_WORDS))]
    
    menu = [{"Phrase_ID": r['Phrase_ID'], "Phrase": r['Phrase'], "Type": str(r['Phrase_Type']).capitalize(), "Competency": str(r['Competency']).capitalize()} for _, r in pool.iterrows()]
    random.shuffle(menu)
    return menu[:150]

def validate_jd(text):
    text_lower = text.lower()
    
    # 1. Seniority-Agnostic Check: The text must be safe for BOTH Entry and Senior titles
    if any(w in text_lower for w in TOXIC_WORDS):
        return False, "Seniority-specific word detected. Text must be completely agnostic."

    # 2. Duplicate Phrase Check
    bullets = [line.strip() for line in text.split('\n') if line.strip().startswith('-')]
    if len(bullets) != len(set(bullets)):
        return False, "Duplicate phrases detected in bullet points."

    # 3. Expanded Entity Check (Feedback 3)
    doc = nlp(text)
    ignore_terms = [
        'company', 'location', 'hq', 'organization', 'sales', 'business', 'project', 
        'supply', 'team', 'office', 'category management', 'operations', 'administration', 
        'ecc', 'outsourcing', 'leadership', '[location]', '[company]', 'bd', 'comscore', 
        'oracle', 'quality', 'customer', 'customs', 'certification', 'management', 
        'planning', 'forecasting', 'client', 'services', 'b2b', 'b2c'
    ]
    
    for ent in doc.ents:
        if ent.label_ in ['GPE', 'LOC', 'ORG']:
            if not any(safe_word in ent.text.lower() for safe_word in ignore_terms):
                return False, f"Entity leak detected: {ent.text} ({ent.label_})"
                
    return True, "Passed"

def main():
    print("Stage 5: LLM Generation + Automated Validation Gate (9-Skeleton Approach)...")
    if not os.path.exists(INPUT_PATH): return print("Phrases missing.")
    
    df = pd.read_csv(INPUT_PATH)
    final_jds, jd_id = [], 1
    
    for ind in INDUSTRIES:
        print(f"\nGenerating 3 Generic Bases for {ind}...")
        menu = get_menu(df, ind)
        
        for variation in range(3):
            num_resp = random.randint(6, 9)
            num_req = random.randint(4, 6)
            num_comp = random.randint(3, 4) 
            
            prompt = f"""
            You are an HR copywriter building a cohesive, strictly SENIORITY-AGNOSTIC job description.
            Target Industry: {ind}
            
            RULES:
            1. Select EXACTLY {num_resp} phrases where Type="Responsibility".
            2. Select EXACTLY {num_req} phrases where Type="Requirement".
            3. Select EXACTLY {num_comp} Agentic Skill phrases and {num_comp} Communal Skill phrases.
            4. ROLE CONSISTENCY: Ensure the phrases form a logical, cohesive role.
            5. EDITING FOR FLOW: Rewrite the phrases so they start with an action verb. 
            6. SENIORITY NEUTRALITY (CRITICAL): The resulting text will be used for both "Entry-Level" and "Senior-Level" job titles. You MUST NOT include any words that imply a specific rank (e.g., do not use words like manager, director, lead, intern, junior, senior). 
            7. ANONYMIZATION: Replace any cities, countries, or company names with [Location] or [Company].
            
            Output ONLY JSON:
            {{
              "Selected_IDs": ["P_123"],
              "Base_Text": "\\n**Responsibilities:**\\n- [Edited Phrase]\\n\\n**Requirements:**\\n- [Edited Phrase]",
              "Competency_Text": "\\n\\n**Competencies:**\\n- [Edited Phrase]"
            }}
            
            CANDIDATE PHRASES: {json.dumps(menu, indent=2)}
            """
            
            valid_jd_created = False
            attempts = 0
            
            while not valid_jd_created and attempts < 10:
                attempts += 1
                try:
                    response = client.models.generate_content(
                        model=MODEL_NAME, contents=prompt,
                        config=types.GenerateContentConfig(response_mime_type="application/json")
                    )
                    
                    data = json.loads(response.text)
                    full_eval_text = data.get("Base_Text", "") + data.get("Competency_Text", "")
                    
                    is_valid, reason = validate_jd(full_eval_text)
                    
                    if is_valid:
                        valid_jd_created = True
                        source_ids = ", ".join(data.get("Selected_IDs", []))
                        
                        # Clone the EXACT SAME TEXT across all Regions and Seniorities
                        for sen in SENIORITIES:
                            for reg in REGIONS:
                                title = f"**Job Title:** {sen} {ind.split('/')[0].strip()}\n**Location:** {reg}\n"
                                
                                # Neutral Variant
                                final_jds.append({"JD_ID": f"EXP_{str(jd_id).zfill(3)}", "Industry": ind, "Role_Family": ind.split('/')[0].strip(), "Seniority": sen, "Region": reg, "Framing": "Neutral", "Full_Job_Description": title + data.get("Base_Text", ""), "Source_Phrases": source_ids})
                                jd_id += 1
                                
                                # Balanced Variant
                                final_jds.append({"JD_ID": f"EXP_{str(jd_id).zfill(3)}", "Industry": ind, "Role_Family": ind.split('/')[0].strip(), "Seniority": sen, "Region": reg, "Framing": "Balanced", "Full_Job_Description": title + full_eval_text, "Source_Phrases": source_ids})
                                jd_id += 1
                                
                        print(f"  - Variation {variation + 1}/3 passed validation and saved (Cloned to 16 rows).")
                    else:
                        print(f"  - Validation Failed ({reason}). Regenerating...")
                        
                except Exception as e:
                    time.sleep(3)

    pd.DataFrame(final_jds).to_csv(OUTPUT_PATH, index=False)
    print(f"\nSuccessfully generated {len(final_jds)} highly validated JDs from 9 Base Skeletons.")

if __name__ == "__main__":
    main()