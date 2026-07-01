import os
import time
import json
import pandas as pd
from google import genai
from google.genai import types
from tqdm import tqdm

# -------------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------------
# PASTE YOUR API KEY HERE
API_KEY = "YOUR_API_KEY_HERE" 

INPUT_FILE = os.path.join("data", "processed", "merged_phrases.csv")
OUTPUT_FILE = os.path.join("data", "processed", "classified_phrases.csv")

BATCH_SIZE = 25
# 15 requests per minute = 1 request every 4 seconds. We use 4.5 to be safe.
RATE_LIMIT_DELAY = 2.25  

# Initialize the new Gemini Client
client = genai.Client(api_key=API_KEY)

# -------------------------------------------------------------------------
# PROMPT TEMPLATE
# -------------------------------------------------------------------------
SYSTEM_PROMPT = """
You are a highly accurate data classification API. 
You will receive a list of resume phrases, each with a Phrase_ID. 
For each phrase, determine its Industry and its Framing.

RULES:
1. "Industry" MUST be exactly one of: ["Sales/Business Development", "Procurement/Sourcing/Supply Chain", "Project/Programme Management", "Unknown"]. 
   (If it truly fits none, use "Unknown").
2. "Framing" MUST be exactly one of: ["Agentic", "Communal", "Neutral", "Balanced"].
3. Return ONLY a JSON array of objects.

Output format requirement:
[
  {
    "Phrase_ID": "P000001",
    "Industry": "Sales/Business Development",
    "Framing": "Agentic"
  }
]
"""

# -------------------------------------------------------------------------
# MAIN EXECUTION
# -------------------------------------------------------------------------
def main():
    print("Starting Gemini API Classification...")
    
    # 1. Load the input data
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return
        
    df_input = pd.read_csv(INPUT_FILE)
    total_phrases = len(df_input)
    print(f"Loaded {total_phrases} phrases.")

    # 2. Check for existing progress (Auto-Resume functionality)
    processed_ids = set()
    if os.path.exists(OUTPUT_FILE):
        df_existing = pd.read_csv(OUTPUT_FILE)
        processed_ids = set(df_existing['Phrase_ID'].astype(str))
        print(f"Found existing output file. Resuming... ({len(processed_ids)} already processed)")
    else:
        # Create empty CSV with headers if starting fresh
        pd.DataFrame(columns=['Phrase_ID', 'Industry', 'Framing']).to_csv(OUTPUT_FILE, index=False)

    # Filter out already processed phrases
    df_todo = df_input[~df_input['Phrase_ID'].astype(str).isin(processed_ids)]
    phrases_to_process = df_todo.to_dict('records')
    
    if not phrases_to_process:
        print("All phrases have already been processed!")
        return

    # 3. Process in Batches
    batches = [phrases_to_process[i:i + BATCH_SIZE] for i in range(0, len(phrases_to_process), BATCH_SIZE)]
    
    print(f"Processing {len(phrases_to_process)} phrases in {len(batches)} batches...")
    
    for batch in tqdm(batches, desc="Classifying Batches"):
        payload = [{"Phrase_ID": row['Phrase_ID'], "Phrase": row['Phrase']} for row in batch]
        user_message = f"{SYSTEM_PROMPT}\n\nHere is the data to classify:\n{json.dumps(payload, indent=2)}"
        
        success = False
        attempts = 0
        
        while not success and attempts < 3:
            try:
                # Use the new SDK generation method
                response = client.models.generate_content(
                    model='gemini-2.5-flash-lite',
                    contents=user_message,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                    )
                )
                
                # Parse the JSON response
                results = json.loads(response.text)
                
                # Append to CSV immediately (Auto-Save)
                df_results = pd.DataFrame(results)
                df_results.to_csv(OUTPUT_FILE, mode='a', header=False, index=False)
                
                success = True
                
            except Exception as e:
                attempts += 1
                print(f"\nError on batch (Attempt {attempts}/3): {e}")
                time.sleep(5) 
                
        if not success:
            print(f"\nFailed to process batch after 3 attempts. Skipping to next batch.")
            
        # Strictly enforce rate limit delay
        time.sleep(RATE_LIMIT_DELAY)

    # 4. Final Merge
    print("\nClassification complete! Merging results with original data...")
    df_original = pd.read_csv(INPUT_FILE)
    df_classified = pd.read_csv(OUTPUT_FILE)
    
    df_final = pd.merge(df_original, df_classified, on='Phrase_ID', how='left')
    df_final.to_csv(OUTPUT_FILE, index=False)
    print(f"Done! Final dataset saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()