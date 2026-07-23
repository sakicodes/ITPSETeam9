import os
import csv
import json
import time
import datetime
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# Limits config
GROQ_MODEL = "llama-3.3-70b-versatile"
MAX_DAILY_TOKENS = 95000  # Stop slightly before 100k
MAX_DAILY_REQUESTS = 950  # Stop slightly before 1k
DELAY_BETWEEN_CALLS = 12.0 # Enforce TPM safety

CSV_HEADERS = [
    "Result_id", "Model_Name", "Prompt_Version", "Timestamp_SGT",
    "Industry", "Seniority_level", "JD_ID", "JD_framing", "JD_Region",
    "Frame_CV_ID", "CV_framing", 
    "Hireability_Score", "Hireability_Reason", "Competence_Score", "Competence_Reason",
    "Fit_Score", "Fit_Reason", "Performance_Score", "Performance_Reason",
    "Leadership_Exhibited", "Leadership_Control", "Leadership_Effective", "Leadership_Avg", "Leadership_Reason",
    "Status_Score", "Status_Reason", "Warmth_Score", "Warmth_Reason",
    "Recommendation", "Raw_JSON", "Error"
]

def load_tracker(tracker_file):
    today_str = datetime.date.today().isoformat()
    if tracker_file.exists():
        with open(tracker_file, "r") as f:
            data = json.load(f)
            if data.get("date") == today_str:
                return data
    return {"date": today_str, "tokens": 0, "requests": 0}

def save_tracker(tracker_file, data):
    with open(tracker_file, "w") as f:
        json.dump(data, f)

def call_groq(client, prompt_text):
    res = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt_text}],
        temperature=0.0,
        seed=42,
        response_format={"type": "json_object"}
    )
    content = res.choices[0].message.content
    total_tokens = res.usage.total_tokens
    return content, total_tokens

def main():
    base_dir = Path(__file__).parent.parent
    outputs_dir = base_dir / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    
    tracker_file = base_dir / "groq_state.json"
    usage_data = load_tracker(tracker_file)

    cv_df = pd.read_csv(base_dir / "datasets" / "cover_letters_final.csv")
    jd_df = pd.read_csv(base_dir / "datasets" / "jds" / "jds_final.csv")

    with open(base_dir / "prompt_design" / "prompt-design.md", "r", encoding="utf-8") as f:
        prompt_standard_template = f.read()
    with open(base_dir / "prompt_design" / "prompt-design-region.md", "r", encoding="utf-8") as f:
        prompt_region_template = f.read()

    sgt = datetime.timezone(datetime.timedelta(hours=8))
    client = Groq(api_key=os.getenv("GROQ_LLAMA_API_KEY"))

    for prompt_version, prompt_template in [("standard", prompt_standard_template), ("region", prompt_region_template)]:
        output_file = outputs_dir / f"results_groq_{prompt_version}.csv"
        processed_ids = set()
        
        if output_file.exists():
            with open(output_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                processed_ids.update(row["Result_id"] for row in reader)
        else:
            with open(output_file, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(CSV_HEADERS)

        print(f"\n--- Running: Groq ({GROQ_MODEL}) | Prompt: {prompt_version} ---")

        for _, jd_row in jd_df.iterrows():
            matched_cvs = cv_df[cv_df["Field"] == jd_row["Industry"]]
            for _, cv_row in matched_cvs.iterrows():
                
                # Check daily limits before generating the next ID
                if usage_data["tokens"] >= MAX_DAILY_TOKENS or usage_data["requests"] >= MAX_DAILY_REQUESTS:
                    print(f"\n[STOP] Approaching Groq daily limits: {usage_data['tokens']} tokens, {usage_data['requests']} requests.")
                    print("Please run this script again tomorrow.")
                    return # Exit the script completely

                result_id = f"{jd_row['JD_ID']}_{cv_row['Frame_CV_ID']}"
                if result_id in processed_ids: continue 

                prompt = prompt_template.replace("{{JOB_DESCRIPTION}}", str(jd_row["Full_Job_Description"])).replace("{{COVER_LETTER}}", str(cv_row["Cover_Letter"]))
                if prompt_version == "region": prompt = prompt.replace("{{REGION}}", str(jd_row["Region"]))

                parsed_data, error_msg, raw_response = {}, "", ""
                for attempt in range(5):
                    try:
                        time.sleep(DELAY_BETWEEN_CALLS)
                        raw_response, used_tokens = call_groq(client, prompt)
                        
                        # Update and save tracker immediately
                        usage_data["tokens"] += used_tokens
                        usage_data["requests"] += 1
                        save_tracker(tracker_file, usage_data)
                        
                        parsed_data = json.loads(raw_response.strip().removeprefix("```json").removesuffix("```").strip())
                        break
                    except Exception as e:
                        error_msg = str(e)
                        backoff_time = (2 ** attempt) * 6
                        print(f"[Error] Groq ({result_id}): {error_msg}. Retrying in {backoff_time}s...")
                        time.sleep(backoff_time)
                
                row_data = {
                    "Result_id": result_id, "Model_Name": GROQ_MODEL, "Prompt_Version": prompt_version,
                    "Timestamp_SGT": datetime.datetime.now(sgt).strftime("%Y-%m-%d %H:%M:%S"),
                    "Industry": jd_row["Industry"], "Seniority_level": jd_row.get("Seniority", ""),
                    "JD_ID": jd_row["JD_ID"], "JD_framing": jd_row.get("Framing", ""), "JD_Region": jd_row.get("Region", ""),
                    "Frame_CV_ID": cv_row["Frame_CV_ID"], "CV_framing": cv_row.get("Framing", ""),
                    "Hireability_Score": parsed_data.get("hireability", {}).get("score", ""), "Hireability_Reason": parsed_data.get("hireability", {}).get("reason", ""),
                    "Competence_Score": parsed_data.get("perceived_competence", {}).get("score", ""), "Competence_Reason": parsed_data.get("perceived_competence", {}).get("reason", ""),
                    "Fit_Score": parsed_data.get("person_job_fit", {}).get("score", ""), "Fit_Reason": parsed_data.get("person_job_fit", {}).get("reason", ""),
                    "Performance_Score": parsed_data.get("expected_performance", {}).get("score", ""), "Performance_Reason": parsed_data.get("expected_performance", {}).get("reason", ""),
                    "Leadership_Exhibited": parsed_data.get("leadership_potential", {}).get("leadership_exhibited", ""), "Leadership_Control": parsed_data.get("leadership_potential", {}).get("control_over_activities", ""),
                    "Leadership_Effective": parsed_data.get("leadership_potential", {}).get("effective_leader", ""), "Leadership_Avg": parsed_data.get("leadership_potential", {}).get("average", ""), "Leadership_Reason": parsed_data.get("leadership_potential", {}).get("reason", ""),
                    "Status_Score": parsed_data.get("expected_status", {}).get("score", ""), "Status_Reason": parsed_data.get("expected_status", {}).get("reason", ""),
                    "Warmth_Score": parsed_data.get("perceived_warmth", {}).get("score", ""), "Warmth_Reason": parsed_data.get("perceived_warmth", {}).get("reason", ""),
                    "Recommendation": parsed_data.get("recommendation", ""), "Raw_JSON": raw_response if error_msg else "", "Error": error_msg if not parsed_data else ""
                }
                
                with open(output_file, "a", newline="", encoding="utf-8") as f:
                    csv.DictWriter(f, fieldnames=CSV_HEADERS).writerow(row_data)
                
                print(f"[Groq] Processed {result_id} | Daily Tokens: {usage_data['tokens']}/100k")

if __name__ == "__main__":
    main()