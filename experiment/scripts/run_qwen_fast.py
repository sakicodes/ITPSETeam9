import os
import csv
import json
import time
import datetime
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv

from openai import OpenAI

load_dotenv()

MODEL_STRING = "qwen3.7-plus"
PROVIDER = "qwen"
MAX_WORKERS = 8 # Number of simultaneous API calls
DELAY = 1.0 # Slight pacing between thread spawns

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

# Thread lock to prevent CSV write collisions
write_lock = threading.Lock()

def call_qwen(client, prompt_text):
    res = client.chat.completions.create(
        model=MODEL_STRING, 
        messages=[{"role": "user", "content": prompt_text + "\n\nEnsure output is strictly JSON format."}],
        temperature=0.0,
        response_format={"type": "json_object"}
    )
    return res.choices[0].message.content

def process_task(task):
    jd_row, cv_row, prompt_template, prompt_version, result_id, output_file, client, sgt = task

    # Construct prompt
    prompt = prompt_template.replace("{{JOB_DESCRIPTION}}", str(jd_row["Full_Job_Description"])).replace("{{COVER_LETTER}}", str(cv_row["Cover_Letter"]))
    if prompt_version == "region": 
        prompt = prompt.replace("{{REGION}}", str(jd_row["Region"]))

    parsed_data, error_msg, raw_response = {}, "", ""
    
    # Retry logic (isolated per thread)
    for attempt in range(5):
        try:
            raw_response = call_qwen(client, prompt)
            
            # Clean <think> tags and markdown
            cleaned_response = re.sub(r'<think>.*?</think>', '', raw_response, flags=re.DOTALL).strip()
            if cleaned_response.startswith("```"):
                cleaned_response = cleaned_response.split("\n", 1)[-1]
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response.rsplit("\n", 1)[0]
                
            cleaned_response = cleaned_response.strip()
            parsed_data = json.loads(cleaned_response)
            error_msg = ""
            break
            
        except Exception as e:
            error_msg = str(e)
            backoff_time = (2 ** attempt) * 6
            print(f"[Error] {PROVIDER} ({result_id}): {error_msg}. Retrying in {backoff_time}s...")
            time.sleep(backoff_time)
            
    # Compile row data
    row_data = {
        "Result_id": result_id, "Model_Name": MODEL_STRING, "Prompt_Version": prompt_version,
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
    
    # Thread-safe write to CSV
    with write_lock:
        with open(output_file, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=CSV_HEADERS).writerow(row_data)
        print(f"[{PROVIDER}] Processed {result_id}")

def main():
    base_dir = Path.cwd()
    outputs_dir = base_dir / "outputs"
    outputs_dir.mkdir(exist_ok=True)

    cv_df = pd.read_csv(base_dir / "datasets" / "cover_letters" / "cover_letters_final.csv")
    jd_df = pd.read_csv(base_dir / "datasets" / "jds" / "jds_final.csv")

    with open(base_dir / "prompt_design" / "prompt-design.md", "r", encoding="utf-8") as f:
        prompt_standard_template = f.read()
    with open(base_dir / "prompt_design" / "prompt-design-region.md", "r", encoding="utf-8") as f:
        prompt_region_template = f.read()

    sgt = datetime.timezone(datetime.timedelta(hours=8))
    
    # Initialize Qwen client with 300s timeout to allow long thinking phases
    qwen_base_url = "https://ws-7zwv3lrntxh3hfsb.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1"
    client = OpenAI(
        api_key=os.getenv("QWEN_API_KEY"), 
        base_url=qwen_base_url,
        timeout=300.0
    )

    for prompt_version, prompt_template in [("standard", prompt_standard_template), ("region", prompt_region_template)]:
        output_file = outputs_dir / f"results_{PROVIDER}_{prompt_version}.csv"
        processed_ids = set()
        
        if output_file.exists():
            with open(output_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                processed_ids.update(row["Result_id"] for row in reader)
        else:
            with open(output_file, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(CSV_HEADERS)

        print(f"\n--- Running: {PROVIDER} ({MODEL_STRING}) | Prompt: {prompt_version} | Threads: {MAX_WORKERS} ---")
        print(f"Already processed {len(processed_ids)} records. Resuming...")

        # Build list of tasks for the thread pool
        tasks = []
        for _, jd_row in jd_df.iterrows():
            matched_cvs = cv_df[cv_df["Field"] == jd_row["Industry"]]
            for _, cv_row in matched_cvs.iterrows():
                result_id = f"{jd_row['JD_ID']}_{cv_row['Frame_CV_ID']}"
                
                if result_id not in processed_ids:
                    tasks.append((jd_row, cv_row, prompt_template, prompt_version, result_id, output_file, client, sgt))

        # Execute tasks using a ThreadPool
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Submit tasks with slight delay to prevent initial rate-limit spike
            futures = []
            for task in tasks:
                futures.append(executor.submit(process_task, task))
                time.sleep(DELAY) # Spread out the initial connections
            
            # Wait for all to complete (errors are handled inside process_task)
            for future in as_completed(futures):
                pass 

if __name__ == "__main__":
    main()