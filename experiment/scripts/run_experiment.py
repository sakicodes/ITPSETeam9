import os
import csv
import json
import time
import datetime
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv

from openai import OpenAI
from anthropic import Anthropic
# Updated Google GenAI SDK imports
from google import genai
from google.genai import types
# Updated Mistral v2 SDK import
from mistralai.client import Mistral

load_dotenv()

MODELS = {
    "openai": "gpt-4.1-mini",
    "anthropic": "claude-haiku-4-5-20251001",
    "google": "gemini-2.5-flash",
    "tencent": "tencent/Hy3", 
    "minimax": "MiniMax-M3",
    "deepseek": "deepseek-v4-pro",
    "qwen": "qwen3.7-plus",
    "mistral": "mistral-medium-latest",
    "sealion": "aisingapore/Gemma-SEA-LION-v4.5-E2B-IT"
}

PROVIDER_DELAYS = {
    "sealion": 6.5,
    "openai": 1.0, "anthropic": 1.0, "google": 1.0, "mistral": 1.0,
    "tencent": 1.0, "minimax": 1.0, "deepseek": 1.0, "qwen": 1.0
}

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

def call_llm(provider, model_string, prompt_text):
    if provider == "openai":
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        res = client.chat.completions.create(
            model=model_string, messages=[{"role": "user", "content": prompt_text}],
            temperature=0.0, seed=42, response_format={"type": "json_object"}
        )
        return res.choices[0].message.content

    elif provider == "anthropic":
        client = Anthropic(api_key=os.getenv("CLAUDE_API_KEY"))
        res = client.messages.create(
            model=model_string, max_tokens=1000, temperature=0.0,
            messages=[{"role": "user", "content": prompt_text + "\n\nReturn JSON only."}]
        )
        return res.content[0].text

    elif provider == "google":
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        res = client.models.generate_content(
            model=model_string,
            contents=prompt_text,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.0
            )
        )
        return res.text
    
    elif provider == "mistral":
        client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
        res = client.chat.complete(
            model=model_string, messages=[{"role": "user", "content": prompt_text}],
            temperature=0.0, response_format={"type": "json_object"}
        )
        return res.choices[0].message.content

    elif provider in ["tencent", "minimax", "deepseek", "qwen", "sealion"]:
        base_urls = {
            "tencent": "https://tokenhub-intl.tencentcloudmaas.com/v1",
            "minimax": "https://api.minimax.io/v1",
            "deepseek": "https://api.deepseek.com",
            "qwen": "https://ws-7zwv3lrntxh3hfsb.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1",
            "sealion": "https://api.sea-lion.ai/v1"
        }
        api_keys = {
            "tencent": os.getenv("TENCENT_API_KEY"), "minimax": os.getenv("MINIMAX_API_KEY"),
            "deepseek": os.getenv("DEEPSEEK_API_KEY"), "qwen": os.getenv("QWEN_API_KEY"), "sealion": os.getenv("SEALION_API_KEY")
        }
        client = OpenAI(api_key=api_keys[provider], base_url=base_urls[provider])
        res = client.chat.completions.create(
            model=model_string, messages=[{"role": "user", "content": prompt_text + "\n\nEnsure output is strictly JSON format."}],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        return res.choices[0].message.content

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

    for provider, model_string in MODELS.items():
        delay = PROVIDER_DELAYS.get(provider, 1.0)
        
        for prompt_version, prompt_template in [("standard", prompt_standard_template), ("region", prompt_region_template)]:
            output_file = outputs_dir / f"results_{provider}_{prompt_version}.csv"
            processed_ids = set()
            
            if output_file.exists():
                with open(output_file, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    processed_ids.update(row["Result_id"] for row in reader)
            else:
                with open(output_file, "w", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow(CSV_HEADERS)

            print(f"\n--- Running: {provider} ({model_string}) | Prompt: {prompt_version} ---")

            for _, jd_row in jd_df.iterrows():
                matched_cvs = cv_df[cv_df["Field"] == jd_row["Industry"]]
                for _, cv_row in matched_cvs.iterrows():
                    result_id = f"{jd_row['JD_ID']}_{cv_row['Frame_CV_ID']}"
                    if result_id in processed_ids: continue 

                    prompt = prompt_template.replace("{{JOB_DESCRIPTION}}", str(jd_row["Full_Job_Description"])).replace("{{COVER_LETTER}}", str(cv_row["Cover_Letter"]))
                    if prompt_version == "region": prompt = prompt.replace("{{REGION}}", str(jd_row["Region"]))

                    parsed_data, error_msg, raw_response = {}, "", ""
                    for attempt in range(5):
                        try:
                            time.sleep(delay)
                            raw_response = call_llm(provider, model_string, prompt)
                            parsed_data = json.loads(raw_response.strip().removeprefix("```json").removesuffix("```").strip())
                            break
                        except Exception as e:
                            error_msg = str(e)
                            backoff_time = (2 ** attempt) * 6
                            print(f"[Error] {provider} ({result_id}): {error_msg}. Retrying in {backoff_time}s...")
                            time.sleep(backoff_time)
                    
                    row_data = {
                        "Result_id": result_id, "Model_Name": model_string, "Prompt_Version": prompt_version,
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
                    print(f"[{provider}] Processed {result_id}")

if __name__ == "__main__":
    main()