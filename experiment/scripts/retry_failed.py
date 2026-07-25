import os
import csv
import json
import time
import datetime
import argparse
import re
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv

from openai import OpenAI
from anthropic import Anthropic
from google import genai
from google.genai import types
from mistralai.client import Mistral

load_dotenv()

MODELS = {
    "openai": "gpt-4.1-mini",
    "anthropic": "claude-haiku-4-5-20251001",
    "google": "gemini-2.5-flash",
    "tencent": "hy3", 
    "minimax": "MiniMax-M3",
    "deepseek": "deepseek-v4-pro",
    "qwen": "qwen3.7-plus",
    "mistral": "mistral-medium-latest",
    "sealion": "aisingapore/Qwen-SEA-LION-v4.5-27B-IT"
}

PROVIDER_DELAYS = {
    "sealion": 6.5,
    "openai": 1.0, "anthropic": 1.0, "google": 1.0, "mistral": 1.0,
    "tencent": 1.0, "minimax": 1.0, "deepseek": 1.0, "qwen": 1.0
}

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
        
        client_kwargs = {
            "api_key": api_keys[provider], 
            "base_url": base_urls[provider]
        }
        if provider == "qwen":
            client_kwargs["timeout"] = 300.0
            
        client = OpenAI(**client_kwargs)
        res = client.chat.completions.create(
            model=model_string, messages=[{"role": "user", "content": prompt_text + "\n\nEnsure output is strictly JSON format."}],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        return res.choices[0].message.content

def needs_retry(row):
    # Condition 1: Check if Error column has text (Catches Mistral's failures)
    err = row.get("Error")
    if pd.notna(err) and str(err).strip() != "" and str(err).lower() != "nan":
        return True
    
    # Condition 2: Check if Hireability Score is completely empty (Catches Sealion's silent failures)
    score = row.get("Hireability_Score")
    if pd.isna(score) or str(score).strip() == "" or str(score).lower() == "nan":
        return True
        
    return False

def main():
    parser = argparse.ArgumentParser(description="Retry failed LLM evaluations.")
    parser.add_argument(
        "--provider", 
        type=str, 
        required=True,
        choices=list(MODELS.keys()),
        help="Specify which model provider to retry (e.g., 'mistral', 'sealion')."
    )
    args = parser.parse_args()

    provider = args.provider
    model_string = MODELS[provider]
    delay = PROVIDER_DELAYS.get(provider, 1.0)

    base_dir = Path.cwd()
    outputs_dir = base_dir / "outputs"

    cv_df = pd.read_csv(base_dir / "datasets" / "cover_letters" / "cover_letters_final.csv")
    jd_df = pd.read_csv(base_dir / "datasets" / "jds" / "jds_final.csv")

    with open(base_dir / "prompt_design" / "prompt-design.md", "r", encoding="utf-8") as f:
        prompt_standard_template = f.read()
    with open(base_dir / "prompt_design" / "prompt-design-region.md", "r", encoding="utf-8") as f:
        prompt_region_template = f.read()

    sgt = datetime.timezone(datetime.timedelta(hours=8))

    for prompt_version, prompt_template in [("standard", prompt_standard_template), ("region", prompt_region_template)]:
        output_file = outputs_dir / f"results_{provider}_{prompt_version}.csv"
        
        if not output_file.exists():
            print(f"File not found: {output_file.name}. Skipping.")
            continue
            
        print(f"\n--- Retrying: {provider} ({model_string}) | Prompt: {prompt_version} ---")
        
        # Load the existing results and cast all columns to object to avoid dtype assignment errors
        results_df = pd.read_csv(output_file, dtype=str)
        results_df = results_df.astype(object)
        
        # Identify rows to retry
        retry_indices = []
        for index, row in results_df.iterrows():
            if needs_retry(row):
                retry_indices.append(index)
                
        if not retry_indices:
            print(f"No failed rows found for {prompt_version}. All good!")
            continue
            
        retry_ids = [results_df.loc[i, 'Result_id'] for i in retry_indices]
        print(f"Found {len(retry_indices)} rows to retry: {retry_ids}")
        
        jd_dict = jd_df.set_index("JD_ID").to_dict("index")
        cv_dict = cv_df.set_index("Frame_CV_ID").to_dict("index")

        for idx in retry_indices:
            row = results_df.loc[idx]
            result_id = row["Result_id"]
            jd_id = row["JD_ID"]
            cv_id = row["Frame_CV_ID"]
            
            jd_info = jd_dict.get(jd_id)
            cv_info = cv_dict.get(cv_id)
            
            if not jd_info or not cv_info:
                print(f"[Error] Could not find original data for {result_id}")
                continue
                
            prompt = prompt_template.replace("{{JOB_DESCRIPTION}}", str(jd_info["Full_Job_Description"])).replace("{{COVER_LETTER}}", str(cv_info["Cover_Letter"]))
            if prompt_version == "region": 
                prompt = prompt.replace("{{REGION}}", str(jd_info["Region"]))

            parsed_data, error_msg, raw_response = {}, "", ""
            
            for attempt in range(5):
                try:
                    time.sleep(delay)
                    raw_response = call_llm(provider, model_string, prompt)
                    
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
                    print(f"[Retry Error] {provider} ({result_id}): {error_msg}. Retrying in {backoff_time}s...")
                    time.sleep(backoff_time)
            
            # Update the dataframe row in-place safely
            results_df.at[idx, "Timestamp_SGT"] = datetime.datetime.now(sgt).strftime("%Y-%m-%d %H:%M:%S")
            results_df.at[idx, "Hireability_Score"] = str(parsed_data.get("hireability", {}).get("score", ""))
            results_df.at[idx, "Hireability_Reason"] = str(parsed_data.get("hireability", {}).get("reason", ""))
            results_df.at[idx, "Competence_Score"] = str(parsed_data.get("perceived_competence", {}).get("score", ""))
            results_df.at[idx, "Competence_Reason"] = str(parsed_data.get("perceived_competence", {}).get("reason", ""))
            results_df.at[idx, "Fit_Score"] = str(parsed_data.get("person_job_fit", {}).get("score", ""))
            results_df.at[idx, "Fit_Reason"] = str(parsed_data.get("person_job_fit", {}).get("reason", ""))
            results_df.at[idx, "Performance_Score"] = str(parsed_data.get("expected_performance", {}).get("score", ""))
            results_df.at[idx, "Performance_Reason"] = str(parsed_data.get("expected_performance", {}).get("reason", ""))
            results_df.at[idx, "Leadership_Exhibited"] = str(parsed_data.get("leadership_potential", {}).get("leadership_exhibited", ""))
            results_df.at[idx, "Leadership_Control"] = str(parsed_data.get("leadership_potential", {}).get("control_over_activities", ""))
            results_df.at[idx, "Leadership_Effective"] = str(parsed_data.get("leadership_potential", {}).get("effective_leader", ""))
            results_df.at[idx, "Leadership_Avg"] = str(parsed_data.get("leadership_potential", {}).get("average", ""))
            results_df.at[idx, "Leadership_Reason"] = str(parsed_data.get("leadership_potential", {}).get("reason", ""))
            results_df.at[idx, "Status_Score"] = str(parsed_data.get("expected_status", {}).get("score", ""))
            results_df.at[idx, "Status_Reason"] = str(parsed_data.get("expected_status", {}).get("reason", ""))
            results_df.at[idx, "Warmth_Score"] = str(parsed_data.get("perceived_warmth", {}).get("score", ""))
            results_df.at[idx, "Warmth_Reason"] = str(parsed_data.get("perceived_warmth", {}).get("reason", ""))
            results_df.at[idx, "Recommendation"] = str(parsed_data.get("recommendation", ""))
            results_df.at[idx, "Raw_JSON"] = raw_response if error_msg else ""
            results_df.at[idx, "Error"] = error_msg if not parsed_data else ""
            
            # Save CSV after every successful update
            results_df.to_csv(output_file, index=False)
            print(f"[{provider}] Successfully updated {result_id}")

if __name__ == "__main__":
    main()