import os
import csv
import json
import time
import datetime
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Limits config
MODEL = "meta-llama/llama-3.3-70b-instruct"

PROVIDER_ORDER = ["groq"]

MAX_DAILY_TOKENS = 100_000_000
MAX_DAILY_REQUESTS = 1_000_000
DELAY_BETWEEN_CALLS = 0.5

# Pilot mode: max NEW rows generated per prompt version, per session.
# BOTH versions run, so one session produces up to PILOT_LIMIT * 2 rows.
# The cap is per-session, not cumulative. Set to None for the full job list.
PILOT_LIMIT = None

# Shuffle JD order so an interrupted run still leaves coverage across
# regions and seniority levels rather than only the first few JDs.
# Fixed seed keeps the execution order reproducible.
SHUFFLE_SEED = 42

# Sort the output CSVs by JD_ID / Frame_CV_ID once the run finishes.
SORT_OUTPUT_ON_FINISH = True

# USD per 1M tokens, used for the running cost estimate only.
PRICE_IN_PER_M = 0.59
PRICE_OUT_PER_M = 0.79

# Measured cost per evaluation from the pilot session summary.
# Update this if your pilot reported something different.
EST_COST_PER_EVAL = 0.00092

CSV_HEADERS = [
    "Result_id", "Model_Name", "Provider_Served", "Prompt_Version", "Timestamp_SGT",
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
                data.setdefault("prompt_tokens", 0)
                data.setdefault("completion_tokens", 0)
                return data
    return {"date": today_str, "tokens": 0, "requests": 0,
            "prompt_tokens": 0, "completion_tokens": 0}

def save_tracker(tracker_file, data):
    with open(tracker_file, "w") as f:
        json.dump(data, f)

def estimate_cost(usage_data):
    return (usage_data["prompt_tokens"] / 1_000_000 * PRICE_IN_PER_M
            + usage_data["completion_tokens"] / 1_000_000 * PRICE_OUT_PER_M)

def call_model(client, prompt_text):
    res = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt_text}],
        temperature=0.0,
        seed=42,
        max_tokens=1200,
        response_format={"type": "json_object"},
        extra_body={
            "provider": {
                "order": PROVIDER_ORDER,
                "allow_fallbacks": False,
                "require_parameters": True,
            }
        },
    )
    content = res.choices[0].message.content
    extra = getattr(res, "model_extra", None) or {}
    return content, res.usage, extra.get("provider", "")

def sort_output_file(path):
    """Re-sort a finished results file into JD / CV order."""
    if not path.exists():
        return
    df = pd.read_csv(path)
    if df.empty:
        return
    df = df.sort_values(["JD_ID", "Frame_CV_ID"], kind="stable")
    df.to_csv(path, index=False)
    print(f"[sorted] {path.name} ({len(df)} rows)")

def main():
    base_dir = Path(__file__).parent.parent
    outputs_dir = base_dir / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    
    tracker_file = base_dir / "openrouter_state.json"
    usage_data = load_tracker(tracker_file)

    cv_df = pd.read_csv(base_dir / "datasets" / "cover_letters" / "cover_letters_final.csv")
    jd_df = pd.read_csv(base_dir / "datasets" / "jds" / "jds_final.csv")

    # Randomise JD execution order (see SHUFFLE_SEED note above)
    if SHUFFLE_SEED is not None:
        jd_df = jd_df.sample(frac=1, random_state=SHUFFLE_SEED).reset_index(drop=True)

    with open(base_dir / "prompt_design" / "prompt-design.md", "r", encoding="utf-8") as f:
        prompt_standard_template = f.read()
    with open(base_dir / "prompt_design" / "prompt-design-region.md", "r", encoding="utf-8") as f:
        prompt_region_template = f.read()

    sgt = datetime.timezone(datetime.timedelta(hours=8))
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )

    # ---- Denominator: how big is this run, and what will it cost? ----
    jobs_per_version = sum(
        len(cv_df[cv_df["Field"] == jd["Industry"]]) for _, jd in jd_df.iterrows()
    )
    total_calls = jobs_per_version * 2
    print(f"\n{len(jd_df)} JDs | {jobs_per_version} JD x CV pairs "
          f"| x2 prompt versions = {total_calls} calls")
    print(f"Estimated cost at ${EST_COST_PER_EVAL:.5f}/eval: "
          f"${total_calls * EST_COST_PER_EVAL:.2f}")
    print(f"Estimated wall clock at {DELAY_BETWEEN_CALLS}s delay + ~2s/call: "
          f"{total_calls * (DELAY_BETWEEN_CALLS + 2) / 60:.0f} min")

    if PILOT_LIMIT is not None:
        print(f"\n*** PILOT MODE: {PILOT_LIMIT} new rows per prompt version "
              f"= up to {PILOT_LIMIT * 2} calls this session (standard + region) ***")
    else:
        print("\n*** FULL RUN (PILOT_LIMIT = None) ***")

    for prompt_version, prompt_template in [("standard", prompt_standard_template), ("region", prompt_region_template)]:
        output_file = outputs_dir / f"results_openrouter_{prompt_version}.csv"
        processed_ids = set()
        
        if output_file.exists():
            with open(output_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                processed_ids.update(row["Result_id"] for row in reader)
        else:
            with open(output_file, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(CSV_HEADERS)

        print(f"\n--- Running: OpenRouter ({MODEL}) via {PROVIDER_ORDER} | Prompt: {prompt_version} ---")
        print(f"{len(processed_ids)} rows already present, "
              f"{jobs_per_version - len(processed_ids)} remaining")

        rows_this_version = 0
        stop_version = False

        for _, jd_row in jd_df.iterrows():
            if stop_version:
                break
            matched_cvs = cv_df[cv_df["Field"] == jd_row["Industry"]]
            for _, cv_row in matched_cvs.iterrows():

                # Pilot cap: stop this prompt version after N new rows
                if PILOT_LIMIT is not None and rows_this_version >= PILOT_LIMIT:
                    print(f"\n[PILOT] '{prompt_version}' capped at {PILOT_LIMIT} new rows. "
                          f"Moving to next prompt version if any.")
                    stop_version = True
                    break

                # Check daily limits before generating the next ID
                if usage_data["tokens"] >= MAX_DAILY_TOKENS or usage_data["requests"] >= MAX_DAILY_REQUESTS:
                    print(f"\n[STOP] Reached configured daily cap: {usage_data['tokens']} tokens, {usage_data['requests']} requests.")
                    print("Please run this script again tomorrow.")
                    return # Exit the script completely

                result_id = f"{jd_row['JD_ID']}_{cv_row['Frame_CV_ID']}"
                if result_id in processed_ids: continue 

                prompt = prompt_template.replace("{{JOB_DESCRIPTION}}", str(jd_row["Full_Job_Description"])).replace("{{COVER_LETTER}}", str(cv_row["Cover_Letter"]))
                if prompt_version == "region": prompt = prompt.replace("{{REGION}}", str(jd_row["Region"]))

                parsed_data, error_msg, raw_response, provider_served = {}, "", "", ""
                for attempt in range(5):
                    try:
                        time.sleep(DELAY_BETWEEN_CALLS)
                        raw_response, usage, provider_served = call_model(client, prompt)

                        # Update and save tracker immediately
                        usage_data["tokens"] += usage.total_tokens
                        usage_data["prompt_tokens"] += usage.prompt_tokens
                        usage_data["completion_tokens"] += usage.completion_tokens
                        usage_data["requests"] += 1
                        save_tracker(tracker_file, usage_data)

                        if provider_served and provider_served.lower() not in [p.lower() for p in PROVIDER_ORDER]:
                            print(f"[WARN] Provider pin did not hold: got '{provider_served}', expected {PROVIDER_ORDER}")

                        parsed_data = json.loads(raw_response.strip().removeprefix("```json").removesuffix("```").strip())
                        break
                    except Exception as e:
                        error_msg = str(e)
                        backoff_time = (2 ** attempt) * 6
                        print(f"[Error] OpenRouter ({result_id}): {error_msg}. Retrying in {backoff_time}s...")
                        time.sleep(backoff_time)
                
                row_data = {
                    "Result_id": result_id, "Model_Name": MODEL, "Provider_Served": provider_served,
                    "Prompt_Version": prompt_version,
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
                    "Recommendation": parsed_data.get("recommendation", ""), "Raw_JSON": raw_response, "Error": error_msg if not parsed_data else ""
                }
                
                with open(output_file, "a", newline="", encoding="utf-8") as f:
                    csv.DictWriter(f, fieldnames=CSV_HEADERS).writerow(row_data)

                rows_this_version += 1
                cap_str = str(PILOT_LIMIT) if PILOT_LIMIT else str(jobs_per_version)
                print(f"[OpenRouter] [{rows_this_version}/{cap_str}] {prompt_version} | "
                      f"{result_id} | {provider_served or 'unknown'} "
                      f"| Est. ${estimate_cost(usage_data):.3f}")

    if SORT_OUTPUT_ON_FINISH:
        print()
        for v in ("standard", "region"):
            sort_output_file(outputs_dir / f"results_openrouter_{v}.csv")

    print("\n=== Session totals ===")
    print(f"Requests:      {usage_data['requests']}")
    print(f"Input tokens:  {usage_data['prompt_tokens']}")
    print(f"Output tokens: {usage_data['completion_tokens']}")
    print(f"Estimated cost: ${estimate_cost(usage_data):.3f}")
    if usage_data["requests"]:
        per_call = estimate_cost(usage_data) / usage_data["requests"]
        print(f"Per evaluation: ${per_call:.5f}  ->  $5 = ~{int(5 / per_call)} evaluations")

if __name__ == "__main__":
    main()