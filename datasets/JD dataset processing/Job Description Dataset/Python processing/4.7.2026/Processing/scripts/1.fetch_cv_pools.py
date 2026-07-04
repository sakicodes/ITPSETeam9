"""
Stage 1 - Fetch & cache CV pools.

Loads the two designated resume pools:
  1. Resume_cleaned.csv (local)
  2. HuggingFace syedroshanzameer/resume-classification (cached locally on first run)

Normalizes both to a common shape:
  Source_Pool, Resume_ID, Resume_Text, Native_Category

Native_Category is left as-is per pool (Resume_cleaned.csv uses string category names;
the HF dataset uses integer labels 0-23, kept as-is here since Stage 2 classifies by
resume text content, not by this label).
"""
import os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
INPUT_DIR = os.path.join(ROOT, "input")
OUTPUT_DIR = os.path.join(ROOT, "output")

RESUME_CLEANED_PATH = os.path.join(INPUT_DIR, "Resume_cleaned.csv")
HF_CACHE_PATH = os.path.join(OUTPUT_DIR, "1.hf_resumes_raw_cache.csv")
HF_DATA_URI = "hf://datasets/syedroshanzameer/resume-classification/train.csv"

OUTPUT_PATH = os.path.join(OUTPUT_DIR, "1.cv_pool_combined.csv")


def load_resume_cleaned():
    df = pd.read_csv(RESUME_CLEANED_PATH, engine="python", on_bad_lines="skip",
                      usecols=[0, 1, 3], names=["ID", "Resume_str", "Category"], header=0)
    df = df.dropna(subset=["Resume_str"])
    out = pd.DataFrame({
        "Source_Pool": "Resume_cleaned",
        "Resume_ID": df["ID"].astype(str),
        "Resume_Text": df["Resume_str"],
        "Native_Category": df["Category"].astype(str).str.strip(),
    })
    return out


def load_hf_resumes():
    if os.path.exists(HF_CACHE_PATH):
        print(f"Using cached HF dataset: {HF_CACHE_PATH}")
        hf = pd.read_csv(HF_CACHE_PATH)
    else:
        print("Fetching HF dataset (syedroshanzameer/resume-classification)...")
        hf = pd.read_csv(HF_DATA_URI)
        hf.to_csv(HF_CACHE_PATH, index=False)
        print(f"Cached HF dataset to {HF_CACHE_PATH}")

    hf = hf.dropna(subset=["text"])
    out = pd.DataFrame({
        "Source_Pool": "HF_resume_classification",
        "Resume_ID": ["HF_" + str(i).zfill(5) for i in range(1, len(hf) + 1)],
        "Resume_Text": hf["text"],
        "Native_Category": hf["labels"].astype(str),
    })
    return out


def main():
    rc = load_resume_cleaned()
    hf = load_hf_resumes()

    combined = pd.concat([rc, hf], ignore_index=True)
    combined = combined.drop_duplicates(subset=["Resume_Text"])

    combined.to_csv(OUTPUT_PATH, index=False)
    print(f"Resume_cleaned pool: {len(rc)} rows")
    print(f"HF pool: {len(hf)} rows")
    print(f"Combined (deduped): {len(combined)} rows -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
