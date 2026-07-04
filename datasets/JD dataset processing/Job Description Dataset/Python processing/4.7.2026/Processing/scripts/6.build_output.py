"""
Stage 6 - Assemble the final output CSV.

Merges the neutralized/standardized 90-row table into the deliverable schema, assigns
CV_ID (industry-prefixed, e.g. PROC-001 / SALES-001 / PM-001), and asserts the 30/30/30
split before writing.
"""
import os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUTPUT_DIR = os.path.join(ROOT, "output")

INPUT_PATH = os.path.join(OUTPUT_DIR, "5.cv_pool_neutralized_90_INTERMEDIATE.csv")
# FINAL deliverable - the 90-row, 30/30/30 neutralized/standardized CV pool.
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "6.neutral_cv_pool_90_FINAL.csv")

INDUSTRY_PREFIX = {
    "Procurement/Sourcing/Supply Chain": "PROC",
    "Sales/Business Development": "SALES",
    "Project/Programme Management": "PM",
}


def assign_cv_ids(df):
    cv_ids = []
    counters = {prefix: 0 for prefix in INDUSTRY_PREFIX.values()}
    for industry in df["Industry"]:
        prefix = INDUSTRY_PREFIX[industry]
        counters[prefix] += 1
        cv_ids.append(f"{prefix}-{counters[prefix]:03d}")
    return cv_ids


def main():
    df = pd.read_csv(INPUT_PATH)

    df["CV_ID"] = assign_cv_ids(df)
    df["Word_Count"] = df["Neutral_Resume_Text"].str.split().str.len()
    # Drop the raw/unresolved category (string name for Resume_cleaned rows, bare HF
    # integer label for HF rows) in favor of Resolved_Category, which is human-readable
    # for both pools - keeping both would collide on rename below.
    df = df.drop(columns=["Native_Category"]).rename(columns={
        "Resolved_Category": "Native_Category",
        "Resume_Text": "Original_Resume_Text",
    })

    final_cols = [
        "CV_ID", "Resume_ID", "Source_Pool", "Industry", "Native_Category",
        "Seniority_Level", "Word_Count", "Classifier_Confidence", "Selection_Basis",
        "Substitutions_Made", "Residual_Agentic_Hits",
        "Residual_Communal_Hits", "Original_Resume_Text", "Neutral_Resume_Text",
    ]
    final = df[final_cols]

    assert len(final) == 90, f"Expected 90 rows, got {len(final)}"
    counts = final["Industry"].value_counts()
    for industry, prefix in INDUSTRY_PREFIX.items():
        assert counts.get(industry, 0) == 30, f"{industry} has {counts.get(industry, 0)} rows, expected 30"

    final.to_csv(OUTPUT_PATH, index=False)
    print("Final row counts per industry:")
    print(counts)
    print(f"\nTotal rows: {len(final)}")
    print(f"Saved final deliverable -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
