"""
Stage 2 - Industry classification.

Sales/Business Development: pulled directly via Native_Category (Resume_cleaned pool)
or the equivalent HF integer label (HF pool uses labels 0-23 for the same 24-category
taxonomy as Resume_cleaned.csv - the mapping below was derived empirically by exact-text
matching HF rows against Resume_cleaned.csv rows, since ~99% of the HF train split turned
out to be the same underlying resumes).

Procurement/Sourcing/Supply Chain and Project/Programme Management have no native
category in either pool, so they're identified via a weighted keyword classifier:
hits in the first ~150 words (title/summary window, where job titles live) count more
than hits in the rest of the document. Classifier_Confidence = title-window hits vs
total hits, scaled, so low-confidence picks can be spot-checked later.
"""
import os
import re
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUTPUT_DIR = os.path.join(ROOT, "output")

INPUT_PATH = os.path.join(OUTPUT_DIR, "1.cv_pool_combined.csv")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "2.cv_pool_classified.csv")

# Empirically derived HF integer-label -> category-name mapping (see docstring above).
HF_LABEL_TO_CATEGORY = {
    0: "HR", 1: "DESIGNER", 2: "INFORMATION-TECHNOLOGY", 3: "TEACHER", 4: "ADVOCATE",
    5: "BUSINESS-DEVELOPMENT", 6: "HEALTHCARE", 7: "FITNESS", 8: "AGRICULTURE", 9: "BPO",
    10: "SALES", 11: "CONSULTANT", 12: "DIGITAL-MEDIA", 13: "AUTOMOBILE", 14: "CHEF",
    15: "FINANCE", 16: "APPAREL", 17: "ENGINEERING", 18: "ACCOUNTANT", 19: "CONSTRUCTION",
    20: "PUBLIC-RELATIONS", 21: "BANKING", 22: "ARTS", 23: "AVIATION",
}

SALES_CATEGORIES = {"SALES", "BUSINESS-DEVELOPMENT"}

# Resumes in both pools consistently open with an ALL-CAPS job title line, followed by
# a section header (Summary / Skills / Highlights / Profile / Overview / Objective /
# Experience). We isolate that title line as the "title zone" and weight keyword hits
# there far more heavily than hits in the body - a passing mention of "vendor sourcing"
# in a skills bullet list on a Designer resume should not outrank an actual job title.
SECTION_HEADER_PATTERN = re.compile(
    r"(professional summary|executive summary|career summary|summary|skills|highlights|"
    r"executive profile|professional profile|career overview|profile|overview|objective|experience)",
    re.IGNORECASE,
)
TITLE_ZONE_FALLBACK_CHARS = 100
TITLE_WEIGHT = 10
BODY_WEIGHT = 1


def get_title_zone(text):
    match = SECTION_HEADER_PATTERN.search(text[:400])
    if match:
        return text[: match.start()]
    return text[:TITLE_ZONE_FALLBACK_CHARS]

PROCUREMENT_KEYWORDS = [
    r"procurement", r"supply chain", r"sourcing", r"purchasing", r"\bbuyer\b",
    r"vendor management", r"\bsupplier\b", r"commodity manager", r"category manager",
    r"strategic sourcing", r"purchase order",
    # Widened after the audit found only ~8 title-confirmed Procurement resumes exist
    # in the whole corpus under the original narrower list - these adjacent titles are
    # genuinely part of the Procurement/Supply Chain function, not scope creep.
    r"inventory control", r"materials manager", r"materials management",
    r"logistics coordinator", r"contract negotiation", r"\brfp\b", r"\brfq\b",
    r"warehouse operations", r"import/export", r"import export", r"spend analysis",
    r"inventory manager", r"supply chain manager", r"supply chain analyst",
    r"purchasing manager", r"purchasing agent", r"logistics manager",
]
PROJECT_MGMT_KEYWORDS = [
    r"project manager", r"program manager", r"programme manager", r"\bpmp\b",
    r"scrum master", r"project lead", r"delivery manager", r"project management",
    r"program management", r"programme management",
]


def resolve_native_category(row):
    if row["Source_Pool"] == "Resume_cleaned":
        return row["Native_Category"].upper()
    else:
        try:
            return HF_LABEL_TO_CATEGORY.get(int(row["Native_Category"]), "UNKNOWN")
        except (ValueError, TypeError):
            return "UNKNOWN"


def keyword_score(text, keywords):
    text_lower = text.lower()
    title_zone = get_title_zone(text_lower)
    title_hits = sum(len(re.findall(pat, title_zone)) for pat in keywords)
    total_hits = sum(len(re.findall(pat, text_lower)) for pat in keywords)
    body_hits = max(total_hits - title_hits, 0)
    weighted = title_hits * TITLE_WEIGHT + body_hits * BODY_WEIGHT
    return title_hits, body_hits, weighted


def classify_row(row):
    category = row["Resolved_Category"]
    text = row["Resume_Text"]

    if category in SALES_CATEGORIES:
        return "Sales/Business Development", 1.0

    proc_title, proc_body, proc_score = keyword_score(text, PROCUREMENT_KEYWORDS)
    pm_title, pm_body, pm_score = keyword_score(text, PROJECT_MGMT_KEYWORDS)

    # Require a genuine signal: either the title line itself names the role, or the
    # body mentions it repeatedly (a real function of the job, even if the title line
    # uses a different label e.g. "Supervisor" who actually runs procurement).
    MIN_BODY_HITS_WITHOUT_TITLE = 4
    proc_qualifies = proc_title >= 1 or proc_body >= MIN_BODY_HITS_WITHOUT_TITLE
    pm_qualifies = pm_title >= 1 or pm_body >= MIN_BODY_HITS_WITHOUT_TITLE

    if not proc_qualifies and not pm_qualifies:
        return None, 0.0

    if proc_score >= pm_score and proc_qualifies:
        industry, title_hits, body_hits = "Procurement/Sourcing/Supply Chain", proc_title, proc_body
    elif pm_qualifies:
        industry, title_hits, body_hits = "Project/Programme Management", pm_title, pm_body
    else:
        industry, title_hits, body_hits = "Procurement/Sourcing/Supply Chain", proc_title, proc_body

    # Confidence: a genuine title-line match is high-confidence on its own; body-only
    # matches (no title hit) are capped lower since they rely on inference.
    if title_hits >= 1:
        confidence = min(1.0, 0.7 + title_hits * 0.15)
    else:
        confidence = min(0.5, body_hits * 0.08)
    return industry, round(confidence, 2)


def main():
    df = pd.read_csv(INPUT_PATH)
    df = df.dropna(subset=["Resume_Text"])

    df["Resolved_Category"] = df.apply(resolve_native_category, axis=1)

    results = df.apply(classify_row, axis=1, result_type="expand")
    results.columns = ["Industry", "Classifier_Confidence"]
    df = pd.concat([df, results], axis=1)

    classified = df[df["Industry"].notna()].copy()

    print("Candidate counts per industry (pre-selection):")
    print(classified["Industry"].value_counts())
    print()
    print("Classifier_Confidence distribution for Procurement/PM (non-Sales):")
    non_sales = classified[classified["Industry"] != "Sales/Business Development"]
    print(non_sales["Classifier_Confidence"].describe())

    classified.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(classified)} classified candidates -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
