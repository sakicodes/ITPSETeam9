"""
Stage 3 - Seniority detection & candidate selection.

Detects a rough seniority band per resume (Senior / Mid / Junior) from:
  - explicit "X years" mentions (an early, large number usually means senior; a small
    number or none at all usually means junior/early-career)
  - title-line seniority markers (Senior/Director/VP/Chief/Principal/CEO/CFO/COO/CTO/
    President vs Intern/Entry-level/Junior/Trainee)

NOTE: an earlier version of SENIOR_MARKERS included a bare "\\bmanager\\b", which
silently misclassified almost every genuine "Project Manager"/"Program Manager" titled
resume as Senior band (since "manager" alone isn't a seniority signal - it's the single
most common word in mid-level titles). That bug was the real reason the original
selection pass under-represented title-confirmed PM candidates in the final 90 - once
"manager" is removed, 30 of the 49 title-confirmed PM candidates read as genuinely
Mid-level, right on target.

Selection strategy per industry (post-audit remediation):
  - Sales/Business Development & Project/Programme Management: prefer candidates that
    are BOTH title-confirmed (Classifier_Confidence >= 0.85) AND Mid-band. Both have
    enough of these to hit 30 without falling back.
  - Procurement/Sourcing/Supply Chain: the corpus only contains ~10 genuinely
    title-confirmed Procurement resumes (a real corpus limitation, not a classifier
    bug - verified by manually reading the title lines of the next ~180 lower-confidence
    candidates, most of which are Chefs/Accountants/IT Directors whose resumes just
    mention "purchase order" or "inventory" in passing). The remaining 20 slots are
    filled from a manually-vetted allowlist of adjacent-function titles (Buyer/
    Storekeeper/Materials/Warehouse/Fulfillment/Merchandising/Supply Sergeant, etc.) -
    hardcoded below with the rationale, not randomly sampled, since this was a one-time
    manual review, not an automatable rule. Each selected row is labeled via
    `Selection_Basis` so this is transparent in the final output rather than silently
    blended in.
"""
import os
import re
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUTPUT_DIR = os.path.join(ROOT, "output")

INPUT_PATH = os.path.join(OUTPUT_DIR, "2.cv_pool_classified.csv")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "3.cv_pool_selected_90.csv")

RANDOM_SEED = 42
PER_INDUSTRY_TARGET = 30
TITLE_CONFIRMED_THRESHOLD = 0.85

YEARS_PATTERN = re.compile(r"(?:over|more than|nearly)?\s*(\d{1,2})\+?\s*years?", re.IGNORECASE)

SENIOR_MARKERS = re.compile(
    r"\bsenior\b|\bsr\.?\b|\bdirector\b|\bvice president\b|\bvp\b|\bchief\b|"
    r"\bprincipal\b|\bhead of\b|\bexecutive\b|\bceo\b|\bcfo\b|\bcoo\b|\bcto\b|\bpresident\b",
    re.IGNORECASE,
)
JUNIOR_MARKERS = re.compile(
    r"\bintern\b|\binternship\b|\bentry.?level\b|\bjunior\b|\bjr\.?\b|\btrainee\b|"
    r"\bapprentice\b|\bassistant\b|\bassociate\b",
    re.IGNORECASE,
)

# Section-header boundary reused from Stage 2's convention (title line precedes it).
SECTION_HEADER_PATTERN = re.compile(
    r"(professional summary|executive summary|career summary|summary|skills|highlights|"
    r"executive profile|professional profile|career overview|profile|overview|objective|experience)",
    re.IGNORECASE,
)

# Manually vetted during the post-audit remediation: read the title line of all 184
# body-only Procurement candidates, excluded the ones that are clearly a different
# profession mentioning a keyword in passing (Chef, Accountant, IT Director, Teacher,
# Designer, HR, Consultant, Banking Officer, etc.), and kept the ones whose title
# genuinely names a Procurement/Supply Chain-adjacent function even without matching
# the exact keyword list (e.g. "Purchaser", "Storekeeper", "Supply Sergeant").
MANUALLY_VETTED_PROCUREMENT_IDS = {
    "10189110": "PURCHASER / PRODUCTION COORDINATOR",
    "16850314": "STOREKEEPER II",
    "26586477": "ASSOCIATE MERCHANT",
    "70198580": "AVIATION SUPPLY TECHNICIAN",
    "16723524": "PRODUCTION CONTROL / SR. MERCHANDISER",
    "11432686": "CATEGORY BRAND MANAGER",
    "94137171": "PLANT FULFILLMENT LEADER",
    "11614114": "MATERIAL CONTROL SPECIALIST",
    "41586420": "STORE KEEPER / PRODUCTION CO-ORDINATOR",
    "34304175": "WAREHOUSE LEAD",
    "21912637": "CONTRACTS AND FINANCE OFFICER",
    "30344127": "SENIOR MATERIALS ANALYST",
    "65062795": "WMS CONSULTANT",
    "26829561": "INVENTORY ANALYST/MATERIALS PLANNER",
    "27176039": "ROUTE MANAGER",
    "17274759": "SENIOR SUPPLY SERGEANT",
    "20324037": "FULFILLMENT ADVOCATE",
    "HF_01448": "MASTER DATA MANAGER",
    "37750854": "SR. MERCHANDISING AUDIENCE LEAD, MICROSOFT US ONLINE STORE",
    "23631188": "ACCOUNTS PAYABLE (CREDITORS) SUPERVISOR",
}


def get_title_zone(text):
    match = SECTION_HEADER_PATTERN.search(text[:400])
    return text[: match.start()] if match else text[:100]


def detect_years(text):
    window = text[:500]
    years = [int(y) for y in YEARS_PATTERN.findall(window)]
    if not years:
        years = [int(y) for y in YEARS_PATTERN.findall(text)]
    return max(years) if years else None


def detect_seniority_band(text):
    title_zone = get_title_zone(text)
    years = detect_years(text)

    if years is not None:
        if years >= 12:
            return "Senior", years
        if years <= 2:
            return "Junior", years
        return "Mid", years

    if SENIOR_MARKERS.search(title_zone) and not JUNIOR_MARKERS.search(title_zone):
        return "Senior", None
    if JUNIOR_MARKERS.search(title_zone) and not SENIOR_MARKERS.search(title_zone):
        return "Junior", None
    return "Mid", None


def select_title_confirmed(df, industry, seed):
    """Sales & Project Management: prefer title-confirmed (>=0.85) + Mid-band."""
    pool = df[df["Industry"] == industry].copy()
    strong = pool[
        (pool["Classifier_Confidence"] >= TITLE_CONFIRMED_THRESHOLD)
        & (pool["Detected_Seniority_Raw"] == "Mid")
    ]

    if len(strong) >= PER_INDUSTRY_TARGET:
        selected = strong.sample(n=PER_INDUSTRY_TARGET, random_state=seed).copy()
        selected["Selection_Basis"] = (
            "Native category (direct)" if industry == "Sales/Business Development" else "Title-confirmed (>=0.85)"
        )
        return selected

    # Fallback (not expected to trigger for Sales or PM given current corpus, but kept
    # for robustness): widen to any Mid-band candidate, then any candidate at all.
    print(f"  [{industry}] only {len(strong)} title-confirmed Mid-band candidates - widening fallback.")
    mid_pool = pool[pool["Detected_Seniority_Raw"] == "Mid"]
    remainder_pool = pool.drop(mid_pool.index).sort_values("Classifier_Confidence", ascending=False)
    needed = PER_INDUSTRY_TARGET - len(strong)
    fill = pd.concat([mid_pool.drop(strong.index, errors="ignore"), remainder_pool]).head(needed)
    selected = pd.concat([strong, fill], ignore_index=True).copy()
    selected["Selection_Basis"] = "Title-confirmed (>=0.85)"
    selected.loc[selected["Classifier_Confidence"] < TITLE_CONFIRMED_THRESHOLD, "Selection_Basis"] = "Fallback (below title-confirmed bar)"
    return selected


def select_procurement(df, seed):
    """Procurement: title-confirmed (>=0.85) + manually-vetted adjacent-function allowlist."""
    pool = df[df["Industry"] == "Procurement/Sourcing/Supply Chain"].copy()

    strong = pool[pool["Classifier_Confidence"] >= TITLE_CONFIRMED_THRESHOLD].copy()
    strong["Selection_Basis"] = "Title-confirmed (>=0.85)"

    vetted_mask = pool["Resume_ID"].astype(str).isin(MANUALLY_VETTED_PROCUREMENT_IDS)
    vetted = pool[vetted_mask].copy()
    vetted["Selection_Basis"] = "Manually-vetted adjacent function"

    selected = pd.concat([strong, vetted], ignore_index=True)
    selected = selected.drop_duplicates(subset=["Resume_ID"])

    print(f"  [Procurement] title-confirmed: {len(strong)}, manually-vetted: {len(vetted)}, "
          f"combined unique: {len(selected)} (target {PER_INDUSTRY_TARGET})")
    print("  Seniority band among these 30:")
    print(selected["Detected_Seniority_Raw"].value_counts().to_string())

    return selected


def main():
    df = pd.read_csv(INPUT_PATH)
    df = df.dropna(subset=["Resume_Text"])

    seniority = df["Resume_Text"].apply(detect_seniority_band)
    df["Detected_Seniority_Raw"] = seniority.apply(lambda t: t[0])
    df["Detected_Years"] = seniority.apply(lambda t: t[1])

    print("Seniority band distribution per industry (pre-selection):")
    print(df.groupby(["Industry", "Detected_Seniority_Raw"]).size())
    print()

    sales = select_title_confirmed(df, "Sales/Business Development", RANDOM_SEED)
    pm = select_title_confirmed(df, "Project/Programme Management", RANDOM_SEED)
    proc = select_procurement(df, RANDOM_SEED)

    for name, sel in [("Sales", sales), ("PM", pm), ("Procurement", proc)]:
        print(f"{name}: selected {len(sel)}")

    final = pd.concat([proc, sales, pm], ignore_index=True)
    assert len(final) == 90, f"Expected 90 rows total, got {len(final)}"
    for industry in [
        "Procurement/Sourcing/Supply Chain", "Sales/Business Development", "Project/Programme Management"
    ]:
        count = (final["Industry"] == industry).sum()
        assert count == PER_INDUSTRY_TARGET, f"{industry} has {count} rows, expected {PER_INDUSTRY_TARGET}"

    final.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(final)} selected candidates -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
