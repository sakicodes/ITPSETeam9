"""
Stage 4/5 - Neutral-framing rewrite + seniority standardization.

Stage 4: applies the word-boundary-safe substitutions authored in
neutral_substitutions.csv (build_neutral_substitutions.py) to each selected resume's
text, preserving the original word's capitalization pattern. Counts replacements made.

An earlier version of this stage also cross-checked the rewritten text against
filtered_phrases.csv (flagging a row `Needs_Review` if a verbatim Agentic/Communal
phrase from that bank survived). That check was dropped after manual review showed it
was mostly a false-positive generator: filtered_phrases.csv is 82.8% labeled "Agentic",
so an exact-text match against it mostly just meant "this is an ordinary resume
sentence", not "this text is actually biased." Reading through every flagged match by
hand across all 90 rows found only 2 rows (3 phrases) with genuine solo/individual-glory
framing that the word-level dictionary substitution couldn't catch on its own - those
are now fixed directly via MANUAL_PHRASE_FIXES below, applied right after the
dictionary-substitution pass. filtered_phrases.csv is no longer read anywhere in this
pipeline as a result.

Stage 5: standardizes seniority language uniformly to Mid-level - normalizes explicit
"X years" mentions to a fixed range and strips unambiguous seniority-coded title words
(Senior/Director/VP/Chief/Principal, Junior/Intern/Entry-level/Trainee). All 90 selected
candidates were already detected as Mid-level in Stage 3, so this is mostly a consistency
pass rather than heavy rewriting.
"""
import os
import re
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
INPUT_DIR = os.path.join(ROOT, "input")
OUTPUT_DIR = os.path.join(ROOT, "output")

SELECTED_PATH = os.path.join(OUTPUT_DIR, "3.cv_pool_selected_90.csv")
SUBSTITUTIONS_PATH = os.path.join(OUTPUT_DIR, "4.neutral_substitutions.csv")
AGENTIC_DICT_PATH = os.path.join(INPUT_DIR, "agentic_expanded.xlsx")
COMMUNAL_DICT_PATH = os.path.join(INPUT_DIR, "communal_expanded.xlsx")

# INTERMEDIATE output - not the final deliverable. Stage 6 (6.build_output.py) takes
# this, trims the internal scaffolding columns, and writes the actual final CSV.
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "5.cv_pool_neutralized_90_INTERMEDIATE.csv")

STANDARD_YEARS_PHRASE = "5-7 years"
# NOTE: the optional prefix words' internal \s* must stay INSIDE that group - putting it
# outside (a bug caught during verification) let the pattern also swallow the leading
# whitespace/newline before the digit, producing glued-together output like
# "Summary5-7 years" instead of "Summary 5-7 years".
YEARS_MENTION_PATTERN = re.compile(
    r"(?:(?:over|more than|nearly)\s*)?\d{1,2}\+?\s*years?", re.IGNORECASE
)

# Unambiguous seniority-coded title words only (not "Manager"/"Lead" - those are
# legitimate neutral mid-level titles on their own, and "lead" is already handled by
# the Agentic "leadership" substitution above).
# NOTE: "\bsr\.\b" (with a trailing \b) never matches when followed by whitespace - a
# period and a space are both non-word characters, so there's no word/non-word
# transition for \b to anchor on there. Anchoring on the period itself is sufficient.
SENIOR_TITLE_REPLACEMENTS = {
    r"\bsenior\b": "",
    r"\bsr\.": "",
    r"\bdirector\b": "manager",
    r"\bvice president\b": "manager",
    r"\bvp\b": "manager",
    r"\bchief\b": "lead",
    r"\bprincipal\b": "",
    r"\bexecutive\b": "",
}
JUNIOR_TITLE_REPLACEMENTS = {
    r"\bjunior\b": "",
    r"\bjr\.": "",
    r"\bentry.level\b": "",
    r"\btrainee\b": "associate",
    r"\bapprentice\b": "associate",
}

# Manually reviewed and rewritten - the only 2 rows (3 phrases, out of 90) found to
# carry genuine solo/individual-glory framing that word-level substitution missed.
# Keyed by Resume_ID so the fix survives a full pipeline re-run.
MANUAL_PHRASE_FIXES = {
    "26829350": [  # PROC-003
        (
            "I single handedly manage global procurement email-box to resolve and execute internal client request and queries.",
            "Manages the global procurement email-box to resolve and execute internal client requests and queries.",
        ),
        (
            "Single handed support to global buyers in req to PO creation process for pre-approved categories.",
            "Provides support to global buyers in the req to PO creation process for pre-approved categories.",
        ),
    ],
    "33578873": [  # SALES-026
        (
            "I am looking for a career position with a company that I can be rewarded by my desire to succeed.",
            "I am looking for a career position with a company that recognizes and rewards strong performance.",
        ),
    ],
}


def load_substitution_map():
    df = pd.read_csv(SUBSTITUTIONS_PATH)
    df = df.dropna(subset=["replacement"])
    df = df[df["replacement"].astype(str).str.strip() != ""]
    # Longest keyword first, so multi-word phrases (e.g. "self-starter") match before
    # any shorter overlapping single-word rules could interfere.
    df = df.assign(kw_len=df["keyword"].str.len()).sort_values("kw_len", ascending=False)
    return list(zip(df["keyword"], df["replacement"]))


def match_case(original_word, replacement):
    if original_word.isupper():
        return replacement.upper()
    if original_word[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def apply_substitutions(text, substitution_map):
    count = 0

    def make_replacer(replacement):
        def _replace(m):
            nonlocal count
            count += 1
            return match_case(m.group(0), replacement)
        return _replace

    for keyword, replacement in substitution_map:
        pattern = re.compile(r"\b" + re.escape(keyword) + r"\b", re.IGNORECASE)
        text = pattern.sub(make_replacer(replacement), text)

    return text, count


def apply_manual_fixes(text, resume_id):
    count = 0
    for old_phrase, new_phrase in MANUAL_PHRASE_FIXES.get(str(resume_id), []):
        if old_phrase in text:
            text = text.replace(old_phrase, new_phrase)
            count += 1
    return text, count


def _title_case_replacer(replacement):
    def _replace(m):
        if not replacement:
            return ""
        return match_case(m.group(0), replacement)
    return _replace


def standardize_seniority(text):
    text = YEARS_MENTION_PATTERN.sub(STANDARD_YEARS_PHRASE, text)

    # Case-preserving: title lines are often ALL CAPS ("...BUSINESS DEVELOPMENT
    # DIRECTOR"), and a literal lowercase replacement there would visually break the
    # header (an earlier verification pass caught "DIRECTOR" -> "manager" mid-caps).
    for pattern, replacement in SENIOR_TITLE_REPLACEMENTS.items():
        text = re.sub(pattern, _title_case_replacer(replacement), text, flags=re.IGNORECASE)
    for pattern, replacement in JUNIOR_TITLE_REPLACEMENTS.items():
        text = re.sub(pattern, _title_case_replacer(replacement), text, flags=re.IGNORECASE)

    # Collapse any double spaces left behind by word removals.
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\s+([,.])", r"\1", text)
    return text


def count_residual_dictionary_hits(text, keywords):
    text_lower = text.lower()
    total = 0
    for kw in keywords:
        if re.search(r"\b" + re.escape(kw.lower()) + r"\b", text_lower):
            total += 1
    return total


def main():
    selected = pd.read_csv(SELECTED_PATH)
    substitution_map = load_substitution_map()

    agentic_keywords = pd.read_excel(AGENTIC_DICT_PATH)["keyword"].astype(str).tolist()
    communal_keywords = pd.read_excel(COMMUNAL_DICT_PATH)["keyword"].astype(str).tolist()

    neutral_texts, substitutions_made = [], []
    residual_agentic, residual_communal = [], []
    manual_fixes_applied = 0

    for _, row in selected.iterrows():
        original = row["Resume_Text"]

        rewritten, sub_count = apply_substitutions(original, substitution_map)
        rewritten, fix_count = apply_manual_fixes(rewritten, row["Resume_ID"])
        manual_fixes_applied += fix_count
        rewritten = standardize_seniority(rewritten)

        neutral_texts.append(rewritten)
        substitutions_made.append(sub_count)
        residual_agentic.append(count_residual_dictionary_hits(rewritten, agentic_keywords))
        residual_communal.append(count_residual_dictionary_hits(rewritten, communal_keywords))

    selected["Neutral_Resume_Text"] = neutral_texts
    selected["Substitutions_Made"] = substitutions_made
    selected["Residual_Agentic_Hits"] = residual_agentic
    selected["Residual_Communal_Hits"] = residual_communal
    selected["Seniority_Level"] = "Mid"

    selected.to_csv(OUTPUT_PATH, index=False)

    print("Substitutions_Made distribution:")
    print(selected["Substitutions_Made"].describe())
    print(f"\nManual phrase fixes applied: {manual_fixes_applied} (expected 3, across 2 rows)")
    print(f"Saved -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
