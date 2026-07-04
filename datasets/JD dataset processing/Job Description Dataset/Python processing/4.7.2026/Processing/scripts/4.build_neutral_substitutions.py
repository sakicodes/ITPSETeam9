"""
Stage 4a - Author the neutral-substitution mapping.

agentic_expanded.xlsx / communal_expanded.xlsx contain ~600 keyword *inflections*
(e.g. lead/leader/leaders/leading/leadership all listed separately), not a curated
"framing word" list - many entries are actually neutral factual/domain nouns that got
swept in by stem-expansion (e.g. "software", "customer", "responsibilities", "tender").
Blanket-replacing everything by category would delete real resume content, not framing.

This script maps each keyword EXPLICITLY (no regex-group tricks - an earlier attempt at
templated suffix expansion silently produced broken words like "outcomees"/"reliabty")
to either:
  - an actual neutral replacement (genuinely trait/affect-loaded words: aggressive,
    dominant, winner, warm, compassionate, etc.), written out in full per inflected form
  - left unchanged (factual/domain words: software, customer, responsibilities, project,
    degree, tender [a Procurement bid term!], etc.) - the default for anything not listed

Output: neutral_substitutions.csv with one row per dictionary keyword and a
`replacement` column (blank = leave unchanged during Stage 4 rewriting).
"""
import os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
INPUT_DIR = os.path.join(ROOT, "input")
OUTPUT_DIR = os.path.join(ROOT, "output")

AGENTIC_PATH = os.path.join(INPUT_DIR, "agentic_expanded.xlsx")
COMMUNAL_PATH = os.path.join(INPUT_DIR, "communal_expanded.xlsx")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "4.neutral_substitutions.csv")

# Explicit keyword -> (replacement, rationale). Case-insensitive lookup; case of the
# original word is reapplied at substitution time (Stage 4). Anything not listed here
# is left unchanged by default (treated as factual/domain content).
AGENTIC_REPLACEMENTS = {
    # leadership
    "lead": ("coordinate", "solo-command -> team-process language"),
    "leaded": ("coordinated", "solo-command -> team-process language"),
    "leads": ("coordinates", "solo-command -> team-process language"),
    "leader": ("coordinator", "solo-command -> team-process language"),
    "leaders": ("coordinators", "solo-command -> team-process language"),
    "leading": ("coordinating", "solo-command -> team-process language"),
    "leadership": ("coordination", "solo-command -> team-process language"),
    "decisively": ("thoroughly", "trait adverb -> neutral"),
    "decisiveness": ("thoroughness", "trait noun -> neutral"),
    "drive": ("contribute to", "verb; solo-agency -> shared contribution"),
    "entrepreneurial": ("resourceful", "softened trait"),
    "influential": ("effective", "softened trait"),
    "initiative": ("effort", "achievement noun -> plain effort"),
    "initiatives": ("efforts", "achievement noun -> plain effort"),
    "strategic": ("planned", "softened trait"),
    "visionary": ("forward-thinking", "softened trait"),
    # workstyle (style descriptors only; factual nouns like project/task/service untouched)
    "autonomous": ("flexible", "softened trait"),
    "driven": ("motivated", "softened trait"),
    "goal-oriented": ("results-focused", "softened trait"),
    "independent": ("adaptable", "softened trait, standalone adjective"),
    "proactive": ("attentive", "softened trait"),
    "results-driven": ("effective", "softened trait"),
    "self-starter": ("team contributor", "solo-initiative -> team framing"),
    "task-oriented": ("organized", "softened trait"),
    # communication
    "assertive": ("clear and direct", "softened trait"),
    "confident": ("capable", "softened trait"),
    "decisive": ("thorough", "softened trait"),
    "persuasive": ("compelling", "softened trait"),
    # expertise (mostly factual; only the grandiose forms)
    "master": ("skilled", "grandiose -> plain competence"),
    "mastered": ("skilled in", "grandiose -> plain competence"),
    "masterful": ("skilled", "grandiose -> plain competence"),
    "masterfully": ("skillfully", "grandiose -> plain competence"),
    "mastering": ("developing skill in", "grandiose -> plain competence"),
    "mastermind": ("key contributor", "grandiose -> plain competence"),
    "masters": ("skills", "grandiose -> plain competence"),
    "mastery": ("skill", "grandiose -> plain competence"),
    "intellect": ("analytical ability", "softened trait"),
    "intellectual": ("analytical", "softened trait"),
    "intellectually": ("analytically", "softened trait"),
    # personality (core agentic trait bucket)
    "achieve": ("help deliver", "individual-achievement -> contribution framing"),
    "achieved": ("helped deliver", "individual-achievement -> contribution framing"),
    "achievement": ("contribution", "individual-achievement -> contribution framing"),
    "achievements": ("contributions", "individual-achievement -> contribution framing"),
    "achieves": ("helps deliver", "individual-achievement -> contribution framing"),
    "active": ("engaged", "softened trait"),
    "actively": ("consistently", "softened trait"),
    "aggression": ("energy", "softened trait"),
    "aggressive": ("energetic", "softened trait"),
    "aggressively": ("energetically", "softened trait"),
    "ambition": ("motivation", "softened trait"),
    "ambitions": ("goals", "softened trait"),
    "ambitious": ("motivated", "softened trait"),
    "asserting": ("communicating clearly", "softened trait"),
    "assertiveness": ("clear communication", "softened trait"),
    "asserts": ("communicates clearly", "softened trait"),
    # NOTE: deliberately NOT mapped to "motivated" - "driven" already uses that word,
    # and the two frequently co-occur ("driven, competitive"), which produced an
    # awkward "motivated, motivated" repeat during verification.
    "competitive": ("results-focused", "softened trait"),
    "competitively": ("with a results focus", "softened trait"),
    "competitiveness": ("focus on results", "softened trait"),
    "compete": ("strive", "softened trait"),
    "competes": ("strives", "softened trait"),
    "competing": ("striving", "softened trait"),
    "competition": ("field", "softened trait"),
    "competitions": ("fields", "softened trait"),
    "competitor": ("peer", "softened trait"),
    "competitors": ("peers", "softened trait"),
    "courage": ("steadiness", "softened trait"),
    "courageous": ("steady", "softened trait"),
    "courageously": ("steadily", "softened trait"),
    "determination": ("dedication", "trait noun -> neutral"),
    "determinations": ("dedication", "trait noun -> neutral"),
    "determined": ("dedicated", "trait adjective -> neutral"),
    "dominance": ("significant role", "softened trait"),
    "dominant": ("significant", "softened trait"),
    "dominate": ("play a significant role in", "softened trait"),
    "dominated": ("played a significant role in", "softened trait"),
    "forceful": ("effective", "softened trait"),
    "forcefully": ("effectively", "softened trait"),
    "success": ("positive outcome", "individual-glory -> plain outcome"),
    "successes": ("positive outcomes", "individual-glory -> plain outcome"),
    "successful": ("effective", "softened trait"),
    "successfully": ("effectively", "softened trait"),
    "win": ("achieve a positive result", "victory language -> plain result"),
    "wins": ("achieves positive results", "victory language -> plain result"),
    "winner": ("top performer", "victory language -> plain result"),
    "winners": ("top performers", "victory language -> plain result"),
    "winning": ("achieving positive results", "victory language -> plain result"),
}

COMMUNAL_REPLACEMENTS = {
    # teamwork (mostly factual collaboration verbs; only the affect-loaded one)
    "harmonious": ("effective", "softened affect"),
    # service (mostly factual; only the emphasis phrase)
    "customer-focused": ("results-focused", "softened affect"),
    # quality (core communal trait bucket, mirrors agentic personality)
    "care": ("attention", "softened trait"),
    "cared": ("attended to", "softened trait"),
    "carefree": ("relaxed", "softened trait"),
    "careful": ("thorough", "softened trait"),
    "carefully": ("thoroughly", "softened trait"),
    "caring": ("attentive", "softened trait"),
    "cares": ("attends to", "softened trait"),
    "cheer": ("encourage", "softened trait"),
    "cheerful": ("positive", "softened trait"),
    "cheering": ("encouraging", "softened trait"),
    "cheerleading": ("encouragement", "softened trait"),
    "compassion": ("consideration", "softened trait"),
    "compassionate": ("conscientious", "softened trait"),
    "dependability": ("reliability", "near-neutral synonym"),
    "dependable": ("reliable", "near-neutral synonym"),
    "emotional": ("measured", "softened trait"),
    "emotionally": ("in a measured way", "softened trait"),
    "empathetic": ("considerate", "softened trait"),
    "empathetically": ("considerately", "softened trait"),
    "empathic": ("considerate", "softened trait"),
    "empathize": ("understand others' views", "softened trait"),
    "empathizes": ("understands others' views", "softened trait"),
    "empathizing": ("understanding others' views", "softened trait"),
    "empathy": ("consideration", "softened trait"),
    "gentle": ("measured", "softened trait"),
    "love": ("value", "affection language -> plain valuation"),
    "loved": ("valued", "affection language -> plain valuation"),
    "lovely": ("valuable", "affection language -> plain valuation"),
    "loves": ("values", "affection language -> plain valuation"),
    "loyal": ("reliable", "softened trait"),
    "loyalty": ("reliability", "softened trait"),
    "modest": ("measured", "softened trait"),
    "nurture": ("support", "softened trait"),
    "nurtured": ("supported", "softened trait"),
    "nurtures": ("supports", "softened trait"),
    "nurturing": ("supporting", "softened trait"),
    "pleasant": ("professional", "softened trait"),
    "pleasantly": ("professionally", "softened trait"),
    "quiet": ("measured", "softened trait"),
    "quieter": ("more measured", "softened trait"),
    "quietly": ("in a measured way", "softened trait"),
    "respectful": ("professional", "softened trait"),
    "sensitive": ("attentive", "softened trait"),
    "sensitivities": ("considerations", "softened trait"),
    "sensitivity": ("attentiveness", "softened trait"),
    "sincere": ("professional", "softened trait"),
    "sincerely": ("professionally", "softened trait"),
    "soothing": ("measured", "softened trait"),
    "sympathetic": ("considerate", "softened trait"),
    "trust-building": ("credibility-building", "softened trait"),
    "trusted": ("reliable", "softened trait"),
    "trustworthiness": ("reliability", "softened trait"),
    "trustworthy": ("reliable", "softened trait"),
    "warm": ("professional", "softened trait"),
    "warmer": ("more professional", "softened trait"),
    "warming": ("becoming professional", "softened trait"),
    "warmly": ("professionally", "softened trait"),
    "warmth": ("professionalism", "softened trait"),
}


def build_table(path, replacements, dictionary_name):
    df = pd.read_excel(path)
    rows = []
    for _, row in df.iterrows():
        keyword, category = str(row["keyword"]), row["category"]
        entry = replacements.get(keyword.lower())
        replacement, rationale = entry if entry else ("", "no rule matched - left unchanged by default")
        rows.append({
            "keyword": keyword,
            "category": category,
            "dictionary": dictionary_name,
            "replacement": replacement,
            "rationale": rationale,
        })
    return pd.DataFrame(rows)


def main():
    agentic_table = build_table(AGENTIC_PATH, AGENTIC_REPLACEMENTS, "Agentic")
    communal_table = build_table(COMMUNAL_PATH, COMMUNAL_REPLACEMENTS, "Communal")

    final = pd.concat([agentic_table, communal_table], ignore_index=True)
    final.to_csv(OUTPUT_PATH, index=False)

    active = final[final["replacement"] != ""]
    print(f"Total dictionary keywords: {len(final)}")
    print(f"Keywords with an active substitution: {len(active)}")
    print(f"Keywords left unchanged (factual/domain/ambiguous): {len(final) - len(active)}")
    print(f"\nSaved -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
