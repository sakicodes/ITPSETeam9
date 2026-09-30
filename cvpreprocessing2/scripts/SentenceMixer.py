import pandas as pd
import re
import json
from rapidfuzz import fuzz
import numpy as np


# ============================================================
# SETTINGS
# ============================================================

INPUT_FILE = "Agentic_Communal_Pilot_Cover_Letter_V2.xlsx"
OUTPUT_FILE = "AC_similarity_output.xlsx"

SHEET_NAME = "AC Initial"

SIMILARITY_THRESHOLD = 95

GLOBAL_SEED = 20240101


# ============================================================
# LOAD EXCEL
# ============================================================

target_df = pd.read_excel(
    INPUT_FILE,
    sheet_name=SHEET_NAME
)


# ============================================================
# SPLIT INTO SENTENCES
# ============================================================

def split_sentences(text):

    if pd.isna(text):
        return []

    text = str(text).strip()

    if not text:
        return []

    paragraphs = re.split(
        r'\n\s*\n+',
        text
    )

    sentences = []

    for paragraph in paragraphs:

        paragraph = paragraph.strip()

        if not paragraph:
            continue

        parts = re.split(
            r'(?<=[.!?])\s+',
            paragraph
        )

        for sentence in parts:

            sentence = sentence.strip()

            if sentence:
                sentences.append(sentence)

    return sentences


# ============================================================
# SPLIT INTO PARAGRAPHS + SENTENCES
# Used when rebuilding the mixed letter
# ============================================================

def split_paragraphs_and_sentences(text):

    if pd.isna(text):
        return []

    text = str(text).strip()

    if not text:
        return []

    parts = re.split(
        r'(\n\s*\n+)',
        text
    )

    result = []

    for part in parts:

        # Preserve blank-line separators
        if re.fullmatch(
            r'\n\s*\n+',
            part
        ):
            result.append({
                "type": "separator",
                "text": part
            })

            continue

        paragraph = part.strip()

        if not paragraph:
            continue

        sentences = re.split(
            r'(?<=[.!?])\s+',
            paragraph
        )

        result.append({
            "type": "paragraph",
            "sentences": [
                sentence.strip()
                for sentence in sentences
                if sentence.strip()
            ]
        })

    return result


# ============================================================
# COMPARE TWO LETTERS
# ============================================================

def compare_letters(base_text, target_text):

    base_sents = split_sentences(base_text)
    target_sents = split_sentences(target_text)

    scores = []
    min_len = min(len(base_sents), len(target_sents))

    # --------------------------------------------------------
    # START FROM SENTENCE 2
    #
    # Sentence 1 = "Dear Hiring Manager"
    # It remains part of the letter but is never mixed.
    # --------------------------------------------------------

    for i in range(1, min_len):

        score = fuzz.ratio(base_sents[i], target_sents[i])

        base_word_count = len(base_sents[i].split())
        target_word_count = len(target_sents[i].split())

        if score < SIMILARITY_THRESHOLD and base_word_count >= 4 and target_word_count >= 4:

            scores.append({
                "sentence_index": i + 1,
                "base_sentence": base_sents[i],
                "target_sentence": target_sents[i],
                "similarity_score": round(score, 2)
            })

    return scores


# ============================================================
# CREATE MIXED LETTER
# ============================================================

def create_mixed_letter(
    agentic_text,
    communal_text,
    eligible_indices,
    arrangement
):

    agentic_structure = split_paragraphs_and_sentences(
        agentic_text
    )

    communal_structure = split_paragraphs_and_sentences(
        communal_text
    )

    mixed_output = []

    sentence_number = 0

    # --------------------------------------------------------
    # Go through every paragraph
    # --------------------------------------------------------

    for agentic_part, communal_part in zip(
        agentic_structure,
        communal_structure
    ):

        # ----------------------------------------------------
        # Preserve paragraph separators
        # ----------------------------------------------------

        if agentic_part["type"] == "separator":

            mixed_output.append(
                agentic_part["text"]
            )

            continue

        mixed_sentences = []

        # ----------------------------------------------------
        # Process each sentence in this paragraph
        # ----------------------------------------------------

        for i, agentic_sentence in enumerate(
            agentic_part["sentences"]
        ):

            sentence_number += 1

            communal_sentence = communal_part["sentences"][i]

            # ------------------------------------------------
            # Sentence 1:
            # Always Agentic
            # ------------------------------------------------

            if sentence_number == 1:

                mixed_sentences.append(
                    agentic_sentence
                )

                continue

            # ------------------------------------------------
            # Eligible sentence:
            # use A/C arrangement
            # ------------------------------------------------

            if sentence_number in eligible_indices:

                eligible_position = (
                    eligible_indices.index(
                        sentence_number
                    )
                )

                source = arrangement[
                    eligible_position
                ]

                if source == "A":

                    mixed_sentences.append(
                        agentic_sentence
                    )

                else:

                    mixed_sentences.append(
                        communal_sentence
                    )

            # ------------------------------------------------
            # Non-eligible sentence:
            # keep Agentic
            # ------------------------------------------------

            else:

                mixed_sentences.append(
                    agentic_sentence
                )

        # ----------------------------------------------------
        # Rebuild paragraph
        # ----------------------------------------------------

        mixed_output.append(
            " ".join(mixed_sentences)
        )

    # --------------------------------------------------------
    # Recombine everything
    # --------------------------------------------------------

    return "".join(mixed_output)


# ============================================================
# SEEDED RANDOM NUMBER GENERATOR
# ============================================================

rng = np.random.default_rng(GLOBAL_SEED)


# ============================================================
# PROCESS EACH ROW
# ============================================================

results = []

eligible_indices = []
total_eligible_sentences = []
arrangements = []
mixed_letters = []


for _, row in target_df.iterrows():

    agentic_text = row["Agentic_Cover_Letter"]
    communal_text = row["Communal_Cover_Letter"]


    # ========================================================
    # SPLIT LETTERS
    # ========================================================

    agentic_sentences = split_sentences(
        agentic_text
    )

    communal_sentences = split_sentences(
        communal_text
    )


    # ========================================================
    # COMPARE SENTENCES
    # ========================================================

    comparison = compare_letters(
        agentic_text,
        communal_text
    )

    results.append(
        comparison
    )


    # ========================================================
    # GET ELIGIBLE SENTENCE NUMBERS
    # ========================================================

    indices = [
        item["sentence_index"]
        for item in comparison
    ]

    eligible_indices.append(
        indices
    )


    # ========================================================
    # COUNT ELIGIBLE SENTENCES
    # ========================================================

    total_sentences = len(indices)

    total_eligible_sentences.append(
        total_sentences
    )


    # ========================================================
    # GENERATE A/C ARRANGEMENT
    # ========================================================

    if total_sentences == 0:

        arrangement = ""

    else:

        # ----------------------------------------------------
        # EVEN NUMBER
        # ----------------------------------------------------

        if total_sentences % 2 == 0:

            half = total_sentences // 2

            arrangement_list = (
                ['A'] * half +
                ['C'] * half
            )

        # ----------------------------------------------------
        # ODD NUMBER
        #
        # Create balanced A/C pool with one extra slot,
        # shuffle it, then remove one random slot.
        # ----------------------------------------------------

        else:

            half = (total_sentences + 1) // 2

            arrangement_list = (
                ['A'] * half +
                ['C'] * half
            )

        # Shuffle
        rng.shuffle(
            arrangement_list
        )

        # Remove one slot for odd number
        if total_sentences % 2 == 1:

            arrangement_list.pop()

        arrangement = "".join(
            arrangement_list
        )


    arrangements.append(
        arrangement
    )


    # ========================================================
    # CREATE MIXED LETTER
    # ========================================================

    mixed_letter = create_mixed_letter(
        agentic_text,
        communal_text,
        indices,
        arrangement
    )

    mixed_letters.append(
        mixed_letter
    )


# ============================================================
# ADD OUTPUT COLUMNS
# ============================================================

target_df["Similarity"] = [
    json.dumps(
        result,
        ensure_ascii=False
    )
    for result in results
]


target_df["Eligible_Sentence_ID"] = [
    ", ".join(
        map(str, indices)
    )
    for indices in eligible_indices
]


target_df["Total_Eligible_Sentences"] = (
    total_eligible_sentences
)


target_df["Arrangements"] = (
    arrangements
)


target_df["Mixed_Cover_Letter"] = (
    mixed_letters
)


# ============================================================
# SAVE OUTPUT
# ============================================================

with pd.ExcelWriter(
    OUTPUT_FILE,
    engine="openpyxl"
) as writer:

    target_df.to_excel(
        writer,
        index=False,
        sheet_name="AC Sentences"
    )


print(
    f"Done! Results saved to {OUTPUT_FILE}"
)