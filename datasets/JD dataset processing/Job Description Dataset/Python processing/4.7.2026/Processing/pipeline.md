# Neutral-Framing CV Pipeline

## Purpose

Builds a controlled resume pool for the ITP research project auditing whether LLMs
favour "agentic" competence framing (assertive, independent, individual achievement)
over "communal/relational" framing (collaborative, team-oriented) when evaluating job
candidates. The pool needs framing and seniority removed as confounds, so any scoring
difference an LLM produces later can be attributed to framing that is deliberately
reintroduced afterward, not to noise already present in the source resumes.

**Deliverable**: `output/6.neutral_cv_pool_90_FINAL.csv` - 90 real resumes, 30 each for
Procurement/Sourcing/Supply Chain, Sales/Business Development, and Project/Programme
Management, rewritten to neutral framing and standardized to Mid-level seniority.

This pipeline went through a full readiness audit and a remediation pass after the
first version was built - see **Audit & remediation history** at the bottom for what
was found and fixed. The description below reflects the current, remediated pipeline.

## Folder layout

```
Processing/
├── scripts/    the 6 pipeline stages, run in numeric order
├── input/      source data (CV pools + dictionaries), read-only to the pipeline
├── output/     everything the pipeline generates, also numbered by stage
└── pipeline.md this file
```

## Inputs (`input/`)

| File | What it is |
|---|---|
| `Resume_cleaned.csv` | 2,472 resumes (`ID, Resume_str, Resume_html, Category`), 24 generic job categories - no native Procurement or Project Management category |
| `agentic_expanded.xlsx` | 350 keywords across 5 sub-categories: `leadership, workstyle, personality, expertise, communication` |
| `communal_expanded.xlsx` | 252 keywords across 3 sub-categories: `teamwork, service, quality` |
| `filtered_phrases.csv` | **No longer used.** Originally used as a QA phrase bank in Stage 5 (see audit history) - dropped because 82.8% of it is labeled "Agentic," making an exact-text match against it mostly a false-positive generator rather than a real bias signal. |

A second CV pool, HuggingFace `syedroshanzameer/resume-classification`, is pulled live
by Stage 1 and cached into `output/` (not stored in `input/` since it's fetched, not
hand-provided). It turned out to be ~99% the same underlying resumes as
`Resume_cleaned.csv` (2,488 unique after combining and deduping).

`base_cvs_90_scrubbed_mid_final.csv` (in the parent folder) is explicitly excluded from
this pipeline - a prior, separate effort.

## Pipeline stages

### 1. `1.fetch_cv_pools.py` - Fetch & cache CV pools
Loads `Resume_cleaned.csv` and pulls the HF dataset (cached locally after first run so
the pipeline works offline afterward). Normalizes both pools to a common shape:
`Source_Pool, Resume_ID, Resume_Text, Native_Category`, concatenates, and drops exact
text duplicates between the two pools.
- Outputs: `1.hf_resumes_raw_cache.csv` (raw HF pull cache), `1.cv_pool_combined.csv`
  (2,488 rows)

### 2. `2.classify_industry.py` - Industry classification
Sales/Business Development is pulled directly from `Native_Category` (no native
category exists for the other two, but this one already exists as `SALES` /
`BUSINESS-DEVELOPMENT`). Procurement and Project Management have no native category in
either pool, so they're identified with a weighted keyword classifier:
- Isolates each resume's "title zone" (text before the first section header like
  `Summary`/`Skills`/`Profile`) - resumes consistently open with an ALL-CAPS job title
  line there.
- Keyword hits in the title zone are weighted far more than hits in the body (10x),
  since a passing mention of "vendor sourcing" in a Designer's skills list shouldn't
  outrank an actual "PURCHASING AGENT" title.
- Requires either a title-zone hit or 4+ body-only hits to qualify at all, filtering out
  resumes that merely mention a keyword in passing.
- The Procurement keyword list was widened post-audit (added `inventory control`,
  `materials manager`, `logistics coordinator`, `warehouse operations`, `RFP/RFQ`, etc.)
  after the audit found the original list left genuine title-confirmed candidates on
  the table - this only raised the count from ~8 to ~10, confirming the corpus itself
  is thin on this category rather than the keyword list being the bottleneck.
- `Classifier_Confidence` (0.32-1.0) reflects how strong the signal was: `>=0.85` means
  a genuine title-zone match, `<0.85` means body-only inference.
- Outputs: `2.cv_pool_classified.csv` (~589 candidates: 234 Sales, 194 Procurement,
  161 Project Mgmt pre-selection)

### 3. `3.select_candidates.py` - Seniority detection & candidate selection
Detects a rough seniority band (`Senior` / `Mid` / `Junior`) per resume from explicit
"X years" mentions and title-line markers (Senior/Director/VP/Chief/CEO/CFO/COO/CTO/
President vs Intern/Entry-level/Junior/Trainee). Selection strategy per industry:
- **Sales & Project Management**: prefer candidates that are both title-confirmed
  (`Classifier_Confidence >= 0.85`) and Mid-band, random-sampled with a fixed seed.
  Both industries have enough of these to hit 30 without falling back (PM: 30 of 49
  title-confirmed candidates are genuinely Mid-band).
- **Procurement**: the corpus only contains ~10 genuinely title-confirmed resumes - a
  real corpus limitation, confirmed by manually reading the title lines of the next
  ~180 lower-confidence candidates (mostly Chefs/Accountants/IT Directors who just
  mention "purchase order" or "inventory" in passing). The remaining 20 slots come from
  a manually-vetted allowlist of adjacent-function titles (Buyer, Storekeeper, Materials
  Analyst, Warehouse Lead, Fulfillment Advocate, Supply Sergeant, etc.), hardcoded in
  the script with the rationale for each. Every row's `Selection_Basis` records which
  tier it came from, so this is transparent rather than silently blended in.

A now-fixed bug is worth noting for anyone touching this file again: the original
`SENIOR_MARKERS` regex included a bare `\bmanager\b`, which misclassified nearly every
genuine "Project Manager"/"Program Manager" resume as Senior band, since "manager" is
just the most common word in mid-level titles, not a seniority signal on its own.
Removing it fixed PM's Mid-band count from a handful up to 30.
- Outputs: `3.cv_pool_selected_90.csv` (90 rows, 30/30/30, asserted)

### 4. `4.build_neutral_substitutions.py` - Author the neutral-substitution mapping
The two dictionaries list ~600 keyword *inflections* (lead/leader/leaders/leading/...),
not a curated "framing word" list - many entries are neutral factual/domain nouns swept
in by stem-expansion (`software`, `customer`, `responsibilities`, even `tender`, which is
a legitimate Procurement bid term). Blanket-replacing by category would delete real
resume content, not framing.

This script maps each of the 602 keywords **explicitly** (no regex-template tricks - an
earlier attempt at suffix-expansion silently produced broken words like "outcomees" and
"reliabty", caught during verification) to either a genuine neutral replacement (145
keywords - e.g. `aggressive`→`energetic`, `winner`→`top performer`,
`compassionate`→`conscientious`) or left unchanged (457 keywords - factual/domain
content, or too ambiguous to safely rewrite, e.g. `independently`, `responsibilities`).
- Outputs: `4.neutral_substitutions.csv` (602 rows: `keyword, category, dictionary,
  replacement, rationale` - blank `replacement` = left unchanged)

### 5. `5.neutralize_text.py` - Neutral-framing rewrite + seniority standardization
- **Rewrite**: applies the substitution map word-boundary-safe, longest keyword first,
  preserving the original word's capitalization (so an all-caps title line stays
  all-caps). Counts replacements made (`Substitutions_Made`).
- **Manual phrase fixes**: 2 specific rows (3 phrases) identified during the audit as
  carrying genuine solo/individual-glory framing that word-level substitution can't
  catch on its own (e.g. "I single-handedly manage...") are fixed with an explicit,
  hardcoded before/after replacement keyed by `Resume_ID`, applied right after the
  dictionary pass. See audit history below for why this replaced a broader automated
  QA-flagging step that turned out to be unreliable.
- **Seniority standardization**: normalizes every explicit "X years" mention to a fixed
  `5-7 years` phrase, and strips/replaces unambiguous seniority-coded title words
  (`Senior`/`Sr.`/`Director`/`VP`/`Chief`/`Principal` → removed or → `manager`;
  `Junior`/`Jr.`/`Intern`/`Trainee` → removed or → `associate`). Deliberately does *not*
  touch `Manager` or `Lead` on their own - those are legitimate neutral mid-level titles.
  Sets `Seniority_Level = "Mid"` uniformly.
- Outputs: `5.cv_pool_neutralized_90_INTERMEDIATE.csv` - all pipeline-internal columns
  still present (`Resolved_Category`, `Detected_Seniority_Raw`, `Detected_Years`,
  `Resume_Text`). **Not the deliverable** - Stage 6 trims this down.

### 6. `6.build_output.py` - Assemble the final output CSV
Drops internal scaffolding columns, renames `Resume_Text` → `Original_Resume_Text`,
assigns `CV_ID` (`PROC-001`..`030`, `SALES-001`..`030`, `PM-001`..`030`), computes
`Word_Count` on the final neutralized text, and asserts exactly 30 rows per industry
(90 total) before writing.
- Outputs: `6.neutral_cv_pool_90_FINAL.csv` - **the deliverable**.

## Final schema (`6.neutral_cv_pool_90_FINAL.csv`)

| Column | Meaning |
|---|---|
| `CV_ID` | Industry-prefixed row ID, e.g. `PROC-014` |
| `Resume_ID` | Source ID within its pool |
| `Source_Pool` | `Resume_cleaned` or `HF_resume_classification` |
| `Industry` | One of the 3 target industries |
| `Native_Category` | The resume's original category in its source pool (e.g. `CONSTRUCTION`) - shows how far from a native match the classifier reached |
| `Seniority_Level` | Always `Mid` (standardized) |
| `Word_Count` | Word count of the neutralized text |
| `Classifier_Confidence` | Stage 2's confidence in the industry assignment (`>=0.85` = genuine title match, `1.0` = direct Sales/BusDev category pull) |
| `Selection_Basis` | How the row entered the pool: `Native category (direct)`, `Title-confirmed (>=0.85)`, or `Manually-vetted adjacent function` (Procurement only) |
| `Substitutions_Made` | How many word-level replacements Stage 5 applied |
| `Residual_Agentic_Hits` / `Residual_Communal_Hits` | Count of full-dictionary keyword matches remaining (diagnostic only - many are intentionally-skipped factual words, not a defect) |
| `Original_Resume_Text` | Untouched source text |
| `Neutral_Resume_Text` | Rewritten, standardized text |

## Running the pipeline end-to-end

```
cd Processing/scripts
python "1.fetch_cv_pools.py"
python "2.classify_industry.py"
python "3.select_candidates.py"
python "4.build_neutral_substitutions.py"
python "5.neutralize_text.py"
python "6.build_output.py"
```

Deterministic given the same inputs (fixed random seed in Stage 3, hardcoded manual
picks/fixes in Stages 3 and 5) - re-running reproduces the same 90 rows and same
`Substitutions_Made` distribution every time.

## Known limitations (current, post-remediation)

- **Procurement is 10 title-confirmed + 20 manually-vetted adjacent-function resumes**,
  not 30 genuine title matches - the corpus (a general-purpose 24-category resume
  dataset) just doesn't contain 30 dedicated Procurement professionals. `Selection_Basis`
  makes this visible per-row rather than hiding it.
- **Un-redacted PII was found during the audit and has not yet been scrubbed**: a small
  number of rows (found: 2 candidate-identifying LinkedIn URLs, 1 third-party name/
  phone/address, 1 phone number) still contain real contact information, which can leak
  identity/gender signal into what's meant to be a demographically-neutral pool. This is
  the highest-priority item still open - see audit history below.
- **Some near-synonym collisions remain possible**: several different original words map
  to the same neutral replacement (e.g. 5 different words all map to `effective`), which
  can occasionally read redundantly if two of them happen to sit close together in the
  same resume. One confirmed instance (`driven, competitive` → `motivated, motivated`)
  was fixed by diversifying that specific pair; others of the same shape may still exist
  but weren't hit during spot-checking.
- **A handful of executive-suite (CEO/CFO/COO/CTO/President) and Intern/Internship
  mentions can still survive** in body text describing past roles or reporting lines
  (e.g. "reported to the CEO", "Marketing Intern" as a prior job) - Stage 5's title-word
  replacement list targets the person's own current title, not every historical mention.

## Audit & remediation history

A full readiness audit was run on the first version of the final output before it was
used. Findings and what was done about each:

1. **Weak industry classification (23-50/90 rows body-only, no title match)** - Fixed.
   Widened the Procurement keyword list (didn't move the needle much - confirmed a
   corpus limitation, not a keyword gap), fixed the `SENIOR_MARKERS` bug that was
   pushing genuine title-confirmed PM resumes out of the Mid-band selection pool, and
   rebuilt selection to prefer `Classifier_Confidence >= 0.85` everywhere it's available
   (all of Sales and PM; 10/30 of Procurement, with the rest from a manually-vetted
   allowlist - see Stage 3 above).
2. **43/90 rows flagged by a phrase-bank QA check** - Replaced. Manual review of every
   flagged phrase found only 2 rows (3 phrases) with genuine solo/individual-glory
   framing; the other 37 were false positives of an over-broad check (the phrase bank
   is 82.8% labeled "Agentic", so an exact match mostly just meant "this is an ordinary
   resume sentence"). The 2 genuine rows were fixed with explicit before/after phrase
   replacements; the QA-flagging mechanism, its `Needs_Review` column, and
   `filtered_phrases.csv` as a pipeline input were all removed rather than kept as an
   unreliable signal.
3. **PII leakage (4/90 rows)** - **Not yet fixed.** Flagged as the highest-priority
   remaining item; no PII-scrubbing stage exists in this pipeline yet.
4. **Residual agentic framing exceeds communal pool-wide** - Not yet addressed. Still
   visible via `Residual_Agentic_Hits` / `Residual_Communal_Hits` per row.
5. **Sales/Business Development is shorter and was more heavily flagged than the other
   two industries** - The flagging half of this is now moot (flagging mechanism
   removed); the word-count gap (~741 vs ~960-1020 words) has not been addressed.
6. **Seniority-title word list gaps (CEO/CFO/COO/CTO/President, Intern/Internship)** -
   Not yet addressed in Stage 5's active rewriting (only in Stage 3's *detection* logic,
   which was fixed as part of item 1).
7. **`reliable`-collision cluster (5 words → 1 replacement)** - Not yet addressed.
