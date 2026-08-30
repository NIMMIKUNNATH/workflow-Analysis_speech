#!/usr/bin/env python3
"""
clinical_errors.py - classify ASR errors by clinical significance, not by WER.

WHY THIS EXISTS
    WER treats every word equally. For OSCE assessment they are not equal:

        "body movement"  -> "bowel movement"     one word, different symptom
        "no chest pain"  -> "chest pain"         one word, opposite meaning
        "150 mg"         -> "50 mg"              one word, wrong dose
        "um"             -> (dropped)            one word, no consequence

    A medical education reviewer will not ask about beam size. They will ask
    whether the transcript can support scoring a student. This measures that.

WHAT IT REPORTS
    - Clinical term error rate, by category (medication, symptom, anatomy,
      temporal, number/dose)
    - CRITICAL errors: negation flips and dose/number changes, listed verbatim
    - Filler-only errors, reported separately so they do not inflate the
      clinical picture
    - Per-recording and aggregate, with examples for qualitative reporting

METHOD
    1. Align reference and hypothesis word sequences (same alignment as WER)
    2. For each reference word, decide which clinical class it belongs to
    3. Ask whether the aligner matched it correctly
    4. Error rate per class = wrong / total occurrences of that class

    Clinical terms are identified two ways: a curated seed lexicon, and terms
    mined from the reference corpus itself. The second matters because no seed
    list anticipates every drug or symptom in 272 consultations.

USAGE
    python clinical_errors.py \\
        --result-dir ${ASR_CACHE_ROOT} \\
        --ref-dir "${ASR_DATA_ROOT}/Clean Transcripts" \\
        --csv clinical_errors.csv --examples clinical_examples.txt
"""
import argparse
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

try:
    import jiwer
except ImportError:
    sys.exit("pip install jiwer pandas openpyxl")


# ============================================================== lexicons ====
# Seed lists. Deliberately conservative - a false positive here inflates the
# denominator and makes the error rate look better than it is.

MEDICATIONS = {
    "aspirin", "tylenol", "acetaminophen", "advil", "ibuprofen", "paracetamol",
    "lisinopril", "ramipril", "enalapril", "losartan", "amlodipine",
    "atorvastatin", "rosuvastatin", "simvastatin", "metformin", "insulin",
    "salbutamol", "albuterol", "ventolin", "fluticasone", "prednisone",
    "prednisolone", "furosemide", "lasix", "warfarin", "heparin", "clopidogrel",
    "omeprazole", "pantoprazole", "ranitidine", "amoxicillin", "azithromycin",
    "ciprofloxacin", "doxycycline", "penicillin", "morphine", "codeine",
    "naproxen", "diclofenac", "gabapentin", "amitriptyline", "sertraline",
    "citalopram", "diazepam", "lorazepam", "levothyroxine", "nitroglycerin",
    "nitro", "puffer", "inhaler", "antibiotic", "antibiotics", "statin",
    "diuretic", "steroid", "steroids", "antihistamine", "benadryl", "reactine",
}

SYMPTOMS = {
    "pain", "cough", "fever", "chills", "nausea", "vomiting", "vomited",
    "diarrhea", "diarrhoea", "constipation", "bloating", "cramping",
    "shortness", "breath", "breathless", "wheeze", "wheezing", "wheezy",
    "palpitations", "dizziness", "dizzy", "faint", "fainting", "syncope",
    "headache", "migraine", "rash", "itching", "itchy", "swelling", "swollen",
    "numbness", "tingling", "weakness", "fatigue", "tired", "tiredness",
    "sweating", "sweats", "night", "weight", "appetite", "jaundice",
    "bleeding", "bruising", "discharge", "burning", "throbbing", "sharp",
    "dull", "aching", "stabbing", "radiating", "cramps", "heartburn",
    "reflux", "indigestion", "hoarse", "hoarseness", "sputum", "phlegm",
    "mucus", "congestion", "stuffy", "runny", "sneezing", "sore",
}

ANATOMY = {
    "chest", "abdomen", "abdominal", "stomach", "bowel", "bowels", "back",
    "head", "neck", "throat", "shoulder", "arm", "elbow", "wrist", "hand",
    "hip", "knee", "ankle", "foot", "leg", "thigh", "calf", "heart", "lung",
    "lungs", "liver", "kidney", "kidneys", "bladder", "skin", "joint",
    "joints", "muscle", "muscles", "spine", "rib", "ribs", "jaw", "eye",
    "eyes", "ear", "ears", "nose", "sinus", "sinuses", "gallbladder",
    "pancreas", "thyroid", "prostate", "uterus", "ovary", "ovaries",
    "epigastric", "quadrant", "flank", "groin", "pelvis", "pelvic",
}

CONDITIONS = {
    "asthma", "copd", "pneumonia", "bronchitis", "emphysema", "tuberculosis",
    "diabetes", "diabetic", "hypertension", "hypertensive", "cholesterol",
    "angina", "infarction", "stroke", "arrhythmia", "fibrillation", "failure",
    "arthritis", "osteoporosis", "fracture", "sprain", "strain", "tendonitis",
    "eczema", "psoriasis", "dermatitis", "cellulitis", "ulcer", "gastritis",
    "colitis", "crohn", "celiac", "reflux", "gerd", "ibs", "anemia", "anaemia",
    "cancer", "tumour", "tumor", "infection", "inflammation", "allergy",
    "allergies", "allergic", "anaphylaxis", "embolism", "thrombosis", "dvt",
    "hypothyroidism", "hyperthyroidism", "migraine", "epilepsy", "seizure",
}

TEMPORAL = {
    "yesterday", "today", "tonight", "week", "weeks", "month", "months",
    "year", "years", "day", "days", "hour", "hours", "minute", "minutes",
    "morning", "afternoon", "evening", "night", "chronic", "acute", "sudden",
    "gradual", "gradually", "constant", "intermittent", "occasional",
}

NEGATION = {
    "no", "not", "never", "none", "nothing", "without", "denies", "denied",
    "negative", "nope", "haven't", "hasn't", "didn't", "don't", "doesn't",
    "wasn't", "weren't", "isn't", "aren't", "can't", "couldn't", "won't",
}

FILLERS = {
    "uh", "um", "umm", "uhh", "mm", "hmm", "mhm", "ah", "oh", "er", "erm",
    "like", "you know", "i mean", "sort", "kind", "well", "so", "okay", "ok",
    "yeah", "yep", "right", "actually", "basically", "just",
}

NUMBER_WORDS = {
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty",
    "ninety", "hundred", "thousand", "half", "quarter", "once", "twice",
}

UNITS = {"mg", "milligram", "milligrams", "ml", "millilitre", "millilitres",
         "gram", "grams", "mcg", "microgram", "micrograms", "unit", "units",
         "puff", "puffs", "tablet", "tablets", "pill", "pills", "dose", "doses"}

CLASSES = [
    ("medication", MEDICATIONS),
    ("condition", CONDITIONS),
    ("symptom", SYMPTOMS),
    ("anatomy", ANATOMY),
    ("temporal", TEMPORAL),
    ("negation", NEGATION),
    ("unit", UNITS),
]


# ------------------------------------------------- text normalisation ------
# Digit/word form is a TRANSCRIPTION CONVENTION, not an error. The Fareez
# references write "8" where Whisper writes "eight", and vice versa. Counting
# those as clinical errors produced a spurious 30% number error rate on the first
# run, which buried the genuine dose errors underneath formatting noise.
#
# Whisper ships an EnglishTextNormalizer for exactly this reason, and the ASR
# literature reports that normalisation shifts WER by 1-5 absolute points. The
# same logic applies, more sharply, to a clinical metric: a reviewer will not
# accept "eight" for "8" as a dose error.
NUM_MAP = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
    "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
    "eighteen": "18", "nineteen": "19", "twenty": "20", "thirty": "30",
    "forty": "40", "fifty": "50", "sixty": "60", "seventy": "70",
    "eighty": "80", "ninety": "90", "hundred": "100", "thousand": "1000",
}

# Negation carries the same meaning across these surface forms. "nope" for "no"
# reverses nothing; "mm" or "okay" for "no" reverses everything. Only the second
# kind is a clinical error, and conflating them makes the metric useless.
NEG_EQUIV = {
    "no": "no", "nope": "no", "nah": "no", "not": "not", "never": "never",
    "none": "none", "nothing": "nothing", "negative": "no",
    "dont": "not", "don't": "not", "doesnt": "not", "doesn't": "not",
    "didnt": "not", "didn't": "not", "havent": "not", "haven't": "not",
    "hasnt": "not", "hasn't": "not", "wasnt": "not", "wasn't": "not",
    "isnt": "not", "isn't": "not", "arent": "not", "aren't": "not",
    "cant": "not", "can't": "not", "couldnt": "not", "couldn't": "not",
    "wont": "not", "won't": "not", "wouldnt": "not", "wouldn't": "not",
}


def equivalent(ref_word, hyp_word):
    """
    Are these two words the same thing written differently?

    Returns True for digit/word pairs and for negation variants that preserve
    polarity. Everything else is a genuine substitution.
    """
    if ref_word == hyp_word:
        return True
    if NUM_MAP.get(ref_word, ref_word) == NUM_MAP.get(hyp_word, hyp_word):
        return True
    if ref_word in NEG_EQUIV and hyp_word in NEG_EQUIV:
        return NEG_EQUIV[ref_word] == NEG_EQUIV[hyp_word]
    return False


def classify(word):
    """
    Which clinical class does this reference word belong to?

    Order matters: a word appearing in two lists takes the first match, and the
    lists are ordered by clinical consequence. "failure" is a condition (heart
    failure) before anything else.
    """
    if re.fullmatch(r"\d+", word) or word in NUMBER_WORDS:
        return "number"
    for name, lex in CLASSES:
        if word in lex:
            return name
    if word in FILLERS:
        return "filler"
    return None


def mine_corpus_terms(all_ref_words, min_count=5):
    """
    Terms the seed lexicon missed.

    No hand-written list anticipates every drug and symptom across 272
    consultations. This surfaces frequent words that are not already classified
    and not common English, so they can be reviewed and added. It does NOT
    auto-classify them - that would be guessing, and a wrong guess corrupts the
    denominator.
    """
    common = {
        "the", "and", "you", "that", "have", "for", "with", "was", "are", "but",
        "not", "this", "your", "any", "been", "had", "has", "would", "could",
        "about", "when", "what", "there", "they", "them", "then", "than",
        "were", "will", "just", "can", "get", "got", "going", "know", "think",
        "yes", "its", "it's", "i'm", "i've", "that's", "there's", "let's",
        "okay", "yeah", "right", "sure", "very", "some", "all", "one", "out",
        "how", "now", "did", "does", "from", "like", "also", "more", "other",
        "start", "started", "feel", "feeling", "felt", "take", "taking", "took",
    }
    counts = Counter(w for w in all_ref_words
                     if len(w) > 4 and classify(w) is None and w not in common)
    return [(w, c) for w, c in counts.most_common(40) if c >= min_count]


# ================================================================ parsing ===
def norm(s):
    s = re.sub(r"[^a-z0-9' ]", " ", str(s).lower())
    return re.sub(r"\s+", " ", s).strip()


def parse_reference(path):
    words, current = [], None
    for raw in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line:
            continue
        m = re.match(r"^\s*([DP])\s*:\s*(.*)$", line, flags=re.I)
        if m:
            current, text = m.group(1).upper(), m.group(2)
        else:
            if current is None:
                continue
            text = line
        words += [(w, current) for w in norm(text).split()]
    return words


def parse_result(path):
    df = pd.read_excel(path)
    col = next((c for c in df.columns if "transcript" in c.lower()
                and "researcher" not in c.lower()), None)
    if col is None:
        return []
    out = []
    for _, r in df.iterrows():
        if pd.isna(r.get(col)):
            continue
        out += norm(r[col]).split()
    return out


# =============================================================== analysis ===
def analyse(ref_pairs, hyp_words, examples):
    ref_words = [w for w, _ in ref_pairs]
    if not ref_words or not hyp_words:
        return None

    out = jiwer.process_words(" ".join(ref_words), " ".join(hyp_words))
    totals = defaultdict(int)
    errors = defaultdict(int)
    normalised = defaultdict(int)
    critical = []

    for ch in out.alignments[0]:
        if ch.type == "equal":
            for k in range(ch.ref_end_idx - ch.ref_start_idx):
                c = classify(ref_words[ch.ref_start_idx + k])
                if c:
                    totals[c] += 1

        elif ch.type == "substitute":
            for k in range(ch.ref_end_idx - ch.ref_start_idx):
                rw = ref_words[ch.ref_start_idx + k]
                hw = hyp_words[ch.hyp_start_idx + k]
                c = classify(rw)
                if c:
                    totals[c] += 1
                    # A formatting difference is not a clinical error. Count it
                    # separately so the normalisation effect stays visible.
                    if equivalent(rw, hw):
                        normalised[c] += 1
                        continue
                    errors[c] += 1
                    # A wrong number is a wrong dose or a wrong duration. A
                    # negation replaced by anything non-negating reverses meaning.
                    if c in ("number", "negation", "medication"):
                        critical.append((c, rw, hw, "substituted"))

        elif ch.type == "delete":
            for k in range(ch.ref_end_idx - ch.ref_start_idx):
                rw = ref_words[ch.ref_start_idx + k]
                c = classify(rw)
                if c:
                    totals[c] += 1
                    errors[c] += 1
                    # A dropped negation turns "no chest pain" into "chest pain".
                    if c in ("number", "negation", "medication"):
                        critical.append((c, rw, "", "deleted"))

    row = {"wer": round(out.wer, 4), "ref_words": len(ref_words)}
    for name in ["medication", "condition", "symptom", "anatomy", "number",
                 "temporal", "negation", "unit", "filler"]:
        row[f"{name}_n"] = totals[name]
        row[f"{name}_err"] = errors[name]
        row[f"{name}_norm"] = normalised[name]
        row[f"{name}_rate"] = round(errors[name] / totals[name], 4) if totals[name] else None

    clin = ["medication", "condition", "symptom", "anatomy", "number", "unit"]
    ct = sum(totals[c] for c in clin)
    ce = sum(errors[c] for c in clin)
    row["clinical_n"] = ct
    row["clinical_err"] = ce
    row["clinical_rate"] = round(ce / ct, 4) if ct else None
    row["critical_n"] = len(critical)

    examples.extend(critical)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-dir", required=True)
    ap.add_argument("--ref-dir", required=True)
    ap.add_argument("--csv", default="clinical_errors.csv")
    ap.add_argument("--examples", default="clinical_examples.txt")
    a = ap.parse_args()

    rd, fd = Path(a.result_dir), Path(a.ref_dir)
    rows, all_examples, all_ref_words = [], [], []

    for xlsx in sorted(rd.glob("*_large_v3_review.xlsx")):
        case = xlsx.stem.split("_large_v3")[0]
        ref = fd / f"{case}.txt"
        if not ref.exists():
            continue
        ref_pairs = parse_reference(ref)
        all_ref_words += [w for w, _ in ref_pairs]
        ex = []
        m = analyse(ref_pairs, parse_result(xlsx), ex)
        if not m:
            continue
        m["case"] = case
        _cm = re.match(r"^([A-Za-z]+)", case)
        m["category"] = _cm.group(1) if _cm else "NA"
        rows.append(m)
        all_examples += [(case, *e) for e in ex]

    if not rows:
        sys.exit("nothing scored")

    df = pd.DataFrame(rows)
    cols = ["case", "category", "wer", "clinical_rate", "critical_n"] + \
           [c for c in df.columns if c not in ("case", "category", "wer",
                                               "clinical_rate", "critical_n")]
    df[cols].to_csv(a.csv, index=False)

    print(f"\n{'='*72}\nCLINICAL ERROR ANALYSIS - {len(df)} recordings\n{'='*72}")
    print(f"overall WER                  {df.wer.mean()*100:6.2f}%")
    print(f"clinical term error rate     {df.clinical_rate.mean()*100:6.2f}%")
    print(f"critical errors (total)      {int(df.critical_n.sum()):6d}"
          f"   ({df.critical_n.mean():.1f} per recording)")

    print(f"\n{'class':12s} {'n':>7s} {'errors':>7s} {'rate':>8s} {'formatting':>11s}")
    print("-" * 50)
    for name in ["medication", "condition", "symptom", "anatomy", "number",
                 "temporal", "negation", "unit", "filler"]:
        n = int(df[f"{name}_n"].sum())
        e = int(df[f"{name}_err"].sum())
        fm = int(df[f"{name}_norm"].sum())
        if n:
            print(f"{name:12s} {n:7d} {e:7d} {e/n*100:7.2f}% {fm:11d}")
    print("\nThe formatting column counts digit/word and negation-variant")
    print("differences ('8'/'eight', 'nope'/'no'). These are transcription")
    print("conventions, not errors, and are excluded from the rates above.")

    print("\nThe filler row is reported separately on purpose: those errors")
    print("inflate WER but carry no clinical consequence, and a medical reviewer")
    print("should be able to see the clinical figure without them.")

    # critical examples
    with open(a.examples, "w") as f:
        f.write("CRITICAL ERRORS - negation, number/dose and medication\n")
        f.write("=" * 72 + "\n\n")
        for case, cls, rw, hw, kind in all_examples:
            f.write(f"[{case}] {cls:10s} {kind:11s} "
                    f"reference '{rw}' -> {'(dropped)' if not hw else repr(hw)}\n")
    print(f"\n{len(all_examples)} critical errors -> {a.examples}")

    if all_examples:
        print("\nfirst 12:")
        for case, cls, rw, hw, kind in all_examples[:12]:
            print(f"  [{case}] {cls:10s} '{rw}' -> "
                  f"{'(dropped)' if not hw else repr(hw)}")

    mined = mine_corpus_terms(all_ref_words)
    if mined:
        print("\nfrequent unclassified terms - review and add to the lexicon:")
        print("  " + ", ".join(f"{w}({c})" for w, c in mined[:20]))

    print(f"\nSaved -> {a.csv}")


if __name__ == "__main__":
    main()
