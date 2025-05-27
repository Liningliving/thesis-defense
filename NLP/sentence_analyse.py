import subprocess
import os

result = subprocess.run('bash -c "source /etc/network_turbo && env | grep proxy"', shell=True, capture_output=True, text=True)
output = result.stdout
for line in output.splitlines():
    if '=' in line:
        var, value = line.split('=', 1)
        os.environ[var] = value


import logging
# Configure logging to suppress INFO messages
logging.getLogger('stanford_openie').setLevel(logging.WARNING)
from openie import StanfordOpenIE
# 1. Start the client at module scope
client = StanfordOpenIE()  

def retrieve_info(summary, conf_threshold=0.05):
    triples = client.annotate(summary)
    return triples

if __name__ == "__main__":
    print(retrieve_info("The image shows a towel on the floor."))

''' 
import spacy
from spacy.matcher import DependencyMatcher
import re

nlp = spacy.load("en_core_web_sm")
matcher = DependencyMatcher(nlp.vocab)

# Pattern for “subj — nsubjpass — rel_token — prep:on — pobj — obj”
pattern = [
    {"RIGHT_ID": "verb",      "RIGHT_ATTRS": {"LEMMA": "position"}},
    {"LEFT_ID": "verb",       "REL_OP": ">", "RIGHT_ID": "subj",     "RIGHT_ATTRS": {"LOWER": {"REGEX": r"obj\d+"}}},  
    {"LEFT_ID": "verb",       "REL_OP": ">", "RIGHT_ID": "prep",     "RIGHT_ATTRS": {"DEP": "prep"}},  
    {"LEFT_ID": "prep",       "REL_OP": ">", "RIGHT_ID": "obj",      "RIGHT_ATTRS": {"LOWER": {"REGEX": r"obj\d+"}}}
]
pattern_prep = [
    {"RIGHT_ID": "verb",  "RIGHT_ATTRS": {"LEMMA": "position"}},  
    {"LEFT_ID": "verb",   "REL_OP": ">", "RIGHT_ID": "subj", "RIGHT_ATTRS": {"LOWER": {"REGEX": r"obj\d+"}}},  
    {"LEFT_ID": "verb",   "REL_OP": ">", "RIGHT_ID": "prep", "RIGHT_ATTRS": {"DEP": "prep"}},  
    {"LEFT_ID": "prep",   "REL_OP": ">", "RIGHT_ID": "obj",  "RIGHT_ATTRS": {"LOWER": {"REGEX": r"obj\d+"}}}
]
pattern_adv = [
    {"RIGHT_ID": "verb", "RIGHT_ATTRS": {"LEMMA": "position"}},
    {"LEFT_ID": "verb", "REL_OP": ">",  "RIGHT_ID": "subj", "RIGHT_ATTRS": {"LOWER": {"REGEX": r"obj\d+"}}},  
    {"LEFT_ID": "verb", "REL_OP": ">",  "RIGHT_ID": "adv",  "RIGHT_ATTRS": {"DEP": "advmod", "LEMMA": {"IN": ["below","above"]}}},
    {"LEFT_ID": "adv",  "REL_OP": ">",  "RIGHT_ID": "obj",  "RIGHT_ATTRS": {"LOWER": {"REGEX": r"obj\d+"}}}
]
pattern_prep = [
    {"RIGHT_ID": "verb", "RIGHT_ATTRS": {"LEMMA": "position"}},
    {"LEFT_ID": "verb", "REL_OP": ">", "RIGHT_ID": "subj", "RIGHT_ATTRS": {"LOWER": {"REGEX": r"obj\d+"}}},  
    {"LEFT_ID": "verb", "REL_OP": ">", "RIGHT_ID": "prep", "RIGHT_ATTRS": {"DEP": "prep", "LOWER": {"IN": ["to"]}}},
    {"LEFT_ID": "prep", "REL_OP": ">", "RIGHT_ID": "pobj", "RIGHT_ATTRS": {"DEP": "pobj", "LOWER": {"REGEX": r"(?:the\s)?(left|right)"}}},  
    {"LEFT_ID": "pobj", "REL_OP": ">", "RIGHT_ID": "obj",  "RIGHT_ATTRS": {"LOWER": {"REGEX": r"obj\d+"}}}
]
matcher.add("PREP_REL", [pattern_prep])
matcher.add("ADV_REL", [pattern_adv])
matcher.add("PREP_REL", [pattern_prep])
matcher.add("POSITION_REL", [pattern])


# 4) Regex fallback
regex = re.compile(
    r"\b(obj\d+)\b.*?\b(positioned (?:below|above|to the left of|to the right of))\b.*?\b(obj\d+)\b",
    flags=re.IGNORECASE
)

def extract_relationship_with_dependency(line):
    doc = nlp(line)
    triples = set()

    for _, token_ids in matcher(doc):
        verb, subj, mod, *rest = [doc[i] for i in token_ids]
        if mod.dep_ == "advmod":
            rel = f"positioned {mod.lemma_}"
            obj = rest[-1].text
        elif mod.dep_ == "prep":
            direction = rest[0].text
            rel = f"positioned to the {direction} of"
            obj = rest[-1].text
        else:
            # BASE_REL: assume nearest object is object
            rel = "positioned"
            obj = rest[-1].text
        triples.add((subj.text, rel, obj))

    # Regex fallback
    for s, r, o in regex.findall(line):
        triples.add((s, r, o))

    # Clean up spurious "of"
    cleaned = set()
    for s, r, o in triples:
        if r.endswith("below of"):
            r = "positioned below"
        elif r.endswith("above of"):
            r = "positioned above"
        cleaned.add((s, r, o))

    # Keep only obj\d+ relations
    return [
        (s, r, o) for (s, r, o) in cleaned
        if re.fullmatch(r"obj\d+", s) and re.fullmatch(r"obj\d+", o)
    ]


def clean_triples(raw_triples, valid_objs):
    cleaned = set()
    for s, r, o in raw_triples:
        # 1. Schema checks
        if s not in valid_objs or o not in valid_objs:
            continue
        # 2. Relation normalization
        r_norm = r.replace(" is positioned ", "positioned_")\
                  .replace(" to the left of", "_left_of")\
                  .replace(" to the right of", "_right_of")\
                  .strip().lower()
        # 3. Drop spurious
        if r_norm in ("is", "are") or "positioned" not in r_norm:
            continue
        # 4. Length heuristic
        tokens = r_norm.split("_")
        if len(tokens) < 2 or len(tokens) > 4:
            continue
        cleaned.add((s, r_norm, o))

    # 5. Consistency & symmetry
    final = set(cleaned)
    for s, r, o in list(cleaned):
        if r == "positioned_below":
            final.add((o, "positioned_above", s))
    return sorted(final)
''' 