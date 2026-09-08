import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]
                       / "analysis" / "post_freeze"))
import negation_insertion_audit as A

CUES = {"no", "not", "never", "none", "nope", "denies", "without"}
EQ = A.build_equiv([("no", "nope"), ("not", "never")])

def ops(r, h):
    return A.align(r.split(), h.split())

def cls(r, h):
    return A.classify(ops(r, h), CUES, EQ)

fails = 0
def chk(name, got, exp):
    global fails
    ok = all(got.get(k, 0) == v for k, v in exp.items())
    if not ok:
        fails += 1
    print(f"{'PASS' if ok else 'FAIL'}  {name}")
    if not ok:
        print(f"      got {dict(got)}")
        print(f"      exp {exp}")

# --- alignment sanity
o = ops("a b c", "a b c")
chk("identical -> all correct", {"n": sum(1 for k,_,_ in o if k=="C")}, {"n": 3})

o = ops("a b c", "a x c")
chk("one substitution", {"n": sum(1 for k,_,_ in o if k=="S")}, {"n": 1})

o = ops("a b c", "a c")
chk("one deletion", {"n": sum(1 for k,_,_ in o if k=="D")}, {"n": 1})

o = ops("a c", "a b c")
chk("one insertion", {"n": sum(1 for k,_,_ in o if k=="I")}, {"n": 1})

o = ops("", "a b")
chk("empty ref -> 2 insertions", {"n": sum(1 for k,_,_ in o if k=="I")}, {"n": 2})

o = ops("a b", "")
chk("empty hyp -> 2 deletions", {"n": sum(1 for k,_,_ in o if k=="D")}, {"n": 2})

# --- negation classification
chk("cue deleted",
    cls("i have no pain", "i have pain"),
    dict(ref_tokens=1, deleted=1, substituted=0, hallucinated=0))

chk("cue substituted (non-equivalent)",
    cls("i have no pain", "i have some pain"),
    dict(ref_tokens=1, deleted=0, substituted=1, hallucinated=0))

chk("cue substituted by equivalent variant -> not an error",
    cls("i have no pain", "i have nope pain"),
    dict(ref_tokens=1, deleted=0, substituted=0, equivalent=1, hallucinated=0))

chk("cue correct",
    cls("i have no pain", "i have no pain"),
    dict(ref_tokens=1, correct=1, deleted=0, substituted=0, hallucinated=0))

chk("HALLUCINATED cue (the new class)",
    cls("i have pain", "i have no pain"),
    dict(ref_tokens=0, deleted=0, substituted=0, hallucinated=1))

chk("hallucination does not inflate ref denominator",
    cls("chest pain today", "chest pain not today"),
    dict(ref_tokens=0, hallucinated=1))

chk("meaning-inverting hallucination is caught",
    cls("patient denies fever", "patient denies no fever"),
    dict(ref_tokens=1, correct=1, hallucinated=1))

chk("multiple mixed events",
    cls("no fever not chills never dizzy",
        "fever some chills never dizzy no cough"),
    dict(ref_tokens=3, deleted=1, substituted=1, correct=1, hallucinated=1))

chk("cue->cue substitution across classes counts as substituted",
    cls("i have no pain", "i have never pain"),
    dict(ref_tokens=1, substituted=1))

# --- denominator invariance: hallucinations must never change ref_tokens
a = cls("no no no", "no no no")
b = cls("no no no", "no no no no no")
chk("denominator invariant to insertions",
    {"same": a["ref_tokens"] == b["ref_tokens"] == 3, "h": b["hallucinated"]},
    {"same": True, "h": 2})

print("\nFAILURES:", fails)
sys.exit(1 if fails else 0)
