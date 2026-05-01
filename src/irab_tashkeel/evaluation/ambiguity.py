"""Annotator-disagreement / ambiguity analysis for Gazelle gold.

Hovy 2019 ("The Ecological Fallacy in Annotation"), Plank 2022, and the broad
"single-gold" critique in NLP all point at the same problem: when expert
annotators genuinely disagree on the right label, scoring systems against ONE
canonical gold label confounds (a) systems being wrong with (b) systems
choosing one of several valid analyses the gold doesn't enumerate. Classical
Arabic naḥw is exactly such a domain — multiple grammatical schools admit
alternative i'rāb for many constructions (e.g. ḥāl vs naʿt, mubtadaʾ vs
fāʿil for some preverbal nouns, taʿalluq of a jārr-majrūr, etc.).

This module:
  1. Asks Claude Sonnet 4.5 for each Gazelle gold word: "are there
     alternative valid i'rāb analyses for this word in this sentence
     according to classical Arabic grammar?"
  2. Records each word's ambiguity class:
       - unambiguous       (one valid analysis)
       - ambiguous_minor   (multiple analyses, but case is fixed)
       - ambiguous_major   (multiple analyses with different cases)
     and the list of alternative analyses (free-form Arabic prose).
  3. Re-scores existing system predictions with PERMISSIVE matching:
     a system is correct on a word if its prediction matches ANY of
     {gold} ∪ {alternatives} on the structural fields.
  4. Reports strict (current) vs permissive accuracy per system; the
     gap quantifies how much "single gold" understates real performance.

Cost: ~$0.50 to annotate the 30 Gazelle sentences once with Sonnet 4.5.

Usage:
    ANTHROPIC_API_KEY=... python -m irab_tashkeel.evaluation.ambiguity \\
        --annotate --out data/ambiguity_annotations.jsonl

    python -m irab_tashkeel.evaluation.ambiguity \\
        --score \\
        --annotations data/ambiguity_annotations.jsonl \\
        --system "stanza=runs/baseline_eval_stanza/stanza.predictions.jsonl" \\
        --system "haiku_rag=runs/baseline_eval_v2/claude_rag.predictions.jsonl" \\
        --system "sonnet_rag=runs/baseline_eval_sonnet/claude_rag.predictions.jsonl" \\
        --out runs/ambiguity_analysis/results.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from ..data.gazelle import load_gazelle_iraab
from .structural import extract, split_sentence_iraab


# ---------------------------------------------------------------------------
# Annotation step (Sonnet)
# ---------------------------------------------------------------------------
ANNOTATION_SYSTEM = (
    "أنت أستاذ متخصص في النحو العربي الكلاسيكي. لكل كلمة، حدّد ما إذا كان "
    "إعرابها يقبل أكثر من تحليل صحيح وفق مدارس النحو الكلاسيكي (سيبويه، "
    "البصرة، الكوفة، …)."
)

ANNOTATION_USER_TEMPLATE = """الجملة: {sentence}

إعراب الكلمات (الإعراب الموثّق المعطى):
{gold_block}

لكل كلمة من الكلمات أعلاه، أجب بكائن JSON بالحقول التالية فقط:
- "word": سطح الكلمة كما ورد
- "ambiguity_class": أحد ["unambiguous", "ambiguous_minor", "ambiguous_major"]
   * unambiguous: تحليل واحد صحيح فقط
   * ambiguous_minor: عدة تحليلات صحيحة لكن الحالة الإعرابية ثابتة (مرفوع/منصوب/مجرور/مجزوم)
   * ambiguous_major: عدة تحليلات صحيحة بحالات مختلفة (مثلاً فاعل أو مبتدأ)
- "alternatives": قائمة بالتحليلات البديلة (لا تكرّر التحليل المعطى) كنصّ عربي قصير لكلٍّ منها
- "rationale": سبب موجز جدًا (سطر واحد) للتصنيف

أرجع مصفوفة JSON واحدة بنفس ترتيب الكلمات أعلاه. لا تضف أي نصّ خارج المصفوفة."""


def _normalize(s: str) -> str:
    s = unicodedata.normalize("NFC", s or "")
    s = re.sub(r"[ً-ْٰ]", "", s)
    return re.sub(r"[^ء-ي]+", "", s)


def _claude_annotate(sentence: str, gold_pairs: Sequence[Tuple[str, str]],
                     model: str = "claude-sonnet-4-5") -> List[dict]:
    """Annotate one sentence's words with ambiguity tags via Sonnet."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    from anthropic import Anthropic
    client = Anthropic()

    gold_block = "\n".join(f"- {w}: {ir}" for w, ir in gold_pairs)
    user = ANNOTATION_USER_TEMPLATE.format(sentence=sentence, gold_block=gold_block)
    r = client.messages.create(
        model=model, max_tokens=2048, temperature=0.0,
        system=ANNOTATION_SYSTEM,
        messages=[{"role": "user", "content": user}],
    )
    text = "".join(b.text for b in r.content if b.type == "text").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
    try:
        items = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\[.*\]", text, flags=re.DOTALL)
        items = json.loads(m.group(0)) if m else []
    if not isinstance(items, list):
        return []
    return [it for it in items if isinstance(it, dict)]


def annotate_gazelle(out_path: Path | str = "data/ambiguity_annotations.jsonl",
                     model: str = "claude-sonnet-4-5") -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    items = load_gazelle_iraab()
    n_total = 0
    n_amb_minor = 0
    n_amb_major = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for it in items:
            pairs = split_sentence_iraab(it.answer)
            if not pairs:
                continue
            try:
                annots = _claude_annotate(it.sentence, pairs, model=model)
            except Exception as e:
                print(f"  ! error on '{it.sentence[:40]}…': {e}")
                continue
            # Align annots to gold by word surface (best effort)
            ann_by_word = {_normalize(a.get("word", "")): a for a in annots}
            for word, gold_irab in pairs:
                a = ann_by_word.get(_normalize(word), {})
                rec = {
                    "sentence": it.sentence,
                    "word": word,
                    "gold_irab": gold_irab,
                    "ambiguity_class": a.get("ambiguity_class", "unannotated"),
                    "alternatives": a.get("alternatives") or [],
                    "rationale": a.get("rationale", ""),
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_total += 1
                if rec["ambiguity_class"] == "ambiguous_minor": n_amb_minor += 1
                if rec["ambiguity_class"] == "ambiguous_major": n_amb_major += 1
            print(f"  ✓ {it.sentence[:50]:<52} ({len(pairs)} words)")
    print(f"\nwrote {n_total} word annotations → {out_path}")
    print(f"  unambiguous:      {n_total - n_amb_minor - n_amb_major}")
    print(f"  ambiguous_minor:  {n_amb_minor}")
    print(f"  ambiguous_major:  {n_amb_major}")
    return out_path


# ---------------------------------------------------------------------------
# Permissive scoring
# ---------------------------------------------------------------------------
def load_predictions(path: Path) -> Dict[Tuple[str, str], str]:
    out: Dict[Tuple[str, str], str] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            sent = row.get("sentence", "")
            for p in row.get("pred") or []:
                w = p.get("word", "") if isinstance(p, dict) else ""
                ir = p.get("irab", "") if isinstance(p, dict) else ""
                if w:
                    out[(sent, _normalize(w))] = ir
    return out


def load_annotations(path: Path) -> List[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def _matches_any(pred_irab: str, gold_irab: str, alternatives: List[str],
                 dimension: str) -> bool:
    """Does the pred match the gold OR any alternative on `dimension`?"""
    p = extract(pred_irab)
    targets = [extract(gold_irab)] + [extract(alt) for alt in alternatives]
    for t in targets:
        if dimension == "case" and t.case is not None and t.case == p.case:
            return True
        if dimension == "role" and t.role is not None and t.role == p.role:
            return True
        if dimension == "marker" and t.marker is not None and t.marker == p.marker:
            return True
        if dimension == "fully":
            if (t.case is not None and t.case == p.case
                and t.role is not None and t.role == p.role
                and t.marker is not None and t.marker == p.marker):
                return True
    return False


def score_with_permissive(annotations: List[dict],
                          systems: Dict[str, Path]) -> Dict[str, dict]:
    sys_preds = {name: load_predictions(p) for name, p in systems.items()}
    out: Dict[str, dict] = {}
    for sys_name, preds in sys_preds.items():
        per_dim_strict: Dict[str, List[int]] = {d: [] for d in ("case", "role", "marker", "fully")}
        per_dim_perm:   Dict[str, List[int]] = {d: [] for d in ("case", "role", "marker", "fully")}
        per_class_perm: Dict[str, Dict[str, List[int]]] = {
            d: defaultdict(list) for d in ("case", "role", "marker", "fully")
        }
        for a in annotations:
            sent = a["sentence"]
            word_n = _normalize(a["word"])
            pred_irab = preds.get((sent, word_n), "")
            cls = a.get("ambiguity_class", "unannotated")
            # Counted whether or not the system emitted a prediction; missing pred
            # is treated as wrong on every dimension. This matches the headline
            # `StructuralMetrics` semantics on the same 134-word eval.
            for dim in ("case", "role", "marker", "fully"):
                if not pred_irab:
                    strict = perm = False
                else:
                    strict = _matches_any(pred_irab, a["gold_irab"], [], dim)
                    perm   = _matches_any(pred_irab, a["gold_irab"], a.get("alternatives") or [], dim)
                per_dim_strict[dim].append(int(strict))
                per_dim_perm[dim].append(int(perm))
                per_class_perm[dim][cls].append(int(perm))
        report = {
            "n": len(per_dim_strict["case"]),
            "strict": {
                d: {"n": len(v), "rate_pct": (sum(v) / len(v) * 100) if v else 0.0}
                for d, v in per_dim_strict.items()
            },
            "permissive": {
                d: {"n": len(v), "rate_pct": (sum(v) / len(v) * 100) if v else 0.0}
                for d, v in per_dim_perm.items()
            },
            "permissive_by_class": {
                d: {cls: {"n": len(vs), "rate_pct": (sum(vs)/len(vs)*100) if vs else 0.0}
                    for cls, vs in by_cls.items()}
                for d, by_cls in per_class_perm.items()
            },
        }
        out[sys_name] = report
    return out


def pretty(scores: Dict[str, dict], systems: List[str]) -> str:
    lines = []
    lines.append("\n  Strict vs permissive scoring (permissive: matches gold OR any alternative analysis)")
    head = f"  {'system':18}  {'metric':6}  {'strict %':>10}  {'permissive %':>14}  {'gap pp':>8}"
    lines.append(head)
    for s in systems:
        r = scores[s]
        for d in ("case", "role", "marker", "fully"):
            st = r["strict"][d]["rate_pct"]
            pm = r["permissive"][d]["rate_pct"]
            lines.append(f"  {s:18}  {d:6}  {st:>9.1f}%  {pm:>13.1f}%  {pm-st:+7.1f}")
        lines.append("")
    return "\n".join(lines)


def class_summary(annotations: List[dict]) -> Dict[str, int]:
    return Counter(a.get("ambiguity_class", "unannotated") for a in annotations)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Ambiguity annotation + permissive scoring")
    p.add_argument("--annotate", action="store_true", help="run Sonnet annotation pass")
    p.add_argument("--score", action="store_true", help="run permissive scoring")
    p.add_argument("--annotations", type=Path, default=Path("data/ambiguity_annotations.jsonl"))
    p.add_argument("--out", type=Path, default=Path("runs/ambiguity_analysis/results.json"))
    p.add_argument("--system", action="append", default=[], metavar="NAME=PATH")
    p.add_argument("--model", default="claude-sonnet-4-5")
    args = p.parse_args()

    if args.annotate:
        annotate_gazelle(args.annotations, model=args.model)

    if args.score:
        ann = load_annotations(args.annotations)
        cls = class_summary(ann)
        print(f"\n  ambiguity-class distribution across {len(ann)} word annotations:")
        for k, v in cls.most_common():
            print(f"    {k:20} {v:>4}")

        if not args.system:
            print("(no --system arguments → skipping scoring)")
            return

        systems: Dict[str, Path] = {}
        for spec in args.system:
            name, path = spec.split("=", 1)
            systems[name] = Path(path)

        scores = score_with_permissive(ann, systems)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({
                "ambiguity_class_distribution": dict(cls),
                "per_system": scores,
            }, f, indent=2, ensure_ascii=False)
        print(pretty(scores, list(systems.keys())))
        print(f"\n  wrote → {args.out}")


if __name__ == "__main__":
    main()
