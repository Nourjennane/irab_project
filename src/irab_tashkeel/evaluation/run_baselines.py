"""End-to-end evaluation harness comparing baselines on Gazelle (gold MSA).

Compares (any subset of):
  - claude-zero-shot
  - claude-fewshot-rag (using Yarob as retrieval pool)
  - per-word decoder (your existing model checkpoint)

Metric: structural extraction. Reports:
  - n
  - well-formedness rate
  - case accuracy
  - role-F1 (macro)
  - marker exact-match
  - pos accuracy

Each prediction is also passed through `inference.constrained.constrain` to
collapse it to canonical taxonomy templates before scoring.

Usage:
    ANTHROPIC_API_KEY=... python -m irab_tashkeel.evaluation.run_baselines \\
        --eval gazelle \\
        --baselines claude_zero,claude_rag \\
        --out runs/baseline_eval/
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from ..data.gazelle import GazelleItem, load_gazelle_iraab
from ..evaluation.structural import StructuralMetrics, split_sentence_iraab
from ..inference.constrained import constrain


def _gold_pairs_from_gazelle(items: Sequence[GazelleItem]) -> List[Tuple[str, List[Tuple[str, str]]]]:
    """For each Gazelle item, parse the gold answer into (word, irab) pairs."""
    out: List[Tuple[str, List[Tuple[str, str]]]] = []
    for it in items:
        pairs = split_sentence_iraab(it.answer)
        if pairs:
            out.append((it.sentence, pairs))
    return out


def _align_pred(gold_words: Sequence[str], preds: Sequence[Dict]) -> List[str]:
    """Align prediction items to gold words by surface match (best effort).

    For Gazelle, gold words may have diacritics; preds usually don't. We strip
    diacritics on both sides and look up by exact match. Words not found get "".
    """
    import re, unicodedata
    def norm(s: str) -> str:
        s = unicodedata.normalize("NFC", s or "")
        s = re.sub(r"[ً-ْٰ]", "", s)
        return re.sub(r"[^ء-ي]+", "", s)

    pred_by_word = {norm(p.get("word", "")): p.get("irab", "") for p in preds if p.get("word")}
    out: List[str] = []
    for gw in gold_words:
        out.append(pred_by_word.get(norm(gw), ""))
    return out


def evaluate_baseline(
    name: str,
    sentences: Sequence[str],
    gold_word_irabs: Sequence[List[Tuple[str, str]]],
    predict_fn,
    out_dir: Path,
    apply_constrain: bool = True,
) -> Dict:
    """Run predict_fn on each sentence, score against gold, save predictions."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = StructuralMetrics()
    metrics_constrained = StructuralMetrics()

    pred_path = out_dir / f"{name}.predictions.jsonl"
    n_ok = 0
    with open(pred_path, "w", encoding="utf-8") as fpred:
        for sentence, gold_pairs in zip(sentences, gold_word_irabs):
            try:
                pred_items = predict_fn(sentence)
            except Exception as e:
                print(f"  [{name}] error on '{sentence[:40]}…': {e}")
                continue
            pred_dicts = [p.to_dict() if hasattr(p, "to_dict") else dict(p) for p in pred_items]
            gold_words = [w for w, _ in gold_pairs]
            aligned_irabs = _align_pred(gold_words, pred_dicts)

            for (gold_word, gold_irab), pred_irab in zip(gold_pairs, aligned_irabs):
                metrics.update(gold_irab, pred_irab)
                if apply_constrain:
                    metrics_constrained.update(gold_irab, constrain(pred_irab) if pred_irab else "")

            fpred.write(json.dumps({
                "sentence": sentence,
                "gold": [{"word": w, "irab": i} for w, i in gold_pairs],
                "pred": pred_dicts,
            }, ensure_ascii=False) + "\n")
            n_ok += 1
    print(f"  [{name}] scored {n_ok} sentences")

    report = {
        "baseline": name,
        "n_sentences": n_ok,
        "raw": metrics.report(),
        "constrained": metrics_constrained.report() if apply_constrain else None,
    }
    with open(out_dir / f"{name}.metrics.json", "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"  [{name}] raw:        {metrics.pretty()}")
    if apply_constrain:
        print(f"  [{name}] constrained: {metrics_constrained.pretty()}")
    return report


def per_word_decoder_predictor(checkpoint_path: str):
    """Wrapper that builds a callable returning [{word, irab}, ...] from your trained model."""
    from ..inference.predictor import Predictor
    pred = Predictor.from_checkpoint(checkpoint_path)
    def _predict(sentence: str):
        result = pred.predict(sentence)
        return [{"word": w["surface"], "irab": w.get("irab_text", "")} for w in result.words]
    return _predict


def main():
    p = argparse.ArgumentParser(description="Compare i'rāb baselines on Gazelle")
    p.add_argument("--eval", choices=["gazelle"], default="gazelle")
    p.add_argument("--baselines", default="claude_zero,claude_rag",
                   help="comma-separated subset: claude_zero, claude_rag, decoder")
    p.add_argument("--decoder_ckpt", default=None,
                   help="path to FullModel .pt for the 'decoder' baseline")
    p.add_argument("--marker_model", default=None,
                   help="path to AraT5v2 marker fine-tune for the 'hybrid' baseline")
    p.add_argument("--model", default="claude-haiku-4-5",
                   help="Claude model id for the LLM baselines")
    p.add_argument("--rag_k", type=int, default=5)
    p.add_argument("--no_distilled_pool", action="store_true",
                   help="exclude data/distilled_irab.jsonl from the RAG retrieval pool "
                        "(default: include it on top of Yarob)")
    p.add_argument("--out", type=Path, default=Path("runs/baseline_eval"))
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    # ---- Eval set ----
    if args.eval == "gazelle":
        items = load_gazelle_iraab()
        gold = _gold_pairs_from_gazelle(items)
        print(f"loaded {len(gold)} gazelle items as gold pairs")
    else:
        raise ValueError(f"unknown --eval {args.eval}")
    sentences = [s for s, _ in gold]
    gold_pairs = [g for _, g in gold]

    baselines = [b.strip() for b in args.baselines.split(",") if b.strip()]
    reports = []

    if "claude_zero" in baselines:
        from ..inference.llm_baselines import claude_zero_shot
        rep = evaluate_baseline(
            "claude_zero", sentences, gold_pairs,
            predict_fn=lambda s: claude_zero_shot(s, model=args.model),
            out_dir=args.out,
        )
        reports.append(rep)

    if "claude_rag" in baselines:
        from ..inference.llm_baselines import (
            claude_fewshot_rag, load_combined_fewshots,
        )
        pool = load_combined_fewshots(
            include_yarob=True,
            include_distilled=not args.no_distilled_pool,
        )
        print(f"  RAG pool: {len(pool)} examples "
              f"(distilled={'yes' if not args.no_distilled_pool else 'no'})")
        rep = evaluate_baseline(
            "claude_rag", sentences, gold_pairs,
            predict_fn=lambda s: claude_fewshot_rag(s, pool, k=args.rag_k, model=args.model),
            out_dir=args.out,
        )
        reports.append(rep)

    if "hybrid" in baselines:
        if not args.marker_model:
            print("  [hybrid] skipped — pass --marker_model path/to/final/")
        else:
            from ..inference.hybrid import HybridPredictor
            from ..inference.llm_baselines import load_combined_fewshots
            pool = load_combined_fewshots(
                include_yarob=True,
                include_distilled=not args.no_distilled_pool,
            )
            print(f"  Hybrid: marker_model={args.marker_model}  RAG pool={len(pool)}")
            hybrid = HybridPredictor(
                marker_model_path=args.marker_model,
                rag_pool=pool, rag_k=args.rag_k, rag_model=args.model,
            )
            rep = evaluate_baseline(
                "hybrid", sentences, gold_pairs,
                predict_fn=hybrid.predict,
                out_dir=args.out,
            )
            reports.append(rep)

    if "decoder" in baselines:
        if not args.decoder_ckpt:
            print("  [decoder] skipped — pass --decoder_ckpt path/to/best.pt")
        else:
            predict_fn = per_word_decoder_predictor(args.decoder_ckpt)
            class _Item:
                def __init__(self, d): self.__dict__.update(d)
                def to_dict(self): return self.__dict__
            wrapped = lambda s: [_Item(d) for d in predict_fn(s)]
            rep = evaluate_baseline(
                "decoder", sentences, gold_pairs,
                predict_fn=wrapped, out_dir=args.out,
            )
            reports.append(rep)

    # Summary table
    summary = args.out / "summary.json"
    with open(summary, "w") as f:
        json.dump(reports, f, indent=2, ensure_ascii=False)
    print(f"\n=== Summary written to {summary} ===")
    print(f"{'baseline':16} {'well':>8} {'pos':>8} {'case':>8} {'role-F1':>10} {'marker':>8}")
    for r in reports:
        c = r["constrained"] if r["constrained"] else r["raw"]
        if c:
            print(f"{r['baseline']:16} "
                  f"{c['well_formed_rate']*100:>7.1f}% "
                  f"{c['pos_accuracy']*100:>7.1f}% "
                  f"{c['case_accuracy']*100:>7.1f}% "
                  f"{c['role_f1_macro']*100:>9.1f}% "
                  f"{c['marker_em']*100:>7.1f}%")


if __name__ == "__main__":
    main()
