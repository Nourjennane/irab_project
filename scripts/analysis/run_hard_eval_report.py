"""Per-hard-bucket eval report.

Runs a checkpoint over every ``data_v2/hard_eval/<bucket>/all.jsonl``
and reports fully / role / case / marker accuracy per bucket.
Output: ``docs/hard_eval/hard_eval_report.md`` + ``hard_eval.json``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def _load_model(ckpt_dir: Path, encoder_name: str, device):
    import torch
    from irab_tashkeel.morphology.dep_aware_model import DepAwareStructuredModel

    sd = torch.load(ckpt_dir / "pytorch_model.bin", map_location="cpu",
                    weights_only=True)
    has_graph = any(k.startswith("graph_refiner.") or k == "graph_gate"
                    for k in sd)
    model = DepAwareStructuredModel(
        encoder_name=encoder_name,
        enable_morph_heads=True, morph_heads_enabled=None,
        enable_dep_features=True, enable_graph_refiner=has_graph,
    )
    model.load_state_dict(sd, strict=False)
    model.to(device); model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(ROOT / "runs" / "validated_nextgen_recovery"))
    ap.add_argument("--hard_root", default=str(ROOT / "data_v2" / "hard_eval"))
    ap.add_argument("--out_dir", default=str(ROOT / "docs" / "hard_eval"))
    ap.add_argument("--encoder_name", default="UBC-NLP/AraT5v2-base-1024")
    ap.add_argument("--batch_size", type=int, default=16)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import torch
    from transformers import AutoTokenizer
    from irab_tashkeel.data_v2.schema_v2 import read_jsonl
    from irab_tashkeel.training_v2.eval_hook import predict_for_eval
    from irab_tashkeel.eval_v2 import aggregate_outcomes, extract_outcomes

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint if (Path(args.checkpoint) / "tokenizer.json").exists()
        else args.encoder_name,
    )
    model = _load_model(Path(args.checkpoint), args.encoder_name, device)

    rows: List[Dict] = []
    for bucket_dir in sorted(Path(args.hard_root).glob("*/")):
        if not (bucket_dir / "all.jsonl").exists():
            continue
        sents = list(read_jsonl(str(bucket_dir / "all.jsonl")))
        if not sents:
            continue
        preds = predict_for_eval(model, tokenizer, sents,
                                  batch_size=args.batch_size, device=device)
        outcomes = extract_outcomes(sents, preds)
        agg = aggregate_outcomes(outcomes)
        strict = aggregate_outcomes(
            [o for o in outcomes if o.is_fully_observable]
        )
        rows.append({
            "bucket":   bucket_dir.name,
            "n_sent":   len(sents),
            "n_words":  agg["n_words"],
            "case_acc": agg["case_acc"],
            "role_acc": agg["role_acc"],
            "marker_em": agg["marker_em"],
            "fully":    agg["fully"],
            "strict_fully": strict["fully"],
            "strict_n_words": strict["n_words"],
        })
        print(f"  {bucket_dir.name:25} fully={agg['fully']:.3f} "
              f"strict_fully={strict['fully']:.3f}")

    # Markdown
    lines = [f"# Hard-eval per-bucket report",
             f"",
             f"Checkpoint: `{args.checkpoint}`",
             "",
             "| bucket | n_sent | n_words | case | role | marker | fully | strict_fully |",
             "|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: r["fully"]):
        lines.append(f"| {r['bucket']} | {r['n_sent']} | {r['n_words']} | "
                     f"{r['case_acc']:.3f} | {r['role_acc']:.3f} | "
                     f"{r['marker_em']:.3f} | {r['fully']:.3f} | "
                     f"{r['strict_fully']:.3f} |")
    (out_dir / "hard_eval_report.md").write_text("\n".join(lines))
    (out_dir / "hard_eval.json").write_text(
        json.dumps({"checkpoint": args.checkpoint, "rows": rows},
                   indent=2, ensure_ascii=False)
    )
    print(f"\n✓ wrote {out_dir / 'hard_eval_report.md'}")


if __name__ == "__main__":
    main()
