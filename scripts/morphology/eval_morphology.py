"""Phase 1 evaluation: per-feature morphology accuracy + confusion + calibration.

Loads a trained Phase 1 model (with morph heads), runs it over UD-PADT
held-out (test split by default), and reports:

* per-feature accuracy (macro across all 7 morph heads)
* per-feature confusion matrices
* per-feature calibration (mean confidence on correct vs wrong)
* per-POS accuracy (e.g. gender accuracy on nouns vs verbs)

Outputs JSON + per-feature confusion CSVs to ``--out_dir``.

Usage:
    python scripts/morphology/eval_morphology.py \\
        --model runs/phase1_morph_490987/final \\
        --conllu data/ud_padt/ar_padt-ud-test.conllu \\
        --out_dir runs/phase1_morph_eval_490987
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--conllu", default="data/ud_padt/ar_padt-ud-test.conllu")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_words", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    import torch
    from transformers import AutoTokenizer

    from irab_tashkeel.morphology.morph_model import MorphAugmentedStructuredModel
    from irab_tashkeel.morphology.schema import (
        MORPH_FEATURES,
        GENDER_TO_ID, NUMBER_TO_ID, DEFINITE_TO_ID, PERSON_TO_ID,
        ASPECT_TO_ID, MOOD_TO_ID, VOICE_TO_ID,
        ID_TO_GENDER, ID_TO_NUMBER, ID_TO_DEFINITE, ID_TO_PERSON,
        ID_TO_ASPECT, ID_TO_MOOD, ID_TO_VOICE,
    )
    from irab_tashkeel.morphology.ud_loader import parse_conllu

    feat_to_id = {
        "gender": GENDER_TO_ID, "number": NUMBER_TO_ID,
        "definite": DEFINITE_TO_ID, "person": PERSON_TO_ID,
        "aspect": ASPECT_TO_ID, "mood": MOOD_TO_ID, "voice": VOICE_TO_ID,
    }
    feat_id_to_str = {
        "gender": ID_TO_GENDER, "number": ID_TO_NUMBER,
        "definite": ID_TO_DEFINITE, "person": ID_TO_PERSON,
        "aspect": ID_TO_ASPECT, "mood": ID_TO_MOOD, "voice": ID_TO_VOICE,
    }

    print(f"Loading model from {args.model} ...")
    tcfg_path = Path(args.model) / "structured_config.json"
    encoder_name = "UBC-NLP/AraT5v2-base-1024"
    if tcfg_path.exists():
        encoder_name = json.loads(tcfg_path.read_text()).get("encoder_name", encoder_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    model = MorphAugmentedStructuredModel(
        encoder_name=encoder_name,
        enable_morph_heads=True,
        morph_heads_enabled=set(MORPH_FEATURES),
    )
    sd = torch.load(Path(args.model) / "pytorch_model.bin", map_location="cpu", weights_only=True)
    model.load_state_dict(sd, strict=False)
    model.eval()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model.to(device)

    print(f"Reading UD CoNLL-U from {args.conllu} ...")

    # Per-feature stats: total, correct, confusion matrix
    n_total = Counter()
    n_correct = Counter()
    confusion = {f: defaultdict(lambda: defaultdict(int)) for f in MORPH_FEATURES}
    conf_correct: dict = {f: [] for f in MORPH_FEATURES}
    conf_wrong: dict = {f: [] for f in MORPH_FEATURES}
    per_pos_correct = {f: defaultdict(int) for f in MORPH_FEATURES}
    per_pos_total = {f: defaultdict(int) for f in MORPH_FEATURES}

    n_sent = 0
    n_words = 0
    with torch.no_grad():
        for sent in parse_conllu(args.conllu):
            n_sent += 1
            words = [w.word for w in sent.items[: args.max_words]]
            if not words:
                continue
            # tokenize per-word and build alignment (same logic as the training dataset)
            ids = []
            word_starts = []
            word_ends = []
            kept_idx = []
            for i, w in enumerate(words):
                sub = tokenizer.encode(w, add_special_tokens=False)
                if not sub:
                    continue
                if len(ids) + len(sub) >= 320 - 1:
                    break
                start = len(ids)
                ids.extend(sub)
                word_starts.append(start)
                word_ends.append(len(ids))
                kept_idx.append(i)
            if tokenizer.eos_token_id is not None:
                ids.append(int(tokenizer.eos_token_id))
            kept_words = [sent.items[i] for i in kept_idx]

            input_ids = torch.tensor([ids], dtype=torch.long, device=device)
            attention_mask = torch.ones_like(input_ids)
            ws = torch.tensor([word_starts], dtype=torch.long, device=device)
            we = torch.tensor([word_ends], dtype=torch.long, device=device)
            wm = torch.ones((1, len(kept_words)), dtype=torch.long, device=device)

            out = model(
                input_ids=input_ids, attention_mask=attention_mask,
                word_starts=ws, word_ends=we, word_mask=wm,
                return_dict=True,
            )

            for f in MORPH_FEATURES:
                logits = out[f"{f}_logits"][0]  # (W, K)
                probs = torch.softmax(logits, dim=-1)
                conf, idx = probs.max(dim=-1)
                pred_ids = idx.tolist()
                pred_confs = conf.tolist()
                for w_obj, p_id, p_conf in zip(kept_words, pred_ids, pred_confs):
                    gold_str = getattr(w_obj, f) or "und"
                    gold_id = feat_to_id[f].get(gold_str)
                    if gold_id is None:
                        continue
                    pred_str = feat_id_to_str[f].get(p_id, "und")
                    n_total[f] += 1
                    correct = (p_id == gold_id)
                    if correct:
                        n_correct[f] += 1
                        conf_correct[f].append(p_conf)
                    else:
                        conf_wrong[f].append(p_conf)
                    confusion[f][gold_str][pred_str] += 1
                    pos = w_obj.pos or "und"
                    per_pos_total[f][pos] += 1
                    if correct:
                        per_pos_correct[f][pos] += 1
            n_words += len(kept_words)

    # Aggregate
    summary = {
        "n_sentences": n_sent,
        "n_words": n_words,
        "per_feature_accuracy": {},
        "per_feature_calibration": {},
        "per_pos_accuracy": {},
    }
    for f in MORPH_FEATURES:
        n_t = n_total[f]
        if n_t == 0:
            continue
        acc = n_correct[f] / n_t
        summary["per_feature_accuracy"][f] = {
            "accuracy": acc, "n": n_t, "n_correct": n_correct[f],
        }
        cc = conf_correct[f] or [0.0]
        cw = conf_wrong[f] or [0.0]
        summary["per_feature_calibration"][f] = {
            "mean_conf_correct": float(np.mean(cc)),
            "mean_conf_wrong": float(np.mean(cw)),
            "calibration_gap": float(np.mean(cc) - np.mean(cw)),
        }
        per_pos = {}
        for pos in per_pos_total[f]:
            t = per_pos_total[f][pos]
            c = per_pos_correct[f][pos]
            per_pos[pos] = {"accuracy": c / t if t else 0.0, "n": t}
        summary["per_pos_accuracy"][f] = per_pos

    # Macro
    accs = [summary["per_feature_accuracy"][f]["accuracy"]
            for f in MORPH_FEATURES if f in summary["per_feature_accuracy"]]
    summary["macro_accuracy"] = float(np.mean(accs)) if accs else 0.0

    # Write outputs
    (out_dir / "morphology_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2)
    )
    for f in MORPH_FEATURES:
        rows = []
        labels = sorted(set(list(confusion[f].keys()) +
                            [k for v in confusion[f].values() for k in v.keys()]))
        rows.append(["gold\\pred"] + labels)
        for gold in labels:
            row = [gold]
            for pred in labels:
                row.append(str(confusion[f][gold].get(pred, 0)))
            rows.append(row)
        with (out_dir / f"confusion_{f}.csv").open("w") as fh:
            for r in rows:
                fh.write(",".join(r) + "\n")

    # Pretty print
    print("\n=== Phase 1 morphology evaluation ===")
    print(f"sentences {n_sent}, words {n_words}")
    print(f"{'feature':<12} {'accuracy':>10} {'n':>8} {'cal-gap':>10}")
    print("-" * 48)
    for f in MORPH_FEATURES:
        if f not in summary["per_feature_accuracy"]:
            continue
        a = summary["per_feature_accuracy"][f]
        c = summary["per_feature_calibration"][f]
        print(f"{f:<12} {a['accuracy']*100:>9.2f}% {a['n']:>8} {c['calibration_gap']:>10.3f}")
    print(f"{'macro':<12} {summary['macro_accuracy']*100:>9.2f}%")
    print(f"\nResults written to {out_dir}")


if __name__ == "__main__":
    main()
