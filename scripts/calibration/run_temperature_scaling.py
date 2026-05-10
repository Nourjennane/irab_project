"""Post-hoc temperature scaling on the production checkpoint.

Splits MASAQ into a small calibration shard (last 100 sentences by id)
and a held-out reporting shard (first 524). Fits one scalar T per
axis (case / role / marker) on the calibration shard via L-BFGS over
the negative log-likelihood. Applies T at re-eval and measures ECE
before/after.

Output:
  docs/calibration/
    temperature_fits.json         { case: T, role: T, marker: T }
    calibration_report.md         before/after ECE + reliability bins
    reliability_bins.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def _load_model(ckpt_dir: Path, encoder_name: str, device):
    import torch
    from irab_tashkeel.morphology.dep_aware_model import DepAwareStructuredModel

    sd = torch.load(ckpt_dir / "pytorch_model.bin", map_location="cpu",
                    weights_only=True)
    model = DepAwareStructuredModel(
        encoder_name=encoder_name,
        enable_morph_heads=True, morph_heads_enabled=None,
        enable_dep_features=True,
    )
    model.load_state_dict(sd, strict=False)
    model.to(device); model.eval()
    return model


def _collect_logits_and_labels(model, tokenizer, sentences, batch_size, device):
    """Run the model over sentences and return per-axis (logits, labels) tensors."""
    import torch
    from irab_tashkeel.training_v2.collator import (
        SchemaV2Collator, CollatorConfig,
    )
    from irab_tashkeel.training_v2.dataset import SchemaV2Dataset

    coll = SchemaV2Collator(tokenizer, config=CollatorConfig(
        pad_token_id=tokenizer.pad_token_id or 0,
    ))
    ds = SchemaV2Dataset(sentences)
    case_logits, role_logits, marker_logits = [], [], []
    case_labels, role_labels, marker_labels = [], [], []

    with torch.no_grad():
        for start in range(0, len(ds), batch_size):
            items = [ds[i] for i in range(start, min(start + batch_size, len(ds)))]
            b = coll(items)
            b = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in b.items()}
            out = model(
                input_ids=b["input_ids"], attention_mask=b["attention_mask"],
                word_starts=b["word_starts"], word_ends=b["word_ends"],
                word_mask=b["word_mask"], return_dict=True,
            )
            mask = b["word_mask"].bool()
            case_logits.append(out["case_logits"][mask].cpu())
            role_logits.append(out["role_logits"][mask].cpu())
            marker_logits.append(out["marker_logits"][mask].cpu())
            case_labels.append(b["case_labels"][mask].cpu())
            role_labels.append(b["role_labels"][mask].cpu())
            marker_labels.append(b["marker_labels"][mask].cpu())

    return {
        "case":   (torch.cat(case_logits),   torch.cat(case_labels)),
        "role":   (torch.cat(role_logits),   torch.cat(role_labels)),
        "marker": (torch.cat(marker_logits), torch.cat(marker_labels)),
    }


def _ece(logits, labels, T: float = 1.0, n_bins: int = 10) -> float:
    import torch
    import torch.nn.functional as F
    valid = labels != -100
    if valid.sum() == 0:
        return 0.0
    log = (logits[valid] / T)
    lab = labels[valid]
    p = F.softmax(log, dim=-1)
    conf, pred = p.max(dim=-1)
    correct = (pred == lab).float()

    ece = 0.0
    n = len(conf)
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        mask = (conf >= lo) & (conf < (hi + (1e-6 if b == n_bins - 1 else 0)))
        if mask.sum() == 0:
            continue
        bin_acc  = correct[mask].mean().item()
        bin_conf = conf[mask].mean().item()
        ece += mask.sum().item() * abs(bin_acc - bin_conf)
    return round(ece / n, 4)


def _reliability_bins(logits, labels, T: float, n_bins: int = 10) -> List[Dict]:
    import torch
    import torch.nn.functional as F
    valid = labels != -100
    log = (logits[valid] / T)
    lab = labels[valid]
    p = F.softmax(log, dim=-1)
    conf, pred = p.max(dim=-1)
    correct = (pred == lab).float()
    out = []
    n = len(conf)
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        mask = (conf >= lo) & (conf < (hi + (1e-6 if b == n_bins - 1 else 0)))
        nb = int(mask.sum().item())
        out.append({
            "bin":      [round(lo, 2), round(hi, 2)],
            "n":        nb,
            "accuracy": round(correct[mask].mean().item(), 4) if nb else 0.0,
            "mean_conf": round(conf[mask].mean().item(), 4) if nb else 0.0,
        })
    return out


def fit_temperature(logits, labels, *, n_iter: int = 100) -> float:
    import torch
    import torch.nn.functional as F
    valid = labels != -100
    if valid.sum() == 0:
        return 1.0
    log_v = logits[valid]
    lab_v = labels[valid]
    log_T = torch.zeros((), requires_grad=True)
    optimizer = torch.optim.LBFGS([log_T], lr=0.1, max_iter=n_iter)
    def closure():
        optimizer.zero_grad()
        T = torch.exp(log_T).clamp(min=0.05, max=10.0)
        loss = F.cross_entropy(log_v / T, lab_v, ignore_index=-100)
        loss.backward()
        return loss
    optimizer.step(closure)
    return round(float(torch.exp(log_T.detach()).item()), 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(ROOT / "runs" / "validated_nextgen_recovery"))
    ap.add_argument("--source",     default="masaq_quranic",
                    help="held-out source to split for calibration")
    ap.add_argument("--cal_n",      type=int, default=100)
    ap.add_argument("--out_dir",    default=str(ROOT / "docs" / "calibration"))
    ap.add_argument("--encoder_name", default="UBC-NLP/AraT5v2-base-1024")
    ap.add_argument("--batch_size", type=int, default=16)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import os
    os.environ.setdefault("USE_TF", "NO")
    import torch
    from transformers import AutoTokenizer
    from irab_tashkeel.data_v2.schema_v2 import read_jsonl

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint
        if (Path(args.checkpoint) / "tokenizer.json").exists()
        else args.encoder_name
    )
    model = _load_model(Path(args.checkpoint), args.encoder_name, device)

    # Split source by sentence_id (deterministic)
    source_path = ROOT / "data_v2" / "annotated" / args.source / "all.jsonl"
    sentences = list(read_jsonl(str(source_path)))
    sentences.sort(key=lambda s: s.sentence_id)
    cal_sents  = sentences[-args.cal_n:]
    test_sents = sentences[:-args.cal_n]
    print(f"calibration shard: {len(cal_sents)} sentences "
          f"(last {args.cal_n} by sentence_id)")
    print(f"reporting shard:   {len(test_sents)} sentences")

    print("\nCollecting logits on calibration shard...")
    cal = _collect_logits_and_labels(model, tokenizer, cal_sents,
                                      args.batch_size, device)
    print("Collecting logits on reporting shard...")
    rep = _collect_logits_and_labels(model, tokenizer, test_sents,
                                      args.batch_size, device)

    # ===== Fit T per axis =====
    fits: Dict[str, float] = {}
    for axis in ("case", "role", "marker"):
        logits, labels = cal[axis]
        T = fit_temperature(logits, labels)
        fits[axis] = T
        print(f"  {axis}: fitted T = {T}")

    # ===== Report ECE + reliability before/after =====
    report: Dict[str, Any] = {"fits": fits, "shards": {
        "calibration_n": len(cal_sents),
        "reporting_n":   len(test_sents),
    }}
    rel: Dict[str, Any] = {}
    for axis in ("case", "role", "marker"):
        logits, labels = rep[axis]
        ece_before = _ece(logits, labels, T=1.0)
        ece_after  = _ece(logits, labels, T=fits[axis])
        rel[axis] = {
            "ece_before": ece_before,
            "ece_after":  ece_after,
            "delta":      round(ece_after - ece_before, 4),
            "T":          fits[axis],
            "before_bins": _reliability_bins(logits, labels, T=1.0),
            "after_bins":  _reliability_bins(logits, labels, T=fits[axis]),
        }
        print(f"  {axis}: ECE {ece_before} → {ece_after} "
              f"(Δ {ece_after - ece_before:+.4f})")

    report["reliability"] = rel

    (out_dir / "temperature_fits.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False)
    )

    # Markdown
    lines = ["# Post-Hoc Temperature Scaling — Validated Recovery", "",
             f"**Calibration shard:** {len(cal_sents)} sentences "
             f"(last by sentence_id, from `{args.source}`)",
             f"**Reporting shard:** {len(test_sents)} sentences",
             ""]
    lines.append("## Fitted temperatures")
    lines.append("")
    lines.append("| axis | T |")
    lines.append("|---|---:|")
    for axis, T in fits.items():
        lines.append(f"| {axis} | {T} |")
    lines.append("")
    lines.append("## ECE before vs after")
    lines.append("")
    lines.append("| axis | ECE before | ECE after | Δ |")
    lines.append("|---|---:|---:|---:|")
    for axis, info in rel.items():
        lines.append(f"| {axis} | {info['ece_before']} | {info['ece_after']} "
                      f"| {info['delta']:+.4f} |")
    lines.append("")
    lines.append("## Reliability bins (role, before scaling)")
    lines.append("")
    lines.append("| bin | n | accuracy | mean conf |")
    lines.append("|---|---:|---:|---:|")
    for b in rel["role"]["before_bins"]:
        lines.append(f"| [{b['bin'][0]}, {b['bin'][1]}) | {b['n']} | "
                      f"{b['accuracy']} | {b['mean_conf']} |")
    lines.append("")
    lines.append("## Reliability bins (role, after scaling at T=" +
                 f"{rel['role']['T']})")
    lines.append("")
    lines.append("| bin | n | accuracy | mean conf |")
    lines.append("|---|---:|---:|---:|")
    for b in rel["role"]["after_bins"]:
        lines.append(f"| [{b['bin'][0]}, {b['bin'][1]}) | {b['n']} | "
                      f"{b['accuracy']} | {b['mean_conf']} |")
    (out_dir / "calibration_report.md").write_text("\n".join(lines))

    print(f"\n✓ wrote {out_dir / 'calibration_report.md'}")


if __name__ == "__main__":
    main()
