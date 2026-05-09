"""Phase A — independent full-evaluation runner.

Runs a frozen checkpoint over the *full* schema_v2 test sets
(no caps, no curriculum sampling, no train-time augmentations,
deterministic seed). Emits raw per-(checkpoint × dataset) JSON
files into ``--output_root``; ``aggregate_full_eval.py`` then
folds them into the final report.

Design intent
-------------

This is **not** ``gate_metrics_for_stage``. The training-loop
metric path is shared (``predict_for_eval``) but every metric
returned here is recomputed from scratch via the eval_v2 module
so Phase 3-A and stage_7 are scored by **bit-identical** code.

Output JSON shape (per row)::

    {
      "checkpoint":  "phase3a",
      "dataset":     "gazelle_test",
      "n_sentences": 30,
      "n_tokens":    241,
      "metrics_full":      { case_acc, role_acc, marker_em, fully, calib_gap, ... },
      "metrics_observable": { ... fully_observable_only=True ... },
      "calibration":        { case: {...}, role: {...}, marker: {...} },
      "constructions":      { kana_sisters: {p,r,f1,n}, ... },
      "ambiguity":          { ... },
      "stratified":         { domain: {...}, families: {...} }
    }
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def _set_seeds(seed: int) -> None:
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def _parse_kv_list(items: List[str]) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for it in items:
        name, _, path = it.partition(":")
        if not name or not path:
            raise SystemExit(f"bad NAME:PATH spec: {it!r}")
        out.append((name, path))
    return out


def _load_model(ckpt_dir: Path, encoder_name: str, device):
    import torch
    from irab_tashkeel.morphology.dep_aware_model import DepAwareStructuredModel

    sd = torch.load(ckpt_dir / "pytorch_model.bin", map_location="cpu",
                    weights_only=True)
    # Auto-detect whether the checkpoint was trained with the graph refiner:
    # if any graph_refiner.* keys exist, enable it on construction so
    # state_dict load matches.
    has_graph = any(k.startswith("graph_refiner.") or k == "graph_gate"
                    for k in sd)
    model = DepAwareStructuredModel(
        encoder_name=encoder_name,
        enable_morph_heads=True, morph_heads_enabled=None,
        enable_dep_features=True,
        enable_graph_refiner=has_graph,
    )
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"  [warn] missing keys: {len(missing)} (first 3: {missing[:3]})")
    if unexpected:
        print(f"  [warn] unexpected keys: {len(unexpected)} (first 3: {unexpected[:3]})")
    model.to(device)
    model.eval()
    return model


def _serialise_construction_metrics(cd: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for fam, m in cd.items():
        if hasattr(m, "__dict__"):
            d = dict(m.__dict__)
        elif hasattr(m, "_asdict"):
            d = dict(m._asdict())
        else:
            d = m if isinstance(m, dict) else {"value": m}
        out[fam] = d
    return out


def _evaluate_one(
    *, checkpoint_name: str, checkpoint_dir: Path,
    dataset_name: str, dataset_path: Path,
    encoder_name: str, batch_size: int, device,
) -> Dict[str, Any]:
    import torch
    from transformers import AutoTokenizer
    from irab_tashkeel.data_v2.schema_v2 import read_jsonl
    from irab_tashkeel.training_v2.eval_hook import predict_for_eval
    from irab_tashkeel.eval_v2 import (
        aggregate_outcomes, ambiguity_robustness, calibration_for_all_fields,
        construction_detection_metrics, extract_outcomes,
        stratified_metrics,
    )

    print(f"\n=== {checkpoint_name} × {dataset_name} ===")
    sentences = list(read_jsonl(str(dataset_path)))
    n_tokens = sum(s.n_tokens for s in sentences)
    print(f"  loaded {len(sentences)} sentences ({n_tokens} tokens)")

    tokenizer = AutoTokenizer.from_pretrained(
        str(checkpoint_dir) if (checkpoint_dir / "tokenizer.json").exists()
        else encoder_name,
    )
    model = _load_model(checkpoint_dir, encoder_name, device)

    t0 = time.time()
    preds = predict_for_eval(model, tokenizer, sentences,
                             batch_size=batch_size, device=device)
    print(f"  forward pass: {time.time() - t0:.1f}s")

    outcomes = extract_outcomes(sentences, preds)
    full = aggregate_outcomes(outcomes)
    obs  = aggregate_outcomes([o for o in outcomes if o.is_fully_observable])
    calib = calibration_for_all_fields(outcomes)
    constr = construction_detection_metrics(sentences, preds)
    try:
        ambi = ambiguity_robustness(sentences, preds)
        ambi_d = ambi.__dict__ if hasattr(ambi, "__dict__") else {}
    except Exception as e:
        ambi_d = {"error": str(e)}

    try:
        strat_domain = stratified_metrics(outcomes, axes=["domain"])
        strat_family = stratified_metrics(outcomes, axes=["construction_families"])
    except Exception as e:
        strat_domain = {"error": str(e)}
        strat_family = {"error": str(e)}

    # Free GPU
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "checkpoint": checkpoint_name,
        "checkpoint_dir": str(checkpoint_dir),
        "dataset": dataset_name,
        "dataset_path": str(dataset_path),
        "n_sentences": len(sentences),
        "n_tokens": n_tokens,
        "metrics_full": full,
        "metrics_observable": obs,
        "calibration": _serialise_calib(calib),
        "constructions": _serialise_construction_metrics(constr),
        "ambiguity": ambi_d,
        "stratified_domain": _strat_to_dict(strat_domain),
        "stratified_family": _strat_to_dict(strat_family),
    }


def _serialise_calib(calib: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for field, rep in calib.items():
        if hasattr(rep, "__dict__"):
            d = dict(rep.__dict__)
            if "bins" in d and d["bins"] and hasattr(d["bins"][0], "__dict__"):
                d["bins"] = [dict(b.__dict__) for b in d["bins"]]
            out[field] = d
        else:
            out[field] = rep
    return out


def _strat_to_dict(s: Any) -> Any:
    if isinstance(s, dict):
        return {k: (v.__dict__ if hasattr(v, "__dict__") else v)
                for k, v in s.items()}
    if hasattr(s, "__dict__"):
        return s.__dict__
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True,
                    help="NAME:DIR pairs (e.g. phase3a:runs/phase3a_491240/final)")
    ap.add_argument("--datasets", nargs="+", required=True,
                    help="NAME:JSONL pairs (e.g. gazelle:data_v2/annotated/gazelle_test/all.jsonl)")
    ap.add_argument("--output_root", default=str(ROOT / "docs" / "final_eval" / "raw"))
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--encoder_name", default="UBC-NLP/AraT5v2-base-1024")
    args = ap.parse_args()

    _set_seeds(args.seed)
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    ckpts = _parse_kv_list(args.checkpoints)
    dsets = _parse_kv_list(args.datasets)
    print(f"checkpoints: {[c[0] for c in ckpts]}")
    print(f"datasets:    {[d[0] for d in dsets]}")

    for ckpt_name, ckpt_path in ckpts:
        for ds_name, ds_path in dsets:
            out_path = out_root / f"{ckpt_name}__{ds_name}.json"
            if out_path.exists() and os.environ.get("REUSE_EVAL") == "1":
                print(f"[skip] {out_path} exists; REUSE_EVAL=1")
                continue
            try:
                rec = _evaluate_one(
                    checkpoint_name=ckpt_name,
                    checkpoint_dir=Path(ckpt_path),
                    dataset_name=ds_name,
                    dataset_path=Path(ds_path),
                    encoder_name=args.encoder_name,
                    batch_size=args.batch_size,
                    device=device,
                )
            except Exception as e:
                import traceback
                traceback.print_exc()
                rec = {
                    "checkpoint": ckpt_name, "dataset": ds_name,
                    "error": str(e),
                }
            out_path.write_text(json.dumps(rec, indent=2, ensure_ascii=False,
                                           default=str))
            print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
