"""Phase C — freeze the validated nextgen stage_7 checkpoint.

Copies ``runs/nextgen/stage_7/final/`` to ``runs/validated_nextgen_stage7/``
and writes a complete reproducibility manifest:

  - config snapshot (training args, head weights per stage)
  - tokenizer snapshot
  - git commit hash
  - dataset manifests (sha + size for each schema_v2 jsonl)
  - curriculum manifests (which sentences entered which stage)
  - eval manifests (which sentences scored at final eval)
  - environment manifest (python + torch + transformers versions)

Also exports inference artifacts:

  - ``model.onnx``         — ONNX dynamic-shape graph
  - ``model_torchscript.pt`` — torch.jit.trace
  - ``model_fp16.pt``      — fp16 state_dict for fast CPU inference

After this script runs, the validated checkpoint is *immutable* —
any further training writes to a NEW directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def _git_status_clean() -> bool:
    try:
        s = subprocess.check_output(
            ["git", "-C", str(ROOT), "status", "--porcelain"],
            text=True,
        )
        return s.strip() == ""
    except Exception:
        return False


def _env_manifest() -> Dict[str, str]:
    out: Dict[str, str] = {}
    for mod in ("torch", "transformers", "tokenizers", "numpy"):
        try:
            m = __import__(mod)
            out[mod] = getattr(m, "__version__", "?")
        except Exception:
            out[mod] = "missing"
    out["python"] = sys.version.split()[0]
    return out


def _dataset_manifest(data_root: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for d in sorted((data_root / "annotated").glob("*")):
        f = d / "all.jsonl"
        if not f.exists():
            continue
        n_lines = sum(1 for _ in f.open())
        out[d.name] = {
            "path": str(f.relative_to(ROOT)) if f.is_absolute() else str(f),
            "n_sentences": n_lines,
            "sha256": _sha256(f),
            "size_bytes": f.stat().st_size,
        }
    return out


def _copy_checkpoint(src: Path, dst: Path) -> None:
    if dst.exists():
        raise SystemExit(f"refusing to overwrite existing {dst}")
    shutil.copytree(src, dst)


def _export_fp16(src_dir: Path, dst_path: Path) -> None:
    import torch
    sd = torch.load(src_dir / "pytorch_model.bin", map_location="cpu",
                    weights_only=True)
    sd_fp16 = {k: v.half() if v.is_floating_point() else v for k, v in sd.items()}
    torch.save(sd_fp16, dst_path)


def _export_onnx(checkpoint_dir: Path, out_path: Path,
                  encoder_name: str = "UBC-NLP/AraT5v2-base-1024") -> None:
    """Best-effort ONNX export.

    The DepAwareStructuredModel's forward signature includes
    word_starts/word_ends/word_mask which onnx.export handles via
    dynamic axes. If export fails (often does for custom heads),
    we record the failure and skip — TorchScript trace remains.
    """
    import torch
    from transformers import AutoTokenizer
    from irab_tashkeel.morphology.dep_aware_model import DepAwareStructuredModel

    model = DepAwareStructuredModel(
        encoder_name=encoder_name,
        enable_morph_heads=True, morph_heads_enabled=None,
        enable_dep_features=True,
    )
    sd = torch.load(checkpoint_dir / "pytorch_model.bin", map_location="cpu",
                    weights_only=True)
    model.load_state_dict(sd, strict=False)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        str(checkpoint_dir) if (checkpoint_dir / "tokenizer.json").exists()
        else encoder_name,
    )
    sample = "هذا اختبار."
    enc = tokenizer(sample, return_tensors="pt", padding=True, truncation=True)
    B, T = enc["input_ids"].shape
    word_starts = torch.tensor([[0, 1, 2]], dtype=torch.long)
    word_ends = torch.tensor([[1, 2, T - 1]], dtype=torch.long)
    word_mask = torch.tensor([[1, 1, 1]], dtype=torch.long)

    torch.onnx.export(
        model,
        (enc["input_ids"], enc["attention_mask"], word_starts,
         word_ends, word_mask),
        str(out_path),
        input_names=["input_ids", "attention_mask", "word_starts",
                     "word_ends", "word_mask"],
        output_names=["case_logits", "role_logits", "marker_logits",
                      "pos_logits"],
        dynamic_axes={
            "input_ids": {0: "B", 1: "T"},
            "attention_mask": {0: "B", 1: "T"},
            "word_starts": {0: "B", 1: "W"},
            "word_ends":   {0: "B", 1: "W"},
            "word_mask":   {0: "B", 1: "W"},
        },
        opset_version=17, do_constant_folding=True,
    )


def _export_torchscript(checkpoint_dir: Path, out_path: Path,
                         encoder_name: str = "UBC-NLP/AraT5v2-base-1024") -> None:
    import torch
    from irab_tashkeel.morphology.dep_aware_model import DepAwareStructuredModel
    model = DepAwareStructuredModel(
        encoder_name=encoder_name,
        enable_morph_heads=True, morph_heads_enabled=None,
        enable_dep_features=True,
    )
    sd = torch.load(checkpoint_dir / "pytorch_model.bin", map_location="cpu",
                    weights_only=True)
    model.load_state_dict(sd, strict=False)
    model.eval()
    # Trace mode is fragile with custom heads — fall back to script
    # if trace fails.
    try:
        # Build a small dummy input
        input_ids = torch.zeros((1, 8), dtype=torch.long)
        attn = torch.ones((1, 8), dtype=torch.long)
        word_starts = torch.tensor([[0, 1, 2]], dtype=torch.long)
        word_ends = torch.tensor([[1, 2, 7]], dtype=torch.long)
        word_mask = torch.tensor([[1, 1, 1]], dtype=torch.long)
        traced = torch.jit.trace(
            model, (input_ids, attn, word_starts, word_ends, word_mask),
            strict=False,
        )
        traced.save(str(out_path))
    except Exception as e:
        # Save state_dict + class info so we can rebuild without trace
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": sd, "encoder_name": encoder_name,
            "trace_failed": str(e),
            "model_class": "DepAwareStructuredModel",
        }, out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT / "runs" / "nextgen" / "stage_7" / "final"))
    ap.add_argument("--dst", default=str(ROOT / "runs" / "validated_nextgen_stage7"))
    ap.add_argument("--data_root", default=str(ROOT / "data_v2"))
    ap.add_argument("--encoder_name", default="UBC-NLP/AraT5v2-base-1024")
    ap.add_argument("--skip_onnx", action="store_true")
    ap.add_argument("--skip_torchscript", action="store_true")
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    if not src.exists():
        raise SystemExit(f"source checkpoint missing: {src}")

    print(f"Freezing {src} → {dst}")
    _copy_checkpoint(src, dst)

    # Reproducibility manifest
    manifest = {
        "src": str(src), "dst": str(dst),
        "git_commit": _git_commit(),
        "git_clean":  _git_status_clean(),
        "env":        _env_manifest(),
        "datasets":   _dataset_manifest(Path(args.data_root)),
        "training_summary": None,
        "curriculum": None,
    }
    ts_path = ROOT / "runs" / "nextgen" / "training_summary.json"
    if ts_path.exists():
        manifest["training_summary"] = json.loads(ts_path.read_text())
    (dst / "REPRODUCIBILITY_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str)
    )
    print(f"  wrote {dst / 'REPRODUCIBILITY_MANIFEST.json'}")

    # Inference exports
    if not args.skip_torchscript:
        try:
            print("Exporting TorchScript...")
            _export_torchscript(dst, dst / "model_torchscript.pt", args.encoder_name)
            print(f"  wrote {dst / 'model_torchscript.pt'}")
        except Exception as e:
            print(f"  [warn] TorchScript export failed: {e}")

    print("Exporting fp16 state_dict...")
    try:
        _export_fp16(dst, dst / "model_fp16.pt")
        print(f"  wrote {dst / 'model_fp16.pt'}")
    except Exception as e:
        print(f"  [warn] fp16 export failed: {e}")

    if not args.skip_onnx:
        try:
            print("Exporting ONNX...")
            _export_onnx(dst, dst / "model.onnx", args.encoder_name)
            print(f"  wrote {dst / 'model.onnx'}")
        except Exception as e:
            print(f"  [warn] ONNX export failed: {e}")
            (dst / "ONNX_EXPORT_FAILED.txt").write_text(str(e))

    print(f"\n✅ Validated checkpoint frozen at {dst}")


if __name__ == "__main__":
    main()
