"""Interactive Gradio demo for the structured i'rāb predictor (Phase 3 / v1 rebuild).

Loads a trained ``StructuredIrabModel`` and surfaces, per word:

* the predicted structured labels (case / role / marker / POS)
* per-head softmax confidence
* which symbolic constraints fired
* the deterministic-template rendered Arabic prose

Run:

    MODEL_DIR=runs/structured_v1_rebuild_<JOBID>/final \\
    python app/structured_demo.py [--share]

Set ``IRAB_RETRIEVE=1`` to include a Jaccard-similar-sentence panel from the
training corpus (lightweight; the FAISS / dense version lives in the journal
fork).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from irab_tashkeel.inference.structured_predictor import (
    StructuredPredictor, StructuredPredictorConfig,
)
from irab_tashkeel.inference.qualitative_trace import render_markdown
from irab_tashkeel.inference.template_renderer import render_word
from irab_tashkeel.retrieval import (
    GrammarMemory, JaccardRetriever, detect_constructions,
)
from irab_tashkeel.structured.word_irab import SentenceIrab


EXAMPLES = [
    ["ذهب الطالب إلى المدرسة"],
    ["إن العلم نور"],
    ["كان الطالب مجتهدا"],
    ["مررت بأخيك زيد"],
    ["لا تكسر الزجاج"],
    ["الكتاب على الطاولة"],
]


def build_predictor(model_dir: str, *, apply_constraints: bool, retriever: JaccardRetriever | None):
    cfg = StructuredPredictorConfig(
        apply_constraints=apply_constraints,
        render_prose=True,
        device="auto",
    )
    return StructuredPredictor(model_dir, cfg=cfg, retriever=retriever)


def predict_handler(predictor_no_c, predictor_c, jaccard_retriever, grammar_memory):
    def _run(sentence: str, use_constraints: bool, show_retrieved: bool,
             show_grammar_memory: bool) -> tuple[str, str, str]:
        sentence = (sentence or "").strip()
        if not sentence:
            return "_(empty input)_", "", ""

        pred = predictor_c if use_constraints else predictor_no_c
        result: SentenceIrab = pred.predict_sentence(sentence)
        if not result.items:
            return "_(no tokens)_", "", ""

        # Markdown table
        table_md = render_markdown(result, show_confidence=True, show_constraints=True,
                                   title=f"Structured prediction ({'+ constraints' if use_constraints else 'heads only'})")

        # Aggregate per-line prose
        prose_lines = []
        for w in result.items:
            prose = w.irab_prose or render_word(w)
            prose_lines.append(f"**{w.word}**: {prose}")
        prose_md = "\n\n".join(prose_lines)

        # Retrieval panel
        retr_lines = []
        # construction tags for the query
        q_tags = detect_constructions(sentence)
        if q_tags:
            retr_lines.append(f"**Detected constructions:** `{', '.join(sorted(q_tags))}`")

        if show_grammar_memory and grammar_memory is not None:
            hits = grammar_memory.retrieve(sentence, k=4, prefer_shared_constructions=True)
            if hits:
                retr_lines.append("\n### Similar Quranic constructions (grammar memory)")
                for h in hits:
                    tag_str = ", ".join(sorted(h.constructions)) or "—"
                    sv = f" ({h.sura_verse})" if h.sura_verse else ""
                    retr_lines.append(f"- score `{h.score:.3f}` · tags `{tag_str}`{sv} — {h.sentence}")
        if show_retrieved and jaccard_retriever is not None:
            hits = jaccard_retriever.get_top_k(sentence, k=4)
            if hits:
                retr_lines.append("\n### Similar parsed sentences (training-corpus retrieval)")
                for h in hits:
                    retr_lines.append(f"- score `{h.score:.3f}` — {h.sentence}")
        return table_md, prose_md, "\n".join(retr_lines) if retr_lines else ""
    return _run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("MODEL_DIR",
                    "runs/structured_v1_rebuild_490894/final"))
    ap.add_argument("--retrieval_corpus", default="data/structured_v1/train.jsonl")
    ap.add_argument("--grammar_memory", default="data/masaq_eval.jsonl",
                    help="Quranic grammar-memory source (MASAQ jsonl)")
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--port", type=int, default=int(os.environ.get("GRADIO_PORT", 7860)))
    args = ap.parse_args()

    try:
        import gradio as gr
    except ImportError:
        print("ERROR: gradio not installed. Run: pip install gradio", file=sys.stderr)
        raise

    print(f"loading training-corpus retriever from {args.retrieval_corpus} ...")
    jaccard_retriever = JaccardRetriever.from_jsonl(args.retrieval_corpus)
    print(f"  indexed {len(jaccard_retriever)} sentences")

    grammar_memory = None
    if Path(args.grammar_memory).exists():
        print(f"loading grammar memory from {args.grammar_memory} ...")
        grammar_memory = GrammarMemory.from_masaq(args.grammar_memory)
        print(f"  indexed {len(grammar_memory)} Quranic verses; tag counts: {grammar_memory.stats()['tag_counts']}")
    else:
        print(f"WARNING: grammar memory not found at {args.grammar_memory}; panel disabled")

    print(f"loading predictors from {args.model} ...")
    pred_no_c = build_predictor(args.model, apply_constraints=False, retriever=jaccard_retriever)
    pred_c = build_predictor(args.model, apply_constraints=True, retriever=jaccard_retriever)

    handler = predict_handler(pred_no_c, pred_c, jaccard_retriever, grammar_memory)

    with gr.Blocks(title="Arabic i'rāb — interpretable structured prediction") as demo:
        gr.Markdown(
            "# Arabic *i'rāb* — interpretable structured prediction\n\n"
            "AraT5v2-base encoder + four classification heads (case / role / marker / POS) "
            "+ four soft symbolic-constraint reranking families (preposition→jarr, *inna* sisters, "
            "*kāna* sisters, *iḍāfa*) + deterministic Arabic-prose template renderer "
            "+ optional Quranic grammar-memory retrieval (construction-aware Jaccard over MASAQ). "
            "Type or pick a sentence below.\n"
        )
        with gr.Row():
            with gr.Column(scale=3):
                inp = gr.Textbox(label="Arabic sentence", lines=2,
                                 placeholder="مثال: ذهب الطالب إلى المدرسة")
                with gr.Row():
                    use_c = gr.Checkbox(label="apply 4 symbolic constraints", value=True)
                    show_g = gr.Checkbox(label="similar Quranic constructions",
                                         value=grammar_memory is not None,
                                         interactive=grammar_memory is not None)
                    show_r = gr.Checkbox(label="similar training sentences", value=False)
                run_btn = gr.Button("Predict", variant="primary")
                gr.Examples(examples=EXAMPLES, inputs=[inp])
            with gr.Column(scale=4):
                table_out = gr.Markdown(label="Per-word structured prediction")
                prose_out = gr.Markdown(label="Rendered Arabic i'rāb prose")
                retr_out = gr.Markdown(label="Retrieval / grammar memory")

        run_btn.click(handler, inputs=[inp, use_c, show_r, show_g],
                      outputs=[table_out, prose_out, retr_out])
        inp.submit(handler, inputs=[inp, use_c, show_r, show_g],
                   outputs=[table_out, prose_out, retr_out])

    demo.queue().launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
