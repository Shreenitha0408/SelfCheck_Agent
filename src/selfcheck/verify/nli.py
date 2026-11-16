"""Local NLI evidence checker (cross-encoder/nli-deberta-v3-small, ONNX, CPU).

Turns (sub-question, answer) into a hypothesis and measures whether any
retrieved passage entails or contradicts it. No API calls.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

NLI_MODEL = "cross-encoder/nli-deberta-v3-base"  # small was unreliable on real passages


class NLIChecker:
    def __init__(self, model_id: str = NLI_MODEL, max_len: int = 384):
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer

        cfg = json.loads(Path(hf_hub_download(model_id, "config.json")).read_text())
        self.labels = {int(k): v.lower() for k, v in cfg["id2label"].items()}
        self.tok = Tokenizer.from_file(hf_hub_download(model_id, "tokenizer.json"))
        self.tok.enable_truncation(max_length=max_len, strategy="only_first")
        self.tok.enable_padding()
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4
        self.sess = ort.InferenceSession(hf_hub_download(model_id, "onnx/model.onnx"), opts,
                                         providers=["CPUExecutionProvider"])
        self.inputs = {i.name for i in self.sess.get_inputs()}

    def probs(self, premises: list[str], hypothesis: str) -> list[dict[str, float]]:
        if not premises:
            return []
        enc = self.tok.encode_batch([(p, hypothesis) for p in premises])
        feed = {"input_ids": np.array([e.ids for e in enc], dtype=np.int64),
                "attention_mask": np.array([e.attention_mask for e in enc], dtype=np.int64)}
        if "token_type_ids" in self.inputs:
            feed["token_type_ids"] = np.array([e.type_ids for e in enc], dtype=np.int64)
        logits = self.sess.run(None, feed)[0]
        p = np.exp(logits - logits.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        return [{self.labels[j]: float(row[j]) for j in range(row.shape[0])} for row in p]


def hypothesis(sub_question: str, answer: str) -> str:
    # Declarative template; tested to separate right/wrong answers far better than "Q? A."
    return f"The answer to the question '{sub_question.strip()}' is {answer.strip().rstrip('.')}."


def nli_features(checker: NLIChecker, sub_question: str, answer: str, passages: list[str],
                 cited: list[int] | None = None, claim: str = "") -> dict[str, float]:
    """Entailment over all passages; entailment/contradiction over the passages the agent cited
    (contradiction from unrelated passages is noise, so it is only measured on cited ones)."""
    if not answer.strip():
        return {"nli_entail_max": 0.0, "nli_entail_cited": 0.0, "nli_contra_cited": 0.0}
    # Prefer the agent's own declarative claim (NLI models need a plain statement);
    # fall back to a template when the claim is missing or does not mention the answer.
    use_claim = claim and any(w in claim.lower() for w in answer.lower().split()[:3])
    ps = checker.probs(passages, claim if use_claim else hypothesis(sub_question, answer))
    ent = [p.get("entailment", 0.0) for p in ps] or [0.0]
    idx = [i for i in (cited or []) if i < len(ps)] or [0]
    return {"nli_entail_max": max(ent),
            "nli_entail_cited": max(ent[i] for i in idx),
            "nli_contra_cited": max(ps[i].get("contradiction", 0.0) for i in idx) if ps else 0.0}
