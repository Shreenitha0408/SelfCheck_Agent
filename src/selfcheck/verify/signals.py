"""Verification signals for one agent step.

1. NLI evidence check (local)            -> does any retrieved passage entail the step answer?
2. LLM critic (qwen3.8-27b, other family) -> sees the draft answer + passages, judges support.
3. Independent pass (gpt-oss-120b)        -> answers the sub-question WITHOUT seeing the draft;
                                             agreement with the draft is a self-consistency signal.
4. Free features                          -> retrieval scores, answer string present in passages, etc.

The combiner (``confidence.py``) maps these to one calibrated probability that the step is correct.
"""

from __future__ import annotations

from typing import Any

from ..eval.metrics import is_abstention, normalize, token_f1
from ..llm import LLMClient, parse_json
from .nli import NLIChecker, nli_features

CRITIC_MODEL = "qwen/qwen3.8-27b"
INDEPENDENT_MODEL = "openai/gpt-oss-120b"

CRITIC_PROMPT = """You are a strict fact-checker. An assistant answered a sub-question using retrieved passages.
Check the DRAFT ANSWER only against the passages (not your own knowledge).

Sub-question: {sub_question}
Draft answer: {answer}

Passages:
{passages}

Reply with JSON only:
{{"verdict": "supported" | "partially_supported" | "unsupported" | "contradicted",
  "confidence": <0-100, how sure you are the draft answer is correct>,
  "issue": "<one short sentence, or empty>",
  "better_query": "<a better search query if the passages look irrelevant, else empty>"}}"""

INDEPENDENT_PROMPT = """Answer the question using ONLY the passages. Reply with a short phrase.
If the passages do not contain the answer, reply NOT FOUND.

Question: {sub_question}

Passages:
{passages}

Reply with JSON only: {{"answer": "..."}}"""

VERDICT_SCORE = {"supported": 1.0, "partially_supported": 0.5, "unsupported": 0.0, "contradicted": -1.0}


def _passages(texts: list[tuple[str, str]], max_chars: int = 1200) -> str:
    return "\n\n".join(f"[{i + 1}] {t}\n{x[:max_chars]}" for i, (t, x) in enumerate(texts))


class StepVerifier:
    def __init__(self, nli: NLIChecker | None = None, critic: LLMClient | None = None,
                 independent: LLMClient | None = None):
        self.nli = nli or NLIChecker()
        self.critic = critic or LLMClient(CRITIC_MODEL)
        self.independent = independent or LLMClient(INDEPENDENT_MODEL)

    def signals(self, sub_question: str, answer: str, found: bool, passages: list[dict[str, Any]],
                cited_chunk_ids: list[str], use_llm: bool = True, claim: str = "") -> dict[str, Any]:
        texts = [(p["title"], p["text"]) for p in passages]
        cited_idx = [i for i, p in enumerate(passages) if p["chunk_id"] in set(cited_chunk_ids)]
        ans = answer if found else ""
        f: dict[str, Any] = {
            "found": float(found and not is_abstention(answer)),
            "n_cited": float(len(cited_idx)),
            "top_retrieval_score": float(passages[0]["score"]) if passages else 0.0,
            "answer_in_passages": float(bool(normalize(ans)) and any(
                normalize(ans) in normalize(f"{t} {x}") for t, x in texts)),
            "answer_len": float(len(ans.split())),
        }
        f.update(nli_features(self.nli, sub_question, ans, [f"{t}. {x}" for t, x in texts], cited_idx, claim))
        if not use_llm:
            return f

        c = parse_json(self.critic.chat([{"role": "user", "content": CRITIC_PROMPT.format(
            sub_question=sub_question, answer=ans or "NOT FOUND", passages=_passages(texts))}],
            max_tokens=300, json_mode=True).content)
        verdict = str(c.get("verdict", "")).lower()
        try:
            conf = float(c.get("confidence", 50)) / 100
        except (TypeError, ValueError):
            conf = 0.5
        f.update({"critic_verdict": verdict, "critic_score": VERDICT_SCORE.get(verdict, 0.0),
                  "critic_conf": max(0.0, min(1.0, conf)), "critic_issue": c.get("issue", ""),
                  "critic_better_query": c.get("better_query", "")})

        ind = parse_json(self.independent.chat([{"role": "user", "content": INDEPENDENT_PROMPT.format(
            sub_question=sub_question, passages=_passages(texts))}], max_tokens=400, json_mode=True).content)
        ind_ans = str(ind.get("answer", "")).strip()
        both_nf = (not ans) and is_abstention(ind_ans.replace("NOT FOUND", ""))
        f.update({"independent_answer": ind_ans,
                  "agree_f1": 1.0 if both_nf else token_f1(ind_ans, [ans]) if ans else 0.0})
        return f


NUMERIC_FEATURES = ["found", "n_cited", "top_retrieval_score", "answer_in_passages", "answer_len",
                    "nli_entail_max", "nli_entail_cited", "nli_contra_cited",
                    "critic_score", "critic_conf", "agree_f1"]

FEATURE_GROUPS = {
    "nli": ["nli_entail_max", "nli_entail_cited", "nli_contra_cited"],
    "critic": ["critic_score", "critic_conf"],
    "self_consistency": ["agree_f1"],
    "cheap": ["found", "n_cited", "top_retrieval_score", "answer_in_passages", "answer_len"],
}
