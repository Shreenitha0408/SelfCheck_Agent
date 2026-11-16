"""High-level service used by the API and the demo UI: documents + question -> verified answer."""

from __future__ import annotations

import hashlib
import io
import re
from pathlib import Path
from typing import Any

from .agent.agent import AgentConfig, DocQAAgent
from .llm import LLMClient
from .retrieval.index import CorpusIndex
from .verify.confidence import ConfidenceModel
from .verify.nli import NLIChecker
from .verify.policy import PolicyConfig, VerificationPolicy, run_verified
from .verify.signals import CRITIC_MODEL, INDEPENDENT_MODEL, StepVerifier

GENERATOR = "openai/gpt-oss-20b"
ROOT = Path(__file__).resolve().parents[2]


class HeuristicConfidence:
    """Fallback when no trained combiner is available: average of the three main signals."""

    features: list[str] = []

    def score_step(self, sig: dict[str, Any]) -> float:
        critic = sig.get("critic_conf", 0.5) if sig.get("critic_score", 0) > 0 else 1 - sig.get("critic_conf", 0.5)
        return float((critic + sig.get("nli_entail_max", 0.0) + sig.get("agree_f1", 0.0)) / 3)


def split_documents(text: str, source: str = "pasted", max_words: int = 180) -> list[dict[str, str]]:
    """Split raw text into paragraph-sized documents (blank-line separated, long ones re-chunked later)."""
    paras = [re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n\s*\n", text)]
    paras = [p for p in paras if len(p.split()) >= 5]
    docs = []
    for i, p in enumerate(paras):
        h = hashlib.sha1(p.encode()).hexdigest()[:12]
        docs.append({"doc_id": f"{source}:{h}", "title": f"{source} ¶{i + 1}", "text": p})
    return docs


def read_upload(name: str, data: bytes) -> str:
    if name.lower().endswith(".pdf"):
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        return "\n\n".join((p.extract_text() or "") for p in reader.pages)
    return data.decode("utf-8", errors="ignore")


class SelfCheckService:
    def __init__(self, model_path: Path = ROOT / "models" / "confidence.pkl", cache_path: Path | None = None,
                 policy: PolicyConfig | None = None):
        cache = cache_path or ROOT / ".cache" / "llm.sqlite"
        self.generator = LLMClient(GENERATOR, cache_path=cache)
        self.verifier = StepVerifier(NLIChecker(), LLMClient(CRITIC_MODEL, cache_path=cache),
                                     LLMClient(INDEPENDENT_MODEL, cache_path=cache))
        self.confidence = ConfidenceModel.load(model_path) if model_path.exists() else HeuristicConfidence()
        self.policy_cfg = policy or PolicyConfig()

    def ask(self, question: str, docs: list[dict[str, str]], verify: bool = True,
            mode: str = "flag", check_ambiguity: bool = True) -> dict[str, Any]:
        if not docs:
            return {"status": "error", "message": "No documents provided."}
        index = CorpusIndex.from_documents(docs)
        scope = [d["doc_id"] for d in docs]
        agent = DocQAAgent(self.generator, index, AgentConfig())
        if verify:
            cfg = PolicyConfig(**{**self.policy_cfg.__dict__, "mode": mode, "check_ambiguity": check_ambiguity})
            policy = VerificationPolicy(self.verifier, self.confidence, cfg)  # type: ignore[arg-type]
            trace = run_verified(agent, policy, "api", question, scope)
        else:
            trace = agent.run("api", question, scope)
        return to_response(trace)


def to_response(trace) -> dict[str, Any]:
    final = next((e for e in trace.events if e["type"] == "final"), {})
    clarify = next((e for e in trace.events if e["type"] == "clarify"), None)
    status = "clarify" if clarify else final.get("status", "abstained" if trace.abstained else "unverified")
    return {
        "question": trace.question,
        "answer": None if clarify else trace.final_answer,
        "status": status,  # verified | flagged | abstained | clarify | unverified
        "confidence": final.get("confidence"),
        "clarification": clarify and {"message": "This question can be read in several ways. Which one do you mean?",
                                      "interpretations": clarify["interpretations"]},
        "steps": [{
            "sub_question": s["sub_question"], "answer": s["answer"] if s["found"] else None,
            "claim": s.get("claim"), "attempt": s.get("attempt", 0),
            "confidence": s.get("verification", {}).get("confidence"),
            "decision": s.get("verification", {}).get("decision"),
            "critic": {k: s.get("verification", {}).get("signals", {}).get(k)
                       for k in ("critic_verdict", "critic_issue", "independent_answer", "nli_entail_max")},
            "evidence": [{"title": r["title"], "text": r.get("text", "")[:500],
                          "cited": r["chunk_id"] in s["cited_chunk_ids"]} for r in s["retrieved"]],
        } for s in trace.steps],
        "events": [e for e in trace.events if e["type"] in ("retry", "flag")],
        "cost": {"llm_calls": trace.n_calls, "tokens": trace.tokens},
        "error": trace.error,
    }
