"""Multi-step document-QA agent.

Loop:  plan first sub-question -> [retrieve -> answer step + choose next] x N -> final answer.

Each step's output is a self-contained record (sub-question, retrieved passages,
answer, cited evidence). The verification layer (next stage) plugs in through
``step_hook``: it is called after every step and before the agent moves on, and
can accept the step or ask for a retry. With ``step_hook=None`` this is the
**baseline** agent (no verification).
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from ..llm import LLMClient, parse_json
from ..retrieval.index import CorpusIndex, Retrieved
from . import prompts as P

UNANSWERABLE = "UNANSWERABLE"


@dataclass
class AgentConfig:
    top_k: int = 5
    max_steps: int = 5
    retrieval_method: str = "hybrid_rerank"
    step_max_tokens: int = 700
    plan_max_tokens: int = 300


@dataclass
class Trace:
    qid: str
    question: str
    steps: list[dict[str, Any]] = field(default_factory=list)
    final_answer: str = ""
    abstained: bool = False
    n_calls: int = 0
    tokens: int = 0
    latency_s: float = 0.0
    events: list[dict[str, Any]] = field(default_factory=list)  # verifier decisions etc.
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DocQAAgent:
    def __init__(self, llm: LLMClient, index: CorpusIndex, cfg: AgentConfig | None = None,
                 step_hook: Callable[..., dict[str, Any]] | None = None):
        self.llm, self.index, self.cfg, self.step_hook = llm, index, cfg or AgentConfig(), step_hook

    def _call(self, trace: Trace, system: str, user: str, max_tokens: int) -> dict[str, Any]:
        r = self.llm.chat([{"role": "system", "content": system}, {"role": "user", "content": user}],
                          max_tokens=max_tokens, json_mode=True)
        trace.n_calls += 1
        trace.tokens += r.usage.get("total_tokens", 0)
        trace.latency_s += r.latency_s
        return {"parsed": parse_json(r.content), "raw": r.content, "usage": r.usage}

    def run_step(self, trace: Trace, question: str, sub_question: str, scope: list[str],
                 query: str | None = None, attempt: int = 0) -> dict[str, Any]:
        t0 = time.time()
        passages: list[Retrieved] = self.index.search(query or sub_question, scope, k=self.cfg.top_k,
                                                      method=self.cfg.retrieval_method)
        out = self._call(trace, P.STEP_SYSTEM, P.STEP_USER.format(
            question=question, history=P.format_history(trace.steps), sub_question=sub_question,
            passages=P.format_passages(passages)), self.cfg.step_max_tokens)
        j = out["parsed"]
        cited = []
        for n in j.get("evidence") or []:
            try:
                if 1 <= int(n) <= len(passages):
                    cited.append(passages[int(n) - 1].chunk_id)
            except (TypeError, ValueError):
                pass
        return {
            "sub_question": sub_question,
            "query": query or sub_question,
            "attempt": attempt,
            "retrieved": [{"chunk_id": p.chunk_id, "doc_id": p.doc_id, "title": p.title, "text": p.text,
                           "score": p.score} for p in passages],
            "answer": str(j.get("step_answer") or "").strip(),
            "found": bool(j.get("found", False)),
            "cited_chunk_ids": cited,
            "next_sub_question": j.get("next_sub_question"),
            "final_answer": j.get("final_answer"),
            "parse_ok": bool(j),
            "raw": out["raw"],
            "usage": out["usage"],
            "wall_s": round(time.time() - t0, 2),
        }

    def run(self, qid: str, question: str, scope: list[str]) -> Trace:
        trace = Trace(qid=qid, question=question)
        try:
            plan = self._call(trace, P.PLAN_SYSTEM, P.PLAN_USER.format(question=question),
                              self.cfg.plan_max_tokens)
            sub_q = plan["parsed"].get("sub_question") or question
            final = None
            for _ in range(self.cfg.max_steps):
                step = self.run_step(trace, question, sub_q, scope)
                if self.step_hook is not None:
                    step = self.step_hook(agent=self, trace=trace, step=step, question=question, scope=scope)
                trace.steps.append(step)
                if step.get("halt"):  # verifier decided to abstain / clarify
                    final = step.get("final_answer") or UNANSWERABLE
                    break
                if step.get("final_answer"):
                    final = str(step["final_answer"]).strip()
                    break
                if not step.get("next_sub_question"):
                    break
                sub_q = str(step["next_sub_question"]).strip()
            if final is None:
                out = self._call(trace, P.STEP_SYSTEM, P.FORCE_FINAL_USER.format(
                    question=question, history=P.format_history(trace.steps)), self.cfg.plan_max_tokens)
                final = str(out["parsed"].get("final_answer") or UNANSWERABLE).strip()
            trace.final_answer = final
            trace.abstained = final.upper().strip(". ") == UNANSWERABLE
        except Exception as e:  # recorded, never silently dropped
            trace.error = f"{type(e).__name__}: {e}"
            if type(e).__name__ == "DailyQuotaExceeded":
                raise
        return trace
