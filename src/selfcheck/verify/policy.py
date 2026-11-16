"""Online verification policy: plugs into ``DocQAAgent.step_hook``.

After every step:   signals -> calibrated confidence ->
    conf >= tau_accept              : accept, continue
    conf <  tau_accept, no retry yet: retry the step once (critic's better query, wider top-k)
    still low                       : flag the step (mode="flag") or stop and abstain (mode="abstain")
Before the first step (optional): ambiguity check -> ask for clarification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..agent.agent import DocQAAgent, Trace
from ..llm import LLMClient, parse_json
from .confidence import ConfidenceModel
from .signals import CRITIC_MODEL, StepVerifier

AMBIGUITY_PROMPT = """A user asked: "{question}"

Passages found:
{passages}

Does this question have MORE THAN ONE reasonable interpretation that would lead to DIFFERENT answers
according to these passages (e.g. different years, seasons, versions, people with the same name)?
Only say yes if the passages actually support several different answers.

Reply with JSON only:
{{"ambiguous": true/false, "confidence": <0-100>,
  "interpretations": ["<specific rephrased question 1>", "<specific rephrased question 2>", ...]}}"""


@dataclass
class PolicyConfig:
    tau_accept: float = 0.5
    max_retries: int = 1
    retry_top_k: int = 8
    mode: str = "flag"  # "flag" | "abstain"
    check_ambiguity: bool = True
    tau_ambiguous: float = 0.7


class VerificationPolicy:
    def __init__(self, verifier: StepVerifier, model: ConfidenceModel, cfg: PolicyConfig | None = None,
                 critic: LLMClient | None = None):
        self.verifier, self.model, self.cfg = verifier, model, cfg or PolicyConfig()
        self.critic = critic or verifier.critic or LLMClient(CRITIC_MODEL)

    def _verify(self, step: dict[str, Any]) -> tuple[float, dict[str, Any]]:
        sig = self.verifier.signals(step["sub_question"], step["answer"], step["found"], step["retrieved"],
                                    step["cited_chunk_ids"], claim=step.get("claim", ""))
        return self.model.score_step(sig), sig

    def __call__(self, *, agent: DocQAAgent, trace: Trace, step: dict[str, Any], question: str,
                 scope: list[str]) -> dict[str, Any]:
        conf, sig = self._verify(step)
        step["verification"] = {"confidence": conf, "signals": _slim(sig), "decision": "accept"}
        trace.n_calls += 2  # critic + independent pass
        if conf >= self.cfg.tau_accept:
            return step

        best = step
        for attempt in range(1, self.cfg.max_retries + 1):
            query = sig.get("critic_better_query") or f"{step['sub_question']} {step.get('claim', '')}".strip()
            old_k = agent.cfg.top_k
            agent.cfg.top_k = self.cfg.retry_top_k
            try:
                retry = agent.run_step(trace, question, step["sub_question"], scope, query=query, attempt=attempt)
            finally:
                agent.cfg.top_k = old_k
            r_conf, r_sig = self._verify(retry)
            trace.n_calls += 2
            retry["verification"] = {"confidence": r_conf, "signals": _slim(r_sig), "decision": "retry"}
            trace.events.append({"type": "retry", "sub_question": step["sub_question"],
                                 "conf_before": conf, "conf_after": r_conf})
            if r_conf > best["verification"]["confidence"]:
                best = retry
            if r_conf >= self.cfg.tau_accept:
                best["verification"]["decision"] = "accept_after_retry"
                return best

        best["verification"]["decision"] = "flag"
        best["flagged"] = True
        trace.events.append({"type": "flag", "sub_question": step["sub_question"],
                             "confidence": best["verification"]["confidence"]})
        if self.cfg.mode == "abstain":
            best["halt"] = True
            best["final_answer"] = "UNANSWERABLE"
        return best

    # -------------------------------------------------------------- ambiguity
    def check_ambiguity(self, agent: DocQAAgent, question: str, scope: list[str]) -> dict[str, Any]:
        passages = agent.index.search(question, scope, k=6, method=agent.cfg.retrieval_method)
        txt = "\n\n".join(f"[{i + 1}] {p.title}: {p.text[:600]}" for i, p in enumerate(passages))
        j = parse_json(self.critic.chat([{"role": "user", "content": AMBIGUITY_PROMPT.format(
            question=question, passages=txt)}], max_tokens=400, json_mode=True).content)
        try:
            conf = float(j.get("confidence", 0)) / 100
        except (TypeError, ValueError):
            conf = 0.0
        amb = bool(j.get("ambiguous")) and conf >= self.cfg.tau_ambiguous and len(j.get("interpretations") or []) >= 2
        return {"ambiguous": amb, "confidence": conf, "interpretations": (j.get("interpretations") or [])[:5]}


def _slim(sig: dict[str, Any]) -> dict[str, Any]:
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in sig.items()}


def run_verified(agent: DocQAAgent, policy: VerificationPolicy, qid: str, question: str,
                 scope: list[str]) -> Trace:
    """Full verified run: ambiguity check -> agent with per-step verification -> final status."""
    if policy.cfg.check_ambiguity:
        amb = policy.check_ambiguity(agent, question, scope)
        if amb["ambiguous"]:
            t = Trace(qid=qid, question=question, final_answer="CLARIFY", n_calls=1)
            t.events.append({"type": "clarify", **amb})
            return t
    agent.step_hook = policy
    t = agent.run(qid, question, scope)
    if policy.cfg.check_ambiguity:
        t.n_calls += 1
    confs = [s["verification"]["confidence"] for s in t.steps if "verification" in s]
    t.events.append({"type": "final", "confidence": min(confs) if confs else None,
                     "status": "abstained" if t.abstained else
                     "flagged" if any(s.get("flagged") for s in t.steps) else "verified"})
    return t
