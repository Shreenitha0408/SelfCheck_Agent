"""LLM-as-judge correctness labels (gpt-oss-120b), batched to save free-tier requests.

Two label types:
* final answer  -> is the prediction semantically equivalent to a gold answer?
* step answer   -> is the agent's answer to ITS OWN sub-question correct, judged against
                   the gold reference facts (gold hop answers + gold evidence passages)?

Cheap rules decide the obvious cases first (exact match, abstention on an
(un)answerable question); only the rest is sent to the judge. Agreement of the
judge with human labels is measured separately (human audit, Cohen's kappa).
"""

from __future__ import annotations

import json
from typing import Any

from ..llm import LLMClient, parse_json
from .metrics import exact_match, is_abstention

JUDGE_MODEL = "openai/gpt-oss-120b"

FINAL_PROMPT = """You are grading answers to questions. For each item decide whether the PREDICTION
means the same as at least one GOLD answer (same entity/date/number/claim; extra harmless detail or
different wording is fine; a different or partially wrong fact is not; for yes/no, polarity must match).

{items}

Reply with JSON only: {{"labels": [{{"id": <id>, "correct": true/false}} , ...]}}"""

STEP_PROMPT = """You are grading intermediate steps of a question-answering agent. Each item has a
SUB-QUESTION the agent asked, the agent's STEP ANSWER, and REFERENCE FACTS (true facts for this question).
Mark "correct" if the step answer is a correct answer to the sub-question according to the reference
facts. If the reference facts do not cover the sub-question, mark "unknown". "NOT FOUND" answers are
"correct" only if the reference facts show the sub-question cannot be answered.

{items}

Reply with JSON only: {{"labels": [{{"id": <id>, "label": "correct" | "incorrect" | "unknown"}}, ...]}}"""


class Judge:
    def __init__(self, llm: LLMClient | None = None, batch_size: int = 8):
        self.llm = llm or LLMClient(JUDGE_MODEL)
        self.batch_size = batch_size

    def _ask(self, prompt: str, max_tokens: int = 1500) -> dict[str, Any]:
        r = self.llm.chat([{"role": "user", "content": prompt}], max_tokens=max_tokens, json_mode=True)
        return parse_json(r.content)

    # ----------------------------------------------------------------- final
    def label_finals(self, items: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        """items: {key, question, pred, golds, answerable}. Returns key -> {correct, source}."""
        out: dict[str, dict[str, Any]] = {}
        todo = []
        for it in items:
            abst = is_abstention(it["pred"])
            if not it["answerable"]:
                out[it["key"]] = {"correct": abst, "source": "rule"}
            elif abst:
                out[it["key"]] = {"correct": False, "source": "rule"}
            elif exact_match(it["pred"], it["golds"]):
                out[it["key"]] = {"correct": True, "source": "rule"}
            else:
                todo.append(it)
        for b in range(0, len(todo), self.batch_size):
            batch = todo[b : b + self.batch_size]
            txt = "\n\n".join(
                f"[id {i}]\nQUESTION: {it['question']}\nGOLD: {json.dumps(it['golds'][:6])}\nPREDICTION: {it['pred']}"
                for i, it in enumerate(batch)
            )
            labels = {int(x.get("id", -1)): x for x in self._ask(FINAL_PROMPT.format(items=txt)).get("labels", [])}
            for i, it in enumerate(batch):
                if i in labels:
                    out[it["key"]] = {"correct": bool(labels[i].get("correct")), "source": "judge"}
                else:
                    out[it["key"]] = {"correct": None, "source": "judge_missing"}
        return out

    # ------------------------------------------------------------------ step
    def label_steps(self, items: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        """items: {key, sub_question, answer, found, reference}. Returns key -> {label, source}."""
        out: dict[str, dict[str, Any]] = {}
        for b in range(0, len(items), self.batch_size):
            batch = items[b : b + self.batch_size]
            txt = "\n\n".join(
                f"[id {i}]\nSUB-QUESTION: {it['sub_question']}\n"
                f"STEP ANSWER: {it['answer'] if it['found'] else 'NOT FOUND'}\n"
                f"REFERENCE FACTS:\n{it['reference']}"
                for i, it in enumerate(batch)
            )
            labels = {int(x.get("id", -1)): x for x in self._ask(STEP_PROMPT.format(items=txt), 2000).get("labels", [])}
            for i, it in enumerate(batch):
                lab = labels.get(i, {}).get("label")
                out[it["key"]] = {"label": lab if lab in ("correct", "incorrect", "unknown") else None,
                                  "source": "judge" if lab else "judge_missing"}
        return out


def reference_facts(example: dict[str, Any], corpus: dict[str, dict[str, Any]], max_chars: int = 2500) -> str:
    """Gold facts shown to the step judge: gold answers, gold hops, gold evidence text."""
    lines = []
    if example["answerable"] and example["answers"]:
        lines.append(f"- Final answer to the main question: {example['answers'][0]}")
    elif not example["answerable"]:
        lines.append("- The main question is NOT answerable from the documents (a needed fact is missing).")
    for s in example.get("steps", []):
        lines.append(f"- {s['sub_question']} -> {s['answer']}")
    for d in example.get("disambiguations", [])[:4]:
        lines.append(f"- {d['question']} -> {', '.join(d['answers'][:2])}")
    ev = "\n".join(f"  [{corpus[d]['title']}] {corpus[d]['text']}" for d in example.get("gold_doc_ids", []) if d in corpus)
    if ev:
        lines.append("- Gold evidence passages:\n" + ev)
    return "\n".join(lines)[:max_chars]
