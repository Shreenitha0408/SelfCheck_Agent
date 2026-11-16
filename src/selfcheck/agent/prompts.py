"""Prompts for the multi-step document-QA agent."""

PLAN_SYSTEM = (
    "You answer questions using a document collection that you can search. "
    "Complex questions must be broken into simple sub-questions that are answered one at a time."
)

PLAN_USER = """Question: {question}

Write the FIRST simple sub-question to search for. It must be answerable from a single passage.
If the question is already simple, the sub-question can be the question itself.
Do NOT guess any unknown entity from memory: if the question refers to something indirectly
("the leader of X", "the agency that created Y"), the first sub-question must identify it.

Reply with JSON only: {{"sub_question": "..."}}"""

STEP_SYSTEM = (
    "You are a careful research assistant. Answer strictly from the provided passages, never from memory."
)

STEP_USER = """Main question: {question}

Steps completed so far:
{history}

Current sub-question: {sub_question}

Passages retrieved for the current sub-question:
{passages}

Tasks:
1. Answer the current sub-question using ONLY the passages. Be concise (a short phrase).
   If the passages do not contain the answer, set "found" to false.
   Also write "claim": one plain declarative sentence stating your answer as a fact
   (e.g. "Beatrice Heuser was born in Bangkok."), or "" if not found.
2. Decide what to do next:
   - if more information is needed for the main question, give the next simple sub-question
     (substitute earlier answers into it, e.g. write the actual name, not "#1"; never guess
     unknown entities from memory). If the current sub-question was NOT found, you may try
     one rephrased sub-question instead of giving up;
   - otherwise give the final answer to the main question: a short phrase, or "yes"/"no".
     Answer even if the passages use slightly different wording than the question
     (e.g. "district" vs "state"). Use "UNANSWERABLE" only if a needed fact is truly missing
     from the passages.

Reply with JSON only:
{{"step_answer": "...", "found": true, "claim": "...", "evidence": [passage numbers used],
  "next_sub_question": "..." or null, "final_answer": "..." or null}}"""

FORCE_FINAL_USER = """Main question: {question}

Steps completed:
{history}

Give the final answer to the main question as a short phrase or "yes"/"no", based only on the
steps above. Use "UNANSWERABLE" only if a needed fact was not found.

Reply with JSON only: {{"final_answer": "..."}}"""


def format_passages(passages) -> str:
    return "\n\n".join(f"[{i + 1}] {p.title}\n{p.text}" for i, p in enumerate(passages))


def format_history(steps) -> str:
    if not steps:
        return "(none)"
    return "\n".join(
        f"{i + 1}. {s['sub_question']} -> {s['answer'] if s['found'] else 'NOT FOUND'}"
        for i, s in enumerate(steps)
    )
