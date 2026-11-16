"""Unified data schema shared by every dataset.

Every source dataset is normalised into two record types:

* ``Document`` – one retrievable passage (a Wikipedia paragraph, a paper
  paragraph, a search snippet). Documents are de-duplicated per dataset and
  addressed by a content hash so the same paragraph shared by several
  questions has one id.
* ``Example`` – one question with its gold answer(s), gold evidence and,
  where the source provides it, a gold multi-step decomposition.

The verifier evaluation needs to know, for every agent step and final answer,
(a) what the correct answer is and (b) which passages actually support it.
That is why gold evidence and per-step gold answers are first-class fields.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any

ANSWER_TYPES = ("span", "yesno", "free", "none")


def doc_id_for(dataset: str, title: str, text: str) -> str:
    h = hashlib.sha1(f"{title}\n{text}".encode("utf-8")).hexdigest()[:16]
    return f"{dataset}:{h}"


@dataclass
class Document:
    doc_id: str
    dataset: str
    title: str
    text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Step:
    """One gold reasoning hop (MuSiQue provides these)."""

    sub_question: str  # may reference earlier hops as "#1", "#2", ...
    answer: str
    gold_doc_ids: list[str] = field(default_factory=list)


@dataclass
class Example:
    qid: str
    dataset: str
    question: str
    answers: list[str]  # all acceptable gold answers (aliases / annotators); [] if unanswerable
    answer_type: str  # one of ANSWER_TYPES
    answerable: bool  # False -> the correct behaviour is to abstain
    ambiguous: bool  # True -> the correct behaviour is to ask for clarification
    gold_doc_ids: list[str]  # passages that support the answer
    scope_doc_ids: list[str]  # the question's own passage pool (distractor set / paper / snippets)
    steps: list[Step] = field(default_factory=list)
    disambiguations: list[dict[str, Any]] = field(default_factory=list)  # [{question, answers}]
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.answer_type not in ANSWER_TYPES:
            raise ValueError(f"bad answer_type {self.answer_type!r} for {self.qid}")
        if self.answerable and not self.ambiguous and not self.answers:
            raise ValueError(f"answerable example {self.qid} has no gold answers")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Example":
        d = dict(d)
        d["steps"] = [Step(**s) for s in d.get("steps", [])]
        return cls(**d)
