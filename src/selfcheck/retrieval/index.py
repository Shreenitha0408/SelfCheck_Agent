"""Chunking + hybrid retrieval (BM25 + dense, reciprocal-rank fusion) + cross-encoder reranking.

Retrieval is *scoped*: each question searches only its own passage pool
(``Example.scope_doc_ids`` — the distractor set, the paper, or the snippets),
which is the standard setting for these benchmarks. Embeddings for the whole
corpus are computed once on CPU and cached under ``.cache/emb``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = set(
    "a an the of in on at to for from by with and or is are was were be been what which who whom "
    "whose when where why how did does do that this these those as it its their they he she his her".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    text: str


@dataclass
class Retrieved:
    chunk_id: str
    doc_id: str
    title: str
    text: str
    score: float
    ranks: dict[str, int] = field(default_factory=dict)


def chunk_document(doc: dict, max_words: int = 180, overlap: int = 40) -> list[Chunk]:
    words = doc["text"].split()
    if len(words) <= max_words:
        return [Chunk(f"{doc['doc_id']}#0", doc["doc_id"], doc["title"], doc["text"])]
    out, start, k = [], 0, 0
    while start < len(words):
        piece = " ".join(words[start : start + max_words])
        out.append(Chunk(f"{doc['doc_id']}#{k}", doc["doc_id"], doc["title"], piece))
        if start + max_words >= len(words):
            break
        start += max_words - overlap
        k += 1
    return out


class CorpusIndex:
    def __init__(self, processed_dir: Path | None, dataset: str, cache_dir: Path | None = Path(".cache/emb"),
                 max_words: int = 180, overlap: int = 40, docs: list[dict] | None = None):
        self.dataset = dataset
        if docs is None:
            assert processed_dir is not None
            docs = [json.loads(l) for l in open(processed_dir / dataset / "corpus.jsonl")]
        self.chunks: list[Chunk] = [c for d in docs for c in chunk_document(d, max_words, overlap)]
        self.by_doc: dict[str, list[int]] = {}
        for i, c in enumerate(self.chunks):
            self.by_doc.setdefault(c.doc_id, []).append(i)
        self._tokens = [tokenize(f"{c.title} {c.text}") for c in self.chunks]
        self._embedder = None
        self._reranker = None
        self.emb = self._load_embeddings(cache_dir)

    # ------------------------------------------------------------------ models
    @property
    def embedder(self):
        if self._embedder is None:
            from fastembed import TextEmbedding

            self._embedder = TextEmbedding(EMBED_MODEL)
        return self._embedder

    @property
    def reranker(self):
        if self._reranker is None:
            from fastembed.rerank.cross_encoder import TextCrossEncoder

            self._reranker = TextCrossEncoder(RERANK_MODEL)
        return self._reranker

    @classmethod
    def from_documents(cls, docs: list[dict], name: str = "user") -> "CorpusIndex":
        """In-memory index over user-supplied documents (used by the demo app; no disk cache)."""
        return cls(None, name, cache_dir=None, docs=docs)

    def _embed(self, texts: list[str]) -> np.ndarray:
        v = np.asarray(list(self.embedder.embed(texts, batch_size=64)), dtype=np.float32)
        return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)

    def _load_embeddings(self, cache_dir: Path | None) -> np.ndarray:
        if cache_dir is None:
            return self._embed([f"{c.title}. {c.text}" for c in self.chunks]) if self.chunks else np.zeros((0, 384), np.float32)
        cache_dir.mkdir(parents=True, exist_ok=True)
        tag = EMBED_MODEL.replace("/", "_")
        vec_path, ids_path = cache_dir / f"{self.dataset}_{tag}.npy", cache_dir / f"{self.dataset}_{tag}.ids"
        ids = [c.chunk_id for c in self.chunks]
        cached: dict[str, np.ndarray] = {}
        if vec_path.exists() and ids_path.exists():
            old_ids, old_vecs = ids_path.read_text().split("\n"), np.load(vec_path)
            if old_ids == ids:
                return old_vecs
            cached = dict(zip(old_ids, old_vecs))  # chunk ids are content hashes -> safe to reuse
        missing = [i for i, cid in enumerate(ids) if cid not in cached]
        if missing:
            new = self._embed([f"{self.chunks[i].title}. {self.chunks[i].text}" for i in missing])
            cached.update({ids[i]: v for i, v in zip(missing, new)})
        vecs = np.stack([cached[cid] for cid in ids]).astype(np.float32)
        np.save(vec_path, vecs)
        ids_path.write_text("\n".join(ids))
        return vecs

    # ---------------------------------------------------------------- retrieval
    def scope_chunks(self, scope_doc_ids: list[str]) -> list[int]:
        return [i for d in scope_doc_ids for i in self.by_doc.get(d, [])]

    def search(self, query: str, scope_doc_ids: list[str], k: int = 5, method: str = "hybrid_rerank",
               candidates: int = 20, rrf_k: int = 60) -> list[Retrieved]:
        idx = self.scope_chunks(scope_doc_ids)
        if not idx:
            return []
        ranks: dict[int, dict[str, int]] = {i: {} for i in idx}
        score: dict[int, float] = {}

        if method in ("bm25", "hybrid", "hybrid_rerank"):
            bm = BM25Okapi([self._tokens[i] for i in idx])
            s = bm.get_scores(tokenize(query))
            for r, j in enumerate(np.argsort(-s, kind="stable")):
                ranks[idx[j]]["bm25"] = r + 1
        if method in ("dense", "hybrid", "hybrid_rerank"):
            q = np.asarray(next(iter(self.embedder.query_embed(query))), dtype=np.float32)
            q /= np.linalg.norm(q) + 1e-12
            s = self.emb[idx] @ q
            for r, j in enumerate(np.argsort(-s, kind="stable")):
                ranks[idx[j]]["dense"] = r + 1

        for i in idx:
            score[i] = sum(1.0 / (rrf_k + r) for r in ranks[i].values())
        order = sorted(idx, key=lambda i: -score[i])

        if method == "hybrid_rerank":
            cand = order[:candidates]
            rr = list(self.reranker.rerank(query, [f"{self.chunks[i].title}. {self.chunks[i].text}" for i in cand]))
            for i, s_ in zip(cand, rr):
                score[i] = float(s_)
            order = sorted(cand, key=lambda i: -score[i])
            for r, i in enumerate(order):
                ranks[i]["rerank"] = r + 1

        out = []
        for i in order[:k]:
            c = self.chunks[i]
            out.append(Retrieved(c.chunk_id, c.doc_id, c.title, c.text, round(score[i], 4), ranks[i]))
        return out
