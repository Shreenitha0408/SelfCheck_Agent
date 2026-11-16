"""Groq (OpenAI-compatible) chat client with a persistent cache and free-tier pacing.

* Every request/response is cached in SQLite keyed by the full request, so
  re-running an experiment costs nothing and results are reproducible.
* A rolling 60-second token budget keeps us under the tokens-per-minute limit;
  429s are retried after the server's ``retry-after``.
* When the per-day request quota is exhausted we raise ``DailyQuotaExceeded``
  so long runs can stop cleanly and resume the next day (runs are resumable).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


class DailyQuotaExceeded(RuntimeError):
    pass


@dataclass
class LLMResponse:
    content: str
    usage: dict[str, Any]
    latency_s: float
    cached: bool
    model: str


class LLMClient:
    def __init__(self, model: str, cache_path: Path = Path(".cache/llm.sqlite"),
                 tpm_limit: int = 8000, timeout: float = 120.0):
        self.model = model
        self.tpm_limit = tpm_limit
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(cache_path, check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT)")
        self._lock = threading.Lock()
        self._window: deque[tuple[float, int]] = deque()
        # The network proxy injects the real key; a placeholder is enough here.
        key = os.environ.get("GROQ_API_KEY", "proxy-injected")
        self._http = httpx.Client(timeout=timeout, headers={"Authorization": f"Bearer {key}"})

    # ----------------------------------------------------------------- pacing
    def _used_last_minute(self) -> int:
        now = time.time()
        while self._window and now - self._window[0][0] > 60:
            self._window.popleft()
        return sum(t for _, t in self._window)

    def _wait_for_budget(self, est_tokens: int) -> None:
        while self._used_last_minute() + est_tokens > self.tpm_limit and self._window:
            time.sleep(max(0.5, 60 - (time.time() - self._window[0][0])))

    # ------------------------------------------------------------------- call
    def chat(self, messages: list[dict[str, str]], temperature: float = 0.0, max_tokens: int = 600,
             sample_idx: int = 0, json_mode: bool = False, **extra: Any) -> LLMResponse:
        body: dict[str, Any] = {"model": self.model, "messages": messages,
                                "temperature": temperature, "max_tokens": max_tokens, **extra}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        if self.model.startswith("openai/gpt-oss"):
            body.setdefault("reasoning_effort", "low")
        key = hashlib.sha256(json.dumps([body, sample_idx], sort_keys=True).encode()).hexdigest()
        row = self._db.execute("SELECT value FROM cache WHERE key=?", (key,)).fetchone()
        if row:
            v = json.loads(row[0])
            return LLMResponse(v["content"], v["usage"], v["latency_s"], True, self.model)

        est = sum(len(m["content"]) for m in messages) // 4 + max_tokens
        for attempt in range(8):
            with self._lock:
                self._wait_for_budget(est)
                t0 = time.time()
                try:
                    r = self._http.post(GROQ_URL, json=body)
                except httpx.HTTPError:
                    time.sleep(2 ** attempt)
                    continue
                latency = time.time() - t0
                if r.status_code == 200:
                    d = r.json()
                    usage = d.get("usage", {})
                    self._window.append((time.time(), usage.get("total_tokens", est)))
                    content = d["choices"][0]["message"].get("content") or ""
                    self._db.execute("INSERT OR REPLACE INTO cache VALUES (?,?)", (key, json.dumps(
                        {"content": content, "usage": usage, "latency_s": latency})))
                    self._db.commit()
                    return LLMResponse(content, usage, latency, False, self.model)
                if r.status_code == 429:
                    msg = r.text.lower()
                    if "per day" in msg or "(rpd)" in msg or "(tpd)" in msg:
                        raise DailyQuotaExceeded(f"{self.model}: {r.json().get('error', {}).get('message', r.text)[:300]}")
                    wait = float(r.headers.get("retry-after", 2 ** attempt))
                    self._window.append((time.time(), est))
                    time.sleep(min(wait + 0.5, 90))
                    continue
                if r.status_code == 400 and json_mode and "json" in r.text.lower():
                    # model produced invalid JSON under json_mode -> retry once without it
                    body.pop("response_format", None)
                    json_mode = False
                    continue
                if r.status_code >= 500:
                    time.sleep(2 ** attempt)
                    continue
                raise RuntimeError(f"Groq {r.status_code}: {r.text[:500]}")
        raise RuntimeError(f"{self.model}: giving up after retries")


def parse_json(text: str) -> dict[str, Any]:
    """Extract the first JSON object from a model reply (tolerates code fences / prose)."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return {}
