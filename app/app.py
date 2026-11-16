"""SelfCheck Agent — web demo (Gradio UI) + JSON API (FastAPI), one process.

    python app/app.py              # http://localhost:7860  (UI at /, API at /api/ask)

Needs a Groq API key in the environment (GROQ_API_KEY) unless a proxy injects it.
"""

from __future__ import annotations

import html
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import gradio as gr  # noqa: E402
import uvicorn  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from selfcheck.service import SelfCheckService, read_upload, split_documents  # noqa: E402

EXAMPLE_DOC = (ROOT / "app" / "example.txt").read_text() if (ROOT / "app" / "example.txt").exists() else ""
_service: SelfCheckService | None = None


def service() -> SelfCheckService:
    global _service
    if _service is None:
        _service = SelfCheckService()
    return _service


# ---------------------------------------------------------------- JSON API
api = FastAPI(title="SelfCheck Agent API", version="0.1.0")


class AskRequest(BaseModel):
    question: str
    documents: list[str]
    verify: bool = True
    mode: str = "flag"
    check_ambiguity: bool = True


@api.get("/api/health")
def health() -> dict:
    return {"ok": True}


@api.post("/api/ask")
def ask(req: AskRequest) -> dict:
    docs = [d for i, t in enumerate(req.documents) for d in split_documents(t, f"doc{i + 1}")]
    return service().ask(req.question, docs, verify=req.verify, mode=req.mode, check_ambiguity=req.check_ambiguity)


# ---------------------------------------------------------------- UI
BADGE = {"verified": ("✅ Verified", "#1a7f37"), "flagged": ("⚠️ Uncertain — flagged", "#9a6700"),
         "abstained": ("🛑 Not answerable from the documents", "#cf222e"),
         "clarify": ("❓ Needs clarification", "#0969da"), "unverified": ("Unverified (baseline)", "#57606a")}
DECISION = {"accept": "accepted", "accept_after_retry": "accepted after retry", "retry": "retry attempt",
            "flag": "flagged", None: "—"}


def render(res: dict) -> str:
    if res.get("status") == "error":
        return f"<p>{html.escape(res['message'])}</p>"
    label, color = BADGE.get(res["status"], (res["status"], "#57606a"))
    conf = res.get("confidence")
    out = [f"<div style='border-left:4px solid {color};padding:8px 12px;margin-bottom:12px'>",
           f"<div style='color:{color};font-weight:600'>{label}"
           + (f" · confidence {conf:.0%}" if conf is not None else "") + "</div>"]
    if res["status"] == "clarify":
        out.append(f"<p>{html.escape(res['clarification']['message'])}</p><ul>")
        out += [f"<li>{html.escape(q)}</li>" for q in res["clarification"]["interpretations"]]
        out.append("</ul></div>")
        return "".join(out)
    out.append(f"<div style='font-size:1.3em;margin-top:4px'>{html.escape(str(res['answer']))}</div></div>")
    out.append("<h4>Reasoning steps</h4>")
    for i, s in enumerate(res["steps"], 1):
        c = s.get("confidence")
        crit = s.get("critic") or {}
        out.append("<details style='margin-bottom:6px'><summary>"
                   f"<b>Step {i}{' (retry)' if s['attempt'] else ''}:</b> {html.escape(s['sub_question'])} → "
                   f"<b>{html.escape(str(s['answer'] or 'not found'))}</b>"
                   + (f" · {c:.0%} · {DECISION.get(s.get('decision'), s.get('decision'))}" if c is not None else "")
                   + "</summary><div style='padding:6px 12px'>")
        if crit.get("critic_verdict"):
            out.append(f"<p>Critic: <b>{html.escape(str(crit['critic_verdict']))}</b> "
                       f"{html.escape(str(crit.get('critic_issue') or ''))}<br>"
                       f"Independent re-answer: <i>{html.escape(str(crit.get('independent_answer') or ''))}</i> · "
                       f"evidence entailment {float(crit.get('nli_entail_max') or 0):.0%}</p>")
        for e in s["evidence"]:
            out.append(f"<p style='font-size:0.9em;{'' if e['cited'] else 'opacity:0.6'}'>"
                       f"{'📌 ' if e['cited'] else ''}<b>{html.escape(e['title'])}</b>: {html.escape(e['text'])}</p>")
        out.append("</div></details>")
    out.append(f"<p style='opacity:0.7;font-size:0.85em'>{res['cost']['llm_calls']} LLM calls</p>")
    return "".join(out)


def run_ui(question, text, files, verify, mode):
    docs = split_documents(text or "", "pasted")
    for f in files or []:
        p = Path(f if isinstance(f, str) else f.name)
        docs += split_documents(read_upload(p.name, p.read_bytes()), p.stem[:30])
    if not question.strip():
        return "<p>Please enter a question.</p>"
    return render(service().ask(question, docs, verify=verify, mode=mode))


with gr.Blocks(title="SelfCheck Agent") as demo:
    gr.Markdown("# SelfCheck Agent\nDocument QA that **checks its own work**: every reasoning step is verified "
                "against the evidence and by an independent model before the agent moves on. Low-confidence "
                "steps are retried, then flagged — instead of being presented as fact.")
    with gr.Row():
        with gr.Column():
            q = gr.Textbox(label="Question", placeholder="Ask something the documents can answer…")
            txt = gr.Textbox(label="Documents (paste text; blank lines separate passages)", lines=12, value=EXAMPLE_DOC)
            files = gr.File(label="…or upload .txt / .md / .pdf", file_count="multiple",
                            file_types=[".txt", ".md", ".pdf"])
            with gr.Row():
                verify = gr.Checkbox(value=True, label="Verification on")
                mode = gr.Radio(["flag", "abstain"], value="flag", label="When unsure")
            btn = gr.Button("Ask", variant="primary")
        with gr.Column():
            out = gr.HTML()
    btn.click(run_ui, [q, txt, files, verify, mode], out)

app = gr.mount_gradio_app(api, demo, path="/")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))
