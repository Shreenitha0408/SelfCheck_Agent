from selfcheck.eval.metrics import exact_match, is_abstention, score_final, token_f1
from selfcheck.llm import parse_json
from selfcheck.retrieval.index import chunk_document


def test_chunking_overlap_and_ids():
    doc = {"doc_id": "d:1", "title": "T", "text": " ".join(f"w{i}" for i in range(400))}
    chunks = chunk_document(doc, max_words=180, overlap=40)
    assert [c.chunk_id for c in chunks] == ["d:1#0", "d:1#1", "d:1#2"]
    assert chunks[1].text.split()[0] == "w140"  # 180 - 40 overlap
    assert chunks[-1].text.split()[-1] == "w399"
    assert len(chunk_document({"doc_id": "d:2", "title": "", "text": "short"})) == 1


def test_answer_metrics():
    assert exact_match("The Beatles", ["beatles"]) == 1.0
    assert 0 < token_f1("a Ghanaian boxer", ["Ghanaian"]) < 1
    assert is_abstention("UNANSWERABLE.")
    unans = {"answerable": False, "answers": []}
    assert score_final("UNANSWERABLE", unans)["em"] == 1.0
    assert score_final("Paris", unans)["em"] == 0.0
    ans = {"answerable": True, "answers": ["Paris"]}
    assert score_final("UNANSWERABLE", ans)["f1"] == 0.0


def test_parse_json_tolerant():
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json("no json here") == {}
