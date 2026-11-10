import json

import pytest

import agent
import eval as ev
import ingest
import predictor
from conftest import FakeClient, text_response, tool_response


# ---------------------------------------------------------------- ingest
def test_chunk_text_covers_all_content_and_has_no_empty_chunks():
    text = " ".join(f"word{i}" for i in range(500))
    chunks = ingest.chunk_text(text, chunk_size=200, overlap=40)
    assert all(c.strip() for c in chunks)
    assert all(len(c) <= 200 for c in chunks)
    joined = " ".join(chunks)
    assert all(f"word{i}" in joined for i in range(500))


def test_chunks_start_and_end_on_whole_words():
    words = [f"word{i}" for i in range(400)]
    chunks = ingest.chunk_text(" ".join(words), chunk_size=200, overlap=40)
    valid = set(words)
    for c in chunks:
        parts = c.split()
        assert parts[0] in valid and parts[-1] in valid, c[:30]


def test_chunk_text_rejects_bad_overlap():
    with pytest.raises(ValueError):
        ingest.chunk_text("abc " * 100, chunk_size=50, overlap=50)


def test_chunk_text_handles_text_without_spaces():
    chunks = ingest.chunk_text("x" * 1000, chunk_size=100, overlap=10)
    assert len(chunks) > 5  # terminates, and still splits


def test_load_documents_skips_empty_files(tmp_path):
    (tmp_path / "empty.txt").write_text("   ")
    (tmp_path / "ok.txt").write_text("hello world")
    docs = ingest.load_documents(tmp_path)
    assert [d["source"] for d in docs] == ["ok.txt"]


# ------------------------------------------------------------ vector store
def test_retrieval_finds_right_document(store):
    hits = store.retrieve("Who is the chief executive officer of Prairie Health Systems?", k=3)
    assert hits[0]["source"] == "prairie_health_10k.txt"


# --------------------------------------------------------------- predictor
def test_predictor_learns_something_real(data_dir, monkeypatch, tmp_path):
    monkeypatch.setattr(predictor, "MODEL_PATH", tmp_path / "m.joblib")
    monkeypatch.setattr(predictor, "METRICS_PATH", tmp_path / "m.json")
    _, metrics = predictor.train_model(data_dir / "financials.csv")
    assert metrics["r2_test"] > 0.5
    low = predictor.predict_growth(300, 0.0)["predicted_next_growth"]
    high = predictor.predict_growth(300, 0.3)["predicted_next_growth"]
    assert high > low  # more past growth -> more projected growth


# ------------------------------------------------------------------- agent
def test_agent_calls_tool_then_answers(store, monkeypatch, tmp_path):
    monkeypatch.setattr(predictor, "MODEL_PATH", tmp_path / "m.joblib")
    monkeypatch.setattr(predictor, "METRICS_PATH", tmp_path / "m.json")
    # point training at the temp data so nothing is written into the project folder
    monkeypatch.setattr(predictor.config, "DATA_DIR", store.data_dir)
    client = FakeClient([
        tool_response("predict_growth", {"revenue_musd": 300, "prior_growth": 0.1}),
        text_response("Projected growth is about 5%."),
    ])
    out = agent.run_agent("project growth", store, client=client)
    assert out["answer"].startswith("Projected")
    assert [t["tool"] for t in out["trace"]] == ["predict_growth"]
    # the tool result must have been fed back to the model on the 2nd call
    last_msg = client.calls[1]["messages"][-1]
    assert last_msg["content"][0]["type"] == "tool_result"


def test_agent_searches_documents(store):
    client = FakeClient([
        tool_response("search_documents", {"query": "Summit Cloud CEO"}),
        text_response("Priyanka Raman-Hale [summit_cloud_10k.txt]"),
    ])
    out = agent.run_agent("Who runs Summit?", store, client=client)
    assert out["trace"][0]["output"], "search tool returned passages"
    assert "summit_cloud_10k.txt" in {h["source"] for h in out["trace"][0]["output"]}


def test_agent_reports_tool_errors_to_model_instead_of_crashing(store):
    client = FakeClient([
        tool_response("does_not_exist", {}),
        text_response("Sorry, that failed."),
    ])
    out = agent.run_agent("x", store, client=client)
    result = client.calls[1]["messages"][-1]["content"][0]
    assert result["is_error"] is True
    assert out["answer"] == "Sorry, that failed."


def test_agent_stops_after_max_steps(store):
    client = FakeClient([tool_response("search_documents", {"query": "q"}, f"t{i}") for i in range(3)])
    out = agent.run_agent("loop forever", store, client=client, max_steps=3)
    assert "too many" in out["answer"].lower()
    assert out["steps"] == 3


# -------------------------------------------------------------------- eval
def test_keyword_matching_normalises_commas_and_alternatives():
    assert ev.keywords_ok("We employed 1,840 people", ["1840"])
    assert ev.keywords_ok("Plans to open 6 clinics", [["six", "6"]])
    assert not ev.keywords_ok("Revenue was $400 million", ["412.6"])


def test_abstain_detection():
    assert ev.abstained("I don't have enough information to answer that.")
    assert not ev.abstained("The CEO is Marisol Quenby.")


def test_args_close_tolerance():
    assert ev._args_close({"revenue_musd": 412.6, "prior_growth": 0.142}, {"revenue_musd": 412.6, "prior_growth": 0.142})
    # small rounding differences are accepted (within 2%)
    assert ev._args_close({"revenue_musd": 412, "prior_growth": 0.14}, {"revenue_musd": 412.6, "prior_growth": 0.142})
    assert not ev._args_close({"revenue_musd": 100, "prior_growth": 0.142}, {"revenue_musd": 412.6, "prior_growth": 0.142})


def test_retrieval_eval_runs_and_beats_chance(store):
    r = ev.retrieval_eval(store, k=4)
    assert r["n"] == 15
    assert r["hit_at_k"] >= 0.6


def test_eval_set_is_consistent_with_documents(data_dir):
    """Every gold_evidence string must literally exist in its gold document,
    otherwise the retrieval metric would be measuring a typo."""
    for item in json.loads(ev.EVAL_PATH.read_text()):
        if item["type"] == "factual":
            doc = (data_dir / item["gold_source"]).read_text()
            assert ev.norm(item["gold_evidence"]) in ev.norm(doc), item["id"]
