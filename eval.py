"""Evaluation: how do we KNOW this system works?

Three layers, cheapest first:

1. retrieval  (offline, free): did the search return the chunk containing the
   answer? Metrics: hit@k and MRR. Needs no API key.
2. rag        (API): is the generated answer correct, grounded in the
   retrieved text, and does the model refuse unanswerable questions?
3. agent      (API): did the agent pick the right tools with the right inputs,
   and is its final answer correct?

Run:  python eval.py --mode retrieval
      python eval.py --mode all
Results are written to eval_results/ (JSON + a markdown summary).
"""
import argparse
import json
import re
import statistics
import time

import config
from vectorstore import VectorStore

EVAL_PATH = config.ROOT / "eval_set.json"


def load_eval_set() -> list[dict]:
    return json.loads(EVAL_PATH.read_text(encoding="utf-8"))


def norm(text: str) -> str:
    """Lowercase, drop commas and extra whitespace so '1,840' matches '1840'."""
    return re.sub(r"\s+", " ", text.lower().replace(",", "")).strip()


# --------------------------------------------------------------------------
# Layer 1: retrieval
# --------------------------------------------------------------------------
def retrieval_eval(store: VectorStore, k: int = None, eval_set=None) -> dict:
    k = k or config.TOP_K
    rows = []
    for item in (eval_set or load_eval_set()):
        if item["type"] != "factual":
            continue
        hits = store.retrieve(item["question"], k=k)
        rank = None
        for i, h in enumerate(hits, start=1):
            if h["source"] == item["gold_source"] and norm(item["gold_evidence"]) in norm(h["text"]):
                rank = i
                break
        rows.append({"id": item["id"], "question": item["question"], "rank": rank})
    n = len(rows)
    return {
        "k": k,
        "n": n,
        "hit_at_k": round(sum(r["rank"] is not None for r in rows) / n, 3),
        "mrr": round(sum(1 / r["rank"] for r in rows if r["rank"]) / n, 3),
        "misses": [r["id"] for r in rows if r["rank"] is None],
        "rows": rows,
    }


# --------------------------------------------------------------------------
# Layer 2/3: generation + agent (need an API key)
# --------------------------------------------------------------------------
def keywords_ok(answer: str, keywords: list) -> bool:
    a = norm(answer)
    for kw in keywords:
        options = kw if isinstance(kw, list) else [kw]
        if not any(norm(o) in a for o in options):
            return False
    return True


def abstained(answer: str) -> bool:
    return norm(config.ABSTAIN_PHRASE) in norm(answer)


JUDGE_PROMPT = """You are grading whether an ANSWER is fully supported by the CONTEXT.
Reply with JSON only: {{"supported": true|false, "reason": "<one sentence>"}}
"supported" is true only if every factual claim in the answer can be found in the context.

CONTEXT:
{context}

ANSWER:
{answer}"""


def judge_faithfulness(answer: str, context: str, client) -> dict:
    """LLM-as-judge: ask a model whether the answer is backed by the context.
    This is the same idea RAGAS's 'faithfulness' metric uses, in ~10 lines."""
    from llm import ask_claude

    raw = ask_claude(JUDGE_PROMPT.format(context=context, answer=answer), model=config.JUDGE_MODEL,
                     max_tokens=200, client=client)
    match = re.search(r"\{.*\}", raw, re.S)
    try:
        return json.loads(match.group(0))
    except Exception:
        return {"supported": None, "reason": f"unparseable judge output: {raw[:80]}"}


def rag_eval(store: VectorStore, client, k: int = None, judge: bool = True, eval_set=None) -> dict:
    from rag import format_context, rag_answer

    rows = []
    for item in (eval_set or load_eval_set()):
        if item["type"] == "calc":
            continue  # plain RAG has no calculator; the agent eval covers these
        out = rag_answer(item["question"], store, client=client, k=k)
        row = {"id": item["id"], "type": item["type"], "answer": out["answer"], "latency_s": out["latency_s"]}
        if item["type"] == "unanswerable":
            row["correct"] = abstained(out["answer"])
        else:
            row["correct"] = keywords_ok(out["answer"], item["answer_keywords"]) and not abstained(out["answer"])
        if judge and not abstained(out["answer"]):
            row["faithful"] = judge_faithfulness(out["answer"], format_context(out["chunks"]), client).get("supported")
        rows.append(row)
    return summarise_generation(rows)


def agent_eval(store: VectorStore, client, eval_set=None) -> dict:
    from agent import run_agent

    rows = []
    for item in (eval_set or load_eval_set()):
        out = run_agent(item["question"], store, client=client)
        used = [t["tool"] for t in out["trace"]]
        row = {"id": item["id"], "type": item["type"], "answer": out["answer"], "tools": used,
               "steps": out["steps"], "latency_s": out["latency_s"]}
        if item["type"] == "calc":
            called = [t for t in out["trace"] if t["tool"] == "predict_growth"]
            tools_ok = set(item["expected_tools"]) <= set(used)
            args_ok = bool(called) and _args_close(called[-1]["input"], item["expected_args"])
            row.update(tool_choice_ok=tools_ok, args_ok=args_ok, correct=tools_ok and args_ok)
        elif item["type"] == "unanswerable":
            row["correct"] = abstained(out["answer"])
        else:
            row["correct"] = keywords_ok(out["answer"], item["answer_keywords"]) and not abstained(out["answer"])
        rows.append(row)
    return summarise_generation(rows)


def _args_close(actual: dict, expected: dict, tol: float = 0.02) -> bool:
    """Compare tool inputs with a small relative tolerance (e.g. 0.142 vs 0.14)."""
    for key, want in expected.items():
        got = actual.get(key)
        if got is None or abs(float(got) - want) > tol * max(abs(want), 1e-9):
            return False
    return True


def summarise_generation(rows: list[dict]) -> dict:
    def rate(subset, field="correct"):
        vals = [r[field] for r in subset if r.get(field) is not None]
        return round(sum(vals) / len(vals), 3) if vals else None

    by_type = {}
    for t in sorted({r["type"] for r in rows}):
        by_type[t] = rate([r for r in rows if r["type"] == t])
    return {
        "n": len(rows),
        "accuracy": rate(rows),
        "accuracy_by_type": by_type,
        "faithfulness": rate(rows, "faithful"),
        "median_latency_s": round(statistics.median(r["latency_s"] for r in rows), 2),
        "failures": [r["id"] for r in rows if not r["correct"]],
        "rows": rows,
    }


# --------------------------------------------------------------------------
def write_report(results: dict) -> None:
    config.RESULTS_DIR.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    (config.RESULTS_DIR / f"results-{stamp}.json").write_text(json.dumps(results, indent=2))

    lines = ["# Evaluation summary", "", f"Embedder: `{config.EMBEDDER}` | chunk size: {config.CHUNK_SIZE} | "
             f"overlap: {config.CHUNK_OVERLAP} | top-k: {config.TOP_K} | model: `{config.MODEL}`", ""]
    r = results.get("retrieval")
    if r:
        lines += ["## Retrieval", f"- hit@{r['k']}: **{r['hit_at_k']}** over {r['n']} factual questions",
                  f"- MRR: **{r['mrr']}**", f"- misses: {', '.join(r['misses']) or 'none'}", ""]
    for key, title in (("rag", "Plain RAG"), ("agent", "Agent")):
        g = results.get(key)
        if g:
            lines += [f"## {title}", f"- accuracy: **{g['accuracy']}** over {g['n']} questions",
                      f"- by type: {g['accuracy_by_type']}"]
            if g.get("faithfulness") is not None:
                lines.append(f"- faithfulness (LLM judge): **{g['faithfulness']}**")
            lines += [f"- median latency: {g['median_latency_s']}s", f"- failures: {', '.join(g['failures']) or 'none'}", ""]
    (config.RESULTS_DIR / "latest_summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["retrieval", "rag", "agent", "all"], default="retrieval")
    parser.add_argument("--no-judge", action="store_true", help="skip the LLM faithfulness judge (saves API calls)")
    args = parser.parse_args()

    store = VectorStore()
    store.build_index()
    results = {}
    if args.mode in ("retrieval", "all"):
        results["retrieval"] = retrieval_eval(store)
    if args.mode in ("rag", "agent", "all"):
        from llm import get_client

        client = get_client()
        if args.mode in ("rag", "all"):
            results["rag"] = rag_eval(store, client, judge=not args.no_judge)
        if args.mode in ("agent", "all"):
            results["agent"] = agent_eval(store, client)
    write_report(results)


if __name__ == "__main__":
    main()
