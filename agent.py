"""Agentic RAG: Claude decides WHEN to search the documents and WHEN to call
the prediction model, possibly several times, before answering.

Compare with rag.py, which always retrieves exactly once. The agent can:
  - skip retrieval for a pure calculation question
  - search, read the numbers, then pass them to the predictor (multi-step)
  - search again with a better query if the first one missed

The loop is the whole idea of an "agent": call the model -> if it asks for a
tool, run the tool and feed the result back -> repeat until it answers.
"""
import json
import sys
import time

import config
from llm import get_client
from predictor import predict_growth
from vectorstore import VectorStore

SYSTEM = f"""You are an analyst assistant for a small set of company annual reports.
Tools:
- search_documents: look up facts in the reports. Use it for any fact about a company.
- predict_growth: a trained model that projects next-year revenue growth.
  Inputs must be real numbers: revenue in USD millions and last year's growth as a decimal (14.2% -> 0.142).
Rules:
1. Never invent facts. If search results don't contain the answer, say: "{config.ABSTAIN_PHRASE} to answer that."
2. Cite the source file in brackets after each fact, e.g. [summit_cloud_10k.txt].
3. If you used predict_growth, say the number comes from a simple model trained on synthetic data."""

TOOLS = [
    {
        "name": "search_documents",
        "description": "Semantic search over the company annual reports. Returns the most relevant text passages with their source file.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "What to look for, phrased as a short search query."}},
            "required": ["query"],
        },
    },
    {
        "name": "predict_growth",
        "description": "Projects NEXT year's revenue growth from current revenue and last year's growth.",
        "input_schema": {
            "type": "object",
            "properties": {
                "revenue_musd": {"type": "number", "description": "Revenue in USD millions, e.g. 412.6"},
                "prior_growth": {"type": "number", "description": "Last year's growth as a decimal, e.g. 0.142 for 14.2%"},
            },
            "required": ["revenue_musd", "prior_growth"],
        },
    },
]


def execute_tool(name: str, args: dict, store: VectorStore):
    """Run one tool call. Returns a JSON-serialisable result."""
    if name == "search_documents":
        hits = store.retrieve(args["query"])
        return [{"source": h["source"], "text": h["text"]} for h in hits]
    if name == "predict_growth":
        return predict_growth(float(args["revenue_musd"]), float(args["prior_growth"]))
    raise ValueError(f"Unknown tool: {name}")


def run_agent(question: str, store: VectorStore, client=None, model: str = None, max_steps: int = 6) -> dict:
    client = client or get_client()
    model = model or config.MODEL
    messages = [{"role": "user", "content": question}]
    trace, start = [], time.time()

    for step in range(1, max_steps + 1):
        response = client.messages.create(
            model=model, max_tokens=1000, system=SYSTEM, tools=TOOLS, messages=messages
        )
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            text = "".join(b.text for b in response.content if b.type == "text")
            return {"question": question, "answer": text, "trace": trace, "steps": step,
                    "latency_s": round(time.time() - start, 2)}

        results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            try:
                output, is_error = execute_tool(block.name, block.input, store), False
            except Exception as exc:  # report the failure to the model so it can recover
                output, is_error = {"error": str(exc)}, True
            trace.append({"step": step, "tool": block.name, "input": dict(block.input), "output": output})
            results.append({"type": "tool_result", "tool_use_id": block.id,
                            "content": json.dumps(output), "is_error": is_error})
        messages.append({"role": "user", "content": results})

    # Safety net: never loop forever (and never silently burn API credits)
    return {"question": question, "answer": "Stopped: too many tool-use steps without a final answer.",
            "trace": trace, "steps": max_steps, "latency_s": round(time.time() - start, 2)}


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "Use the growth predictor on Summit Cloud Software's fiscal 2025 revenue and growth."
    store = VectorStore()
    store.build_index()
    out = run_agent(q, store)
    for t in out["trace"]:
        print(f"[step {t['step']}] {t['tool']}({t['input']})")
    print("\n" + out["answer"])
