"""Plain RAG: retrieve first, then answer ONLY from what was retrieved.

This is the baseline. agent.py is the more flexible version where the model
itself decides when to search or call a tool. Comparing the two is a good
interview talking point.
"""
import sys
import time

import config
from llm import ask_claude
from vectorstore import VectorStore

SYSTEM = f"""You answer questions about company filings using ONLY the provided context.
Rules:
1. Every factual claim must be followed by its source in brackets, e.g. [lakeshore_robotics_10k.txt].
2. If the context does not contain the answer, reply exactly: "{config.ABSTAIN_PHRASE} to answer that."
3. Never use outside knowledge. Do not guess."""


def format_context(chunks: list[dict]) -> str:
    return "\n\n".join(f"[{c['source']}]\n{c['text']}" for c in chunks)


def rag_answer(question: str, store: VectorStore, client=None, k: int = None, model: str = None) -> dict:
    start = time.time()
    chunks = store.retrieve(question, k=k)
    prompt = f"Context:\n{format_context(chunks)}\n\nQuestion: {question}"
    answer = ask_claude(prompt, system=SYSTEM, model=model, client=client)
    return {
        "question": question,
        "answer": answer,
        "chunks": chunks,
        "sources": sorted({c["source"] for c in chunks}),
        "latency_s": round(time.time() - start, 2),
    }


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "What are the main risk factors for Lakeshore Robotics?"
    store = VectorStore()
    store.build_index()
    result = rag_answer(q, store)
    print(result["answer"])
    print("\nSources retrieved:", result["sources"])
