"""The 'break it on purpose' experiment.

Builds an index for every combination of embedder x chunk size x top-k and
measures retrieval quality for each. Offline and free (no API calls), so run
it as often as you like. The table it prints is your evidence for statements
like "I chose 800-character chunks because ...".

Run:  python compare_configs.py
      python compare_configs.py --embedders hashing minilm --chunk-sizes 200 500 800 1500
"""
import argparse
import tempfile

import pandas as pd

import config
from embeddings import get_embedder
from eval import retrieval_eval
from vectorstore import VectorStore


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--embedders", nargs="+", default=["hashing", "minilm"])
    p.add_argument("--chunk-sizes", nargs="+", type=int, default=[200, 400, 800, 1500])
    p.add_argument("--ks", nargs="+", type=int, default=[1, 2, 4])
    args = p.parse_args()

    rows = []
    with tempfile.TemporaryDirectory() as tmp:  # throwaway index, keeps chroma_db/ clean
        for name in args.embedders:
            try:
                embedder = get_embedder(name)
            except Exception as exc:
                print(f"Skipping embedder '{name}': {exc}")
                continue
            for size in args.chunk_sizes:
                overlap = min(config.CHUNK_OVERLAP, size // 4)
                store = VectorStore(embedder=embedder, chunk_size=size, overlap=overlap, path=tmp)
                n_chunks = store.build_index(force=True)
                for k in args.ks:
                    r = retrieval_eval(store, k=k)
                    rows.append({"embedder": name, "chunk_size": size, "overlap": overlap, "chunks": n_chunks,
                                 "k": k, "hit_at_k": r["hit_at_k"], "mrr": r["mrr"]})

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    config.RESULTS_DIR.mkdir(exist_ok=True)
    out = config.RESULTS_DIR / "config_comparison.csv"
    df.to_csv(out, index=False)
    best = df.sort_values(["hit_at_k", "mrr"], ascending=False).iloc[0]
    print(f"\nSaved {out}\nBest row: {best.to_dict()}")


if __name__ == "__main__":
    main()
