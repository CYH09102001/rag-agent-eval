"""Build (or rebuild) the vector index from the files in data/.

Run:  python build_index.py            # builds if missing
      python build_index.py --force    # wipe and rebuild (after changing data or chunk settings)
"""
import argparse

import config
from vectorstore import VectorStore

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    store = VectorStore()
    n = store.build_index(force=args.force)
    print(f"Index '{store.collection_name}' ready with {n} chunks "
          f"(embedder={config.EMBEDDER}, chunk_size={config.CHUNK_SIZE}).")
