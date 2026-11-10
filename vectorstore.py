"""Store chunk vectors in ChromaDB and search them.

Each (embedder, chunk_size, overlap) combo gets its OWN collection. Vectors
from different embedders (or built from different chunks) can't be mixed, and
this lets compare_configs.py test several setups side by side.
"""
import re

import chromadb

import config
from embeddings import get_embedder
from ingest import build_chunks

BATCH = 64


class VectorStore:
    def __init__(self, embedder=None, chunk_size=None, overlap=None, path=None, data_dir=None):
        self.embedder = embedder or get_embedder()
        self.chunk_size = chunk_size or config.CHUNK_SIZE
        self.overlap = config.CHUNK_OVERLAP if overlap is None else overlap
        self.data_dir = data_dir
        self._client = chromadb.PersistentClient(path=str(path or config.CHROMA_DIR))
        raw = f"docs_{self.embedder.name}_{self.chunk_size}_{self.overlap}"
        self.collection_name = re.sub(r"[^a-zA-Z0-9._-]", "_", raw)
        # cosine distance matches how our embedders are normalised
        self.collection = self._client.get_or_create_collection(
            self.collection_name, metadata={"hnsw:space": "cosine"}
        )

    def count(self) -> int:
        return self.collection.count()

    def build_index(self, force: bool = False) -> int:
        """(Re)build the index from the data folder. Skips work if it exists."""
        if self.count() > 0 and not force:
            return self.count()
        if force:
            self._client.delete_collection(self.collection_name)
            self.collection = self._client.get_or_create_collection(
                self.collection_name, metadata={"hnsw:space": "cosine"}
            )
        chunks = build_chunks(self.data_dir, self.chunk_size, self.overlap)
        if not chunks:
            raise RuntimeError("No chunks produced. Is the data/ folder empty?")
        for i in range(0, len(chunks), BATCH):
            batch = chunks[i : i + BATCH]
            self.collection.add(
                ids=[c["id"] for c in batch],
                documents=[c["text"] for c in batch],
                embeddings=self.embedder.encode([c["text"] for c in batch]),
                metadatas=[{"source": c["source"]} for c in batch],
            )
        return self.count()

    def retrieve(self, query: str, k: int = None) -> list[dict]:
        k = k or config.TOP_K
        res = self.collection.query(
            query_embeddings=self.embedder.encode([query]),
            n_results=min(k, max(self.count(), 1)),
        )
        return [
            {
                "id": cid,
                "text": doc,
                "source": meta["source"],
                "score": round(1 - dist, 4),  # cosine similarity: higher = closer
            }
            for cid, doc, meta, dist in zip(
                res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
            )
        ]
