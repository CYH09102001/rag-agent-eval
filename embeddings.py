"""Turn text into vectors.

An embedding model maps text to a list of numbers so that texts with similar
MEANING end up close together. Retrieval = "find the chunk vectors closest to
the question vector".

Two interchangeable embedders:
  - MiniLMEmbedder   : real semantic model (what you'd use for the project)
  - HashingEmbedder  : keyword-overlap only, works offline. Handy for tests and
                       as a deliberately weaker baseline to compare against.
"""
from sklearn.feature_extraction.text import HashingVectorizer

import config


class HashingEmbedder:
    name = "hashing"

    def __init__(self, n_features: int = 2048):
        self._vec = HashingVectorizer(
            n_features=n_features,
            alternate_sign=False,
            norm="l2",
            ngram_range=(1, 2),
            stop_words="english",
        )

    def encode(self, texts: list[str]) -> list[list[float]]:
        return self._vec.transform(texts).toarray().tolist()


class MiniLMEmbedder:
    name = "minilm"

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        # Imported lazily: it's a heavy import and not needed for offline tests.
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts, normalize_embeddings=True).tolist()


def get_embedder(name: str = None):
    name = (name or config.EMBEDDER).lower()
    if name == "hashing":
        return HashingEmbedder()
    if name == "minilm":
        return MiniLMEmbedder()
    raise ValueError(f"Unknown embedder '{name}' (use 'minilm' or 'hashing')")
