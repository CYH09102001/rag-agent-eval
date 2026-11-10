import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import generate_sample_data  # noqa: E402
from embeddings import HashingEmbedder  # noqa: E402
from vectorstore import VectorStore  # noqa: E402


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory):
    """A throwaway copy of the sample data so tests never touch your real data/."""
    d = tmp_path_factory.mktemp("data")
    for name, text in generate_sample_data.DOCS.items():
        (d / name).write_text(text, encoding="utf-8")
    generate_sample_data.make_financials().to_csv(d / "financials.csv", index=False)
    return d


@pytest.fixture(scope="session")
def store(data_dir, tmp_path_factory):
    s = VectorStore(embedder=HashingEmbedder(), chunk_size=400, overlap=80,
                    path=tmp_path_factory.mktemp("chroma"), data_dir=data_dir)
    s.build_index()
    return s


class FakeClient:
    """Stands in for the Anthropic client: returns scripted responses, so the
    agent loop can be tested with no API key and no cost."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        # snapshot the message list: the agent keeps appending to the same list,
        # and we want to inspect what the model saw at THIS call
        self.calls.append({**kwargs, "messages": list(kwargs["messages"])})
        return self._responses.pop(0)


def text_response(text):
    return SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="text", text=text)])


def tool_response(name, args, tool_id="tool_1"):
    block = SimpleNamespace(type="tool_use", id=tool_id, name=name, input=args)
    return SimpleNamespace(stop_reason="tool_use", content=[block])
