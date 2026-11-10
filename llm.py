"""Thin wrapper around the Claude API (the 'G' in RAG: generation)."""
import os

import config


def get_client():
    """Create the Anthropic client, with a friendly error if the key is missing."""
    from anthropic import Anthropic

    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and paste "
            "your key (get one at console.anthropic.com)."
        )
    return Anthropic(api_key=key)


def ask_claude(prompt: str, system: str = None, model: str = None, max_tokens: int = 800, client=None) -> str:
    client = client or get_client()
    kwargs = {
        "model": model or config.MODEL,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system:
        kwargs["system"] = system
    response = client.messages.create(**kwargs)
    return "".join(b.text for b in response.content if b.type == "text")


if __name__ == "__main__":
    print(ask_claude("What is retrieval-augmented generation? Two sentences."))
