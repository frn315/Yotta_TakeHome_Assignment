"""Two chunking strategies. Chunks remember their parent passage so retrieval can
be scored against passage-level relevance labels."""
from __future__ import annotations

from dataclasses import dataclass

from common.text import split_sentences


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    passage_id: str
    text: str


def fixed_size(text: str, size: int = 40, overlap: int = 10) -> list[str]:
    """Sliding window of `size` words with `overlap` words shared between neighbours."""
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("need size > 0 and 0 <= overlap < size")
    words = text.split()
    if len(words) <= size:
        return [text] if words else []
    out, step = [], size - overlap
    for start in range(0, len(words), step):
        out.append(" ".join(words[start:start + size]))
        if start + size >= len(words):
            break
    return out


def sentence_boundary(text: str, max_words: int = 40) -> list[str]:
    """Pack whole sentences into chunks of at most `max_words` words (never cuts mid-sentence;
    a single sentence longer than the budget becomes its own chunk)."""
    out: list[str] = []
    current: list[str] = []
    count = 0
    for sent in split_sentences(text):
        n = len(sent.split())
        if current and count + n > max_words:
            out.append(" ".join(current))
            current, count = [], 0
        current.append(sent)
        count += n
    if current:
        out.append(" ".join(current))
    return out


def chunk_corpus(passages: dict[str, str], strategy: str, size: int = 40, overlap: int = 10) -> list[Chunk]:
    chunks: list[Chunk] = []
    for pid, text in passages.items():
        if strategy == "fixed":
            parts = fixed_size(text, size, overlap)
        elif strategy == "sentence":
            parts = sentence_boundary(text, size)
        else:
            raise ValueError(f"unknown chunking strategy '{strategy}' (use 'fixed' or 'sentence')")
        chunks.extend(Chunk(f"{pid}#{i}", pid, part) for i, part in enumerate(parts))
    return chunks
