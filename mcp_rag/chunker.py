"""Heading-aware markdown chunker (stdlib only).

Splits a markdown document into overlapping chunks, tracking the heading
hierarchy so each chunk carries a `heading_path` like
`"Proyecto > Setup > Hardware"`. Obsidian YAML frontmatter is skipped and
sections with no body text produce no chunk.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from mcp_rag.errors import RagError

# 16 MiB guard (CHUNK-002).
MAX_FILE_BYTES = 16 * 1024 * 1024

# Token estimate heuristic: ~4 chars per token (stdlib only, no tokenizer).
CHARS_PER_TOKEN = 4

_HEADING_RE = re.compile(r"^(#{1,3})\s+(.*)$")
_FRONTMATTER_RE = re.compile(r"^---\s*\n.*?\n---\s*\n?", re.DOTALL)


@dataclass
class Chunk:
    path: str
    heading_path: str
    text: str
    mtime: float
    md5: str


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


class Chunker:
    """Splits markdown text into heading-aware, overlapping chunks."""

    def __init__(self, chunk_size: int = 800, overlap: int = 100):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, path: str, text: str, mtime: float, md5: str) -> list[Chunk]:
        if len(text.encode("utf-8")) > MAX_FILE_BYTES:
            raise RagError(f"file too large (>16MB): {path}")

        text = _FRONTMATTER_RE.sub("", text, count=1)

        # Split into (level, heading, body) sections, tracking heading stack.
        sections: list[tuple[int, str, str]] = []
        stack: list[tuple[int, str]] = []  # (level, heading) ancestors
        cur_level, cur_heading, cur_body = 0, "", []

        def flush() -> None:
            nonlocal cur_body
            if cur_heading:
                sections.append((cur_level, cur_heading, "".join(cur_body).strip()))
            cur_body = []

        for line in text.splitlines():
            m = _HEADING_RE.match(line)
            if m:
                flush()
                level = len(m.group(1))
                heading = m.group(2).strip()
                while stack and stack[-1][0] >= level:
                    stack.pop()
                stack.append((level, heading))
                cur_level, cur_heading = level, heading
            else:
                cur_body.append(line + "\n")
        flush()

        # Rebuild heading paths per section from the stack state at each section.
        paths: dict[int, str] = {}
        stack = []
        for idx, (level, heading, _body) in enumerate(sections):
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, heading))
            paths[idx] = " > ".join(h for _, h in stack)

        chunks: list[Chunk] = []
        for idx, (level, heading, body) in enumerate(sections):
            if not body:
                continue  # heading-only section -> no chunk
            for piece in _split_body(body, self.chunk_size, self.overlap):
                chunks.append(
                    Chunk(
                        path=path,
                        heading_path=paths[idx],
                        text=piece,
                        mtime=mtime,
                        md5=md5,
                    )
                )
        return chunks


def _split_body(body: str, chunk_size: int, overlap: int) -> list[str]:
    """Split body text into overlapping chunks by token estimate."""
    words = body.split()
    if not words:
        return []
    target_words = max(1, chunk_size // 2)  # ~2 words per token heuristic
    overlap_words = max(0, overlap // 2)
    pieces: list[str] = []
    i = 0
    while i < len(words):
        end = min(i + target_words, len(words))
        piece = " ".join(words[i:end])
        if piece.strip():
            pieces.append(piece)
        if end >= len(words):
            break
        i = max(i + target_words - overlap_words, i + 1)
    return pieces
