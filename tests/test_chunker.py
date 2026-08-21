import pytest

from mcp_rag.chunker import Chunker, MAX_FILE_BYTES
from mcp_rag.errors import RagError


def test_nested_headings_produce_correct_heading_path():
    text = (
        "# Proyecto\n"
        "texto A\n"
        "## Setup\n"
        "texto B\n"
        "### Hardware\n"
        "texto C\n"
    )
    chunks = Chunker().chunk("a.md", text, mtime=1.0, md5="m")
    paths = [c.heading_path for c in chunks]
    assert "Proyecto" in paths
    assert "Proyecto > Setup" in paths
    assert "Proyecto > Setup > Hardware" in paths


def test_frontmatter_skipped():
    text = (
        "---\n"
        "title: Nota\n"
        "tags: [x]\n"
        "---\n"
        "# Titulo\n"
        "cuerpo real\n"
    )
    chunks = Chunker().chunk("a.md", text, mtime=1.0, md5="m")
    assert len(chunks) == 1
    assert "title" not in chunks[0].text
    assert "cuerpo real" in chunks[0].text


def test_empty_and_heading_only_sections_produce_no_chunk():
    text = "# Solo\n\n## Vacio\n\n### Sin cuerpo\n"
    chunks = Chunker().chunk("a.md", text, mtime=1.0, md5="m")
    assert chunks == []


def test_overlap_present_between_consecutive_chunks():
    body = "palabra " * 2000  # long body -> multiple chunks
    text = f"# H\n{body}"
    chunks = Chunker(chunk_size=800, overlap=100).chunk("a.md", text, mtime=1.0, md5="m")
    assert len(chunks) >= 2
    # Overlap: last words of chunk 0 appear in chunk 1.
    tail = chunks[0].text.split()[-5:]
    head = chunks[1].text.split()[:5]
    assert any(w in head for w in tail)


def test_oversized_file_raises_ragerror():
    big = "x" * (MAX_FILE_BYTES + 1)
    with pytest.raises(RagError):
        Chunker().chunk("big.md", big, mtime=1.0, md5="m")
