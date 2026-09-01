import pytest

from bmo_rag.processing.chunking import chunk_text


def test_chunk_text_splits_with_overlap() -> None:
    chunks = chunk_text("abcdefghij", chunk_size=4, overlap=1)

    assert chunks == ["abcd", "defg", "ghij", "j"]


def test_chunk_text_rejects_invalid_overlap() -> None:
    with pytest.raises(ValueError):
        chunk_text("hello", chunk_size=5, overlap=5)
