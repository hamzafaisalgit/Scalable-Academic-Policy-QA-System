"""Smoke tests for the ingestion module."""

import tempfile
from pathlib import Path

import pytest

from ingestion import (
    ChunkRecord,
    PageRecord,
    chunk_pages,
    clean_extracted_text,
    tokenize,
    _normalize_whitespace,
    _is_noise_line,
    _looks_like_heading,
    _split_long_text,
    read_text_pages,
    ingest_document,
    ingest_corpus,
    save_corpus,
)


class TestTokenization:
    """Smoke tests for tokenization functionality."""

    def test_tokenize_simple_text(self):
        """Test basic tokenization."""
        result = tokenize("Hello World")
        assert result == ["hello", "world"]

    def test_tokenize_with_numbers(self):
        """Test tokenization with numbers."""
        result = tokenize("Test 123 example")
        assert result == ["test", "123", "example"]

    def test_tokenize_with_hyphens(self):
        """Test tokenization with hyphenated words."""
        result = tokenize("well-known test")
        assert result == ["well-known", "test"]

    def test_tokenize_empty_string(self):
        """Test tokenization of empty string."""
        result = tokenize("")
        assert result == []

    def test_tokenize_special_characters(self):
        """Test tokenization removes special characters."""
        result = tokenize("Hello, World! @#$")
        assert "hello" in result and "world" in result


class TestTextCleaning:
    """Smoke tests for text cleaning functionality."""

    def test_normalize_whitespace_basic(self):
        """Test basic whitespace normalization."""
        result = _normalize_whitespace("Hello   World")
        assert result == "Hello World"

    def test_normalize_whitespace_newlines(self):
        """Test newline normalization."""
        result = _normalize_whitespace("Hello\n\n\nWorld")
        assert result == "Hello\n\nWorld"

    def test_normalize_whitespace_null_bytes(self):
        """Test null byte removal."""
        result = _normalize_whitespace("Hello\x00World")
        assert "\x00" not in result

    def test_clean_extracted_text_basic(self):
        """Test basic text cleaning."""
        raw = "This is a test document.\nWith multiple lines.\n\nAnd paragraphs."
        result = clean_extracted_text(raw)
        assert "test document" in result.lower()
        assert "multiple lines" in result.lower()

    def test_clean_extracted_text_with_noise(self):
        """Test cleaning removes noise lines."""
        raw = "Header\n\n1\n\n2\n\nActual content here"
        result = clean_extracted_text(raw)
        assert "actual content" in result.lower()

    def test_clean_extracted_text_empty(self):
        """Test cleaning handles empty strings."""
        result = clean_extracted_text("")
        assert result == ""


class TestHeadingDetection:
    """Smoke tests for heading detection."""

    def test_heading_detection_chapter(self):
        """Test chapter heading detection."""
        assert _looks_like_heading("Chapter 1")

    def test_heading_detection_all_caps(self):
        """Test all-caps heading detection."""
        assert _looks_like_heading("IMPORTANT INFORMATION")

    def test_heading_detection_with_colon(self):
        """Test heading detection with colon."""
        assert _looks_like_heading("Section Title:")

    def test_not_heading_long_text(self):
        """Test long text is not detected as heading."""
        assert not _looks_like_heading("This is a very long text that should not be considered a heading because it is too long")

    def test_noise_detection_numbers(self):
        """Test noise detection for numbers."""
        assert _is_noise_line("123")

    def test_noise_detection_roman_numerals(self):
        """Test noise detection for roman numerals."""
        assert _is_noise_line("viii")


class TestTextSplitting:
    """Smoke tests for text splitting functionality."""

    def test_split_long_text_short_input(self):
        """Test splitting text shorter than max."""
        text = "This is short"
        result = _split_long_text(text, max_words=100)
        assert len(result) == 1
        assert result[0] == text

    def test_split_long_text_needs_split(self):
        """Test splitting text longer than max."""
        text = " ".join(["word"] * 300)
        result = _split_long_text(text, max_words=100)
        assert len(result) > 1

    def test_split_maintains_content(self):
        """Test that splitting preserves all content."""
        text = " ".join(["word"] * 150)
        result = _split_long_text(text, max_words=100)
        combined = " ".join(result)
        assert "word" in combined


class TestPageRecord:
    """Smoke tests for PageRecord."""

    def test_page_record_creation(self):
        """Test creating a PageRecord."""
        record = PageRecord(
            source="test-doc",
            source_path="/path/to/test.pdf",
            page_number=1,
            raw_text="Raw text here",
            clean_text="Raw text here",
        )
        assert record.source == "test-doc"
        assert record.page_number == 1

    def test_page_record_to_dict(self):
        """Test converting PageRecord to dict."""
        record = PageRecord(
            source="test-doc",
            source_path="/path/to/test.pdf",
            page_number=1,
            raw_text="Raw",
            clean_text="Clean",
        )
        result = record.to_dict()
        assert result["source"] == "test-doc"
        assert result["page_number"] == 1


class TestChunkRecord:
    """Smoke tests for ChunkRecord."""

    def test_chunk_record_creation(self):
        """Test creating a ChunkRecord."""
        record = ChunkRecord(
            chunk_id="test-chunk-001",
            chunk_index=0,
            source="test-doc",
            source_path="/path/to/test.pdf",
            text="This is chunk text",
            clean_text="this is chunk text",
            page_start=1,
            page_end=1,
            section="Introduction",
            word_count=4,
        )
        assert record.chunk_id == "test-chunk-001"
        assert record.word_count == 4

    def test_chunk_record_to_dict(self):
        """Test converting ChunkRecord to dict."""
        record = ChunkRecord(
            chunk_id="test-chunk-001",
            chunk_index=0,
            source="test-doc",
            source_path="/path/to/test.pdf",
            text="Text",
            clean_text="text",
            page_start=1,
            page_end=1,
            section="Intro",
            word_count=1,
        )
        result = record.to_dict()
        assert "metadata" in result
        assert result["metadata"]["chunk_id"] == "test-chunk-001"


class TestChunkPages:
    """Smoke tests for page chunking."""

    def test_chunk_pages_simple(self):
        """Test chunking simple pages."""
        pages = [
            PageRecord(
                source="test",
                source_path="/test.pdf",
                page_number=1,
                raw_text="Word " * 300,
                clean_text="word " * 300,
            )
        ]
        result = chunk_pages(pages, min_words=50, max_words=100)
        assert len(result) > 0
        assert all(isinstance(chunk, ChunkRecord) for chunk in result)

    def test_chunk_pages_empty(self):
        """Test chunking empty pages list."""
        result = chunk_pages([])
        assert result == []

    def test_chunk_pages_preserves_metadata(self):
        """Test that chunking preserves metadata."""
        pages = [
            PageRecord(
                source="test",
                source_path="/test.pdf",
                page_number=1,
                raw_text="Word " * 300,
                clean_text="word " * 300,
            )
        ]
        result = chunk_pages(pages)
        assert result[0].source == "test"
        assert result[0].source_path == "/test.pdf"


class TestTextPageReading:
    """Smoke tests for reading text pages."""

    def test_read_text_pages_basic(self):
        """Test reading a text file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("This is test content.\nWith multiple lines.")
            temp_path = f.name

        try:
            result = read_text_pages(temp_path)
            assert len(result) == 1
            assert result[0].source == Path(temp_path).stem
            assert "test content" in result[0].clean_text.lower()
        finally:
            Path(temp_path).unlink()

    def test_read_text_pages_returns_page_record(self):
        """Test that reading text returns PageRecord objects."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Test content")
            temp_path = f.name

        try:
            result = read_text_pages(temp_path)
            assert isinstance(result[0], PageRecord)
            assert result[0].page_number == 1
        finally:
            Path(temp_path).unlink()


class TestDocumentIngestion:
    """Smoke tests for document ingestion."""

    def test_ingest_document_text(self):
        """Test ingesting a text document."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Chapter 1\n\n" + "word " * 300)
            temp_path = f.name

        try:
            result = ingest_document(temp_path)
            assert "source" in result
            assert "chunks" in result
            assert "texts" in result
            assert "metadata" in result
            assert len(result["chunks"]) > 0
        finally:
            Path(temp_path).unlink()

    def test_ingest_document_returns_dict(self):
        """Test that ingest_document returns correct structure."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Test document with content " * 50)
            temp_path = f.name

        try:
            result = ingest_document(temp_path)
            assert isinstance(result, dict)
            assert "source_path" in result
            assert "pages" in result
            assert "chunks" in result
        finally:
            Path(temp_path).unlink()


class TestCorpusSaving:
    """Smoke tests for corpus operations."""

    def test_save_corpus_creates_file(self):
        """Test that save_corpus creates output file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir) / "data"
            data_dir.mkdir()

            # Create a test document
            test_file = data_dir / "test.txt"
            test_file.write_text("Test content " * 100)

            output_path = Path(temp_dir) / "corpus.json"
            result = save_corpus(output_path, data_dir=data_dir)

            assert result.exists()
            assert result.is_file()

    def test_save_corpus_valid_json(self):
        """Test that saved corpus has valid JSON structure."""
        import json

        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir) / "data"
            data_dir.mkdir()

            test_file = data_dir / "test.txt"
            test_file.write_text("Test data " * 100)

            output_path = Path(temp_dir) / "corpus.json"
            save_corpus(output_path, data_dir=data_dir)

            # Try to load the JSON
            content = json.loads(output_path.read_text())
            assert "documents" in content
            assert "chunks" in content
            assert "texts" in content


class TestErrorHandling:
    """Smoke tests for error handling."""

    def test_chunk_pages_invalid_min_max(self):
        """Test chunking with invalid parameters."""
        pages = [
            PageRecord(
                source="test",
                source_path="/test.pdf",
                page_number=1,
                raw_text="Test",
                clean_text="test",
            )
        ]
        with pytest.raises(ValueError):
            chunk_pages(pages, min_words=100, max_words=50)

    def test_chunk_pages_negative_size(self):
        """Test chunking with negative size."""
        pages = [
            PageRecord(
                source="test",
                source_path="/test.pdf",
                page_number=1,
                raw_text="Test",
                clean_text="test",
            )
        ]
        with pytest.raises(ValueError):
            chunk_pages(pages, min_words=-1, max_words=100)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
