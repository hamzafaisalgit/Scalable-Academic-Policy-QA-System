from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

try:
    from pypdf import PdfReader
except ImportError:  # pragma: no cover - handled at runtime
    PdfReader = None


DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DEFAULT_MIN_WORDS = 220
DEFAULT_MAX_WORDS = 380
DEFAULT_OVERLAP_WORDS = 50
EXPECTED_HANDBOOK_FILES = ("pg-handbook.pdf", "ug-handbook.pdf")

HANDBOOK_TITLE_PATTERNS = (
    "nust undergraduate student handbook",
    "nust postgraduate student handbook",
)

HEADING_PREFIXES = (
    "chapter ",
    "annex ",
    "appendix ",
    "contents",
    "disclaimer",
    "important",
    "fee structure",
    "hostel allotment policy",
    "re-checking of papers",
    "list of ",
    "undertaking",
)


@dataclass(slots=True)
class PageRecord:
    source: str
    source_path: str
    page_number: int
    raw_text: str
    clean_text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ChunkRecord:
    chunk_id: str
    chunk_index: int
    source: str
    source_path: str
    text: str
    clean_text: str
    page_start: int
    page_end: int
    section: str
    word_count: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["metadata"] = {
            **self.metadata,
            "chunk_id": self.chunk_id,
            "chunk_index": self.chunk_index,
            "source": self.source,
            "source_path": self.source_path,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "section": self.section,
            "word_count": self.word_count,
        }
        return payload


@dataclass(slots=True)
class _ParagraphUnit:
    text: str
    page_number: int
    section: str

    @property
    def word_count(self) -> int:
        return len(tokenize(self.text))


def _normalize_whitespace(text: str) -> str:
    text = text.replace("\x00", " ")
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _cleanup_line(line: str) -> str:
    line = line.replace("\uf0b7", " ")
    line = re.sub(r"[ ]{2,}", " ", line.strip())
    return line


def _is_noise_line(line: str) -> bool:
    compact = _cleanup_line(line)
    lowered = compact.lower()

    if not compact:
        return True

    if lowered in HANDBOOK_TITLE_PATTERNS:
        return True

    if re.fullmatch(r"[ivxlcdm]+", lowered):
        return True

    if re.fullmatch(r"\d+", compact):
        return True

    if len(compact.split()) == 1 and len(compact) <= 6 and lowered not in {
        "chapter",
        "annex",
        "appendix",
        "contents",
        "important",
    }:
        return True

    return False


def _looks_like_heading(line: str) -> bool:
    candidate = _cleanup_line(line)
    lowered = candidate.lower()

    if not candidate or len(candidate) > 140:
        return False

    if any(lowered.startswith(prefix) for prefix in HEADING_PREFIXES):
        return True

    if re.match(r"^chapter\s+\d+", lowered):
        return True

    if re.match(r"^[A-Z][A-Za-z'&/(),\- ]{2,}$", candidate) and candidate == candidate.upper():
        return True

    words = candidate.split()
    if 1 <= len(words) <= 12 and candidate.endswith(":"):
        return True

    alpha_chars = sum(char.isalpha() for char in candidate)
    upper_chars = sum(char.isupper() for char in candidate)
    if alpha_chars >= 6 and upper_chars / max(alpha_chars, 1) > 0.75:
        return True

    return False


def clean_extracted_text(raw_text: str) -> str:
    if not raw_text:
        return ""

    raw_text = raw_text.replace("\r", "\n")
    lines = raw_text.splitlines()
    blocks: list[str] = []
    current: list[str] = []

    for raw_line in lines:
        line = _cleanup_line(raw_line)

        if _is_noise_line(line):
            if current:
                blocks.append(" ".join(current))
                current = []
            continue

        if _looks_like_heading(line):
            if current:
                blocks.append(" ".join(current))
                current = []
            blocks.append(line)
            continue

        if not line:
            if current:
                blocks.append(" ".join(current))
                current = []
            continue

        if current and current[-1].endswith("-"):
            current[-1] = current[-1][:-1] + line
        else:
            current.append(line)

    if current:
        blocks.append(" ".join(current))

    cleaned_blocks = [_normalize_whitespace(block) for block in blocks if _normalize_whitespace(block)]
    return "\n\n".join(cleaned_blocks)


def tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?", text.lower())


def _sentence_split(text: str) -> list[str]:
    pieces = re.split(r"(?<=[.!?])\s+", text.strip())
    return [piece.strip() for piece in pieces if piece.strip()]


def _split_long_text(text: str, max_words: int) -> list[str]:
    words = tokenize(text)
    if len(words) <= max_words:
        return [text.strip()]

    sentences = _sentence_split(text)
    if len(sentences) <= 1:
        raw_words = text.split()
        return [
            " ".join(raw_words[index : index + max_words]).strip()
            for index in range(0, len(raw_words), max_words)
        ]

    units: list[str] = []
    current: list[str] = []
    current_count = 0

    for sentence in sentences:
        sentence_count = len(tokenize(sentence))
        if sentence_count > max_words:
            if current:
                units.append(" ".join(current).strip())
                current = []
                current_count = 0
            units.extend(_split_long_text(sentence, max_words))
            continue

        if current_count + sentence_count > max_words and current:
            units.append(" ".join(current).strip())
            current = [sentence]
            current_count = sentence_count
        else:
            current.append(sentence)
            current_count += sentence_count

    if current:
        units.append(" ".join(current).strip())

    return units


def read_pdf_pages(pdf_path: str | Path) -> list[PageRecord]:
    if PdfReader is None:
        raise ImportError(
            "pypdf is required for PDF ingestion. Install it with `pip install pypdf`."
        )

    pdf_path = Path(pdf_path).resolve()
    reader = PdfReader(str(pdf_path))
    source = pdf_path.stem
    pages: list[PageRecord] = []

    for page_index, page in enumerate(reader.pages, start=1):
        raw_text = page.extract_text() or ""
        clean_text = clean_extracted_text(raw_text)
        pages.append(
            PageRecord(
                source=source,
                source_path=str(pdf_path),
                page_number=page_index,
                raw_text=raw_text,
                clean_text=clean_text,
            )
        )

    return pages


def read_text_pages(text_path: str | Path) -> list[PageRecord]:
    text_path = Path(text_path).resolve()
    source = text_path.stem
    raw_text = text_path.read_text(encoding="utf-8", errors="ignore")
    clean_text = clean_extracted_text(raw_text)
    return [
        PageRecord(
            source=source,
            source_path=str(text_path),
            page_number=1,
            raw_text=raw_text,
            clean_text=clean_text,
        )
    ]


def read_document_pages(document_path: str | Path) -> list[PageRecord]:
    document_path = Path(document_path)
    suffix = document_path.suffix.lower()
    if suffix == ".pdf":
        return read_pdf_pages(document_path)
    if suffix in {".txt", ".text"}:
        return read_text_pages(document_path)
    raise ValueError(f"Unsupported document type: {document_path}")


def _page_to_units(
    page: PageRecord,
    current_section: str,
    max_words: int,
) -> tuple[list[_ParagraphUnit], str]:
    units: list[_ParagraphUnit] = []
    section = current_section

    for block in page.clean_text.split("\n\n"):
        block = block.strip()
        if not block:
            continue

        if _looks_like_heading(block):
            section = block
            continue

        effective_section = section or "General"
        for piece in _split_long_text(block, max_words):
            piece = _normalize_whitespace(piece)
            if piece:
                units.append(
                    _ParagraphUnit(
                        text=piece,
                        page_number=page.page_number,
                        section=effective_section,
                    )
                )

    return units, section


def _units_to_chunk(
    units: Sequence[_ParagraphUnit],
    source: str,
    source_path: str,
    chunk_index: int,
) -> ChunkRecord:
    body = "\n\n".join(unit.text for unit in units).strip()
    section = units[0].section if units else "General"
    prefixed_text = body if body.lower().startswith(section.lower()) else f"{section}\n\n{body}"
    prefixed_text = _normalize_whitespace(prefixed_text)

    page_start = min(unit.page_number for unit in units)
    page_end = max(unit.page_number for unit in units)
    chunk_id = f"{source}-chunk-{chunk_index:04d}"

    return ChunkRecord(
        chunk_id=chunk_id,
        chunk_index=chunk_index,
        source=source,
        source_path=source_path,
        text=prefixed_text,
        clean_text=prefixed_text.lower(),
        page_start=page_start,
        page_end=page_end,
        section=section,
        word_count=len(tokenize(prefixed_text)),
        metadata={"page_label": _page_label(page_start, page_end)},
    )


def _overlap_units(units: Sequence[_ParagraphUnit], overlap_words: int) -> list[_ParagraphUnit]:
    if overlap_words <= 0 or not units:
        return []

    selected: list[_ParagraphUnit] = []
    running_count = 0
    section = units[-1].section

    for unit in reversed(units):
        if unit.section != section and running_count > 0:
            break
        selected.append(unit)
        running_count += unit.word_count
        if running_count >= overlap_words:
            break

    return list(reversed(selected))


def _page_label(page_start: int, page_end: int) -> str:
    return f"p.{page_start}" if page_start == page_end else f"pp.{page_start}-{page_end}"


def chunk_pages(
    pages: Sequence[PageRecord],
    min_words: int = DEFAULT_MIN_WORDS,
    max_words: int = DEFAULT_MAX_WORDS,
    overlap_words: int = DEFAULT_OVERLAP_WORDS,
) -> list[ChunkRecord]:
    if min_words <= 0 or max_words <= 0:
        raise ValueError("Chunk sizes must be positive.")
    if min_words > max_words:
        raise ValueError("min_words cannot be greater than max_words.")

    if not pages:
        return []

    source = pages[0].source
    source_path = pages[0].source_path
    paragraph_units: list[_ParagraphUnit] = []
    current_section = "General"

    for page in pages:
        units, current_section = _page_to_units(page, current_section, max_words)
        paragraph_units.extend(units)

    chunks: list[ChunkRecord] = []
    current_units: list[_ParagraphUnit] = []
    current_words = 0

    for unit in paragraph_units:
        unit_words = unit.word_count
        section_changed = bool(current_units) and unit.section != current_units[-1].section
        would_overflow = bool(current_units) and current_words + unit_words > max_words

        if current_units and current_words >= min_words and (section_changed or would_overflow):
            chunk = _units_to_chunk(current_units, source, source_path, len(chunks))
            chunks.append(chunk)
            current_units = _overlap_units(current_units, overlap_words)
            current_words = sum(item.word_count for item in current_units)

            if section_changed and current_units and current_units[-1].section != unit.section:
                current_units = []
                current_words = 0

        current_units.append(unit)
        current_words += unit_words

    if current_units:
        if chunks and current_words < min_words:
            previous_chunk = chunks[-1]
            tail_section = current_units[0].section
            tail_text = "\n\n".join(unit.text for unit in current_units).strip()
            if tail_section and tail_section != previous_chunk.section:
                tail_text = f"{tail_section}\n\n{tail_text}"

            merged_text = _normalize_whitespace(f"{previous_chunk.text}\n\n{tail_text}")
            page_end = max(previous_chunk.page_end, max(unit.page_number for unit in current_units))
            chunks[-1] = ChunkRecord(
                chunk_id=previous_chunk.chunk_id,
                chunk_index=previous_chunk.chunk_index,
                source=previous_chunk.source,
                source_path=previous_chunk.source_path,
                text=merged_text,
                clean_text=merged_text.lower(),
                page_start=previous_chunk.page_start,
                page_end=page_end,
                section=previous_chunk.section,
                word_count=len(tokenize(merged_text)),
                metadata={"page_label": _page_label(previous_chunk.page_start, page_end)},
            )
        else:
            chunks.append(_units_to_chunk(current_units, source, source_path, len(chunks)))

    return chunks


def ingest_document(
    document_path: str | Path,
    min_words: int = DEFAULT_MIN_WORDS,
    max_words: int = DEFAULT_MAX_WORDS,
    overlap_words: int = DEFAULT_OVERLAP_WORDS,
) -> dict[str, Any]:
    pages = read_document_pages(document_path)
    chunks = chunk_pages(
        pages,
        min_words=min_words,
        max_words=max_words,
        overlap_words=overlap_words,
    )

    return {
        "source": pages[0].source if pages else Path(document_path).stem,
        "source_path": str(Path(document_path).resolve()),
        "pages": [page.to_dict() for page in pages],
        "chunks": [chunk.to_dict() for chunk in chunks],
        "texts": [chunk.text for chunk in chunks],
        "metadata": [chunk.to_dict()["metadata"] for chunk in chunks],
    }


def discover_handbook_files(data_dir: str | Path = DEFAULT_DATA_DIR) -> list[Path]:
    data_dir = Path(data_dir)
    expected_files = [data_dir / name for name in EXPECTED_HANDBOOK_FILES]
    found_expected = [path for path in expected_files if path.exists() and path.is_file()]

    if found_expected:
        missing = [path.name for path in expected_files if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "Expected handbook file(s) missing from data directory: "
                + ", ".join(missing)
            )
        return found_expected

    files = sorted(
        path
        for path in data_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".pdf", ".txt", ".text"}
    )
    return files


def ingest_corpus(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    min_words: int = DEFAULT_MIN_WORDS,
    max_words: int = DEFAULT_MAX_WORDS,
    overlap_words: int = DEFAULT_OVERLAP_WORDS,
) -> dict[str, Any]:
    documents = []
    chunks: list[ChunkRecord] = []

    for path in discover_handbook_files(data_dir):
        document = ingest_document(
            path,
            min_words=min_words,
            max_words=max_words,
            overlap_words=overlap_words,
        )
        documents.append(
            {
                "source": document["source"],
                "source_path": document["source_path"],
                "page_count": len(document["pages"]),
                "chunk_count": len(document["chunks"]),
            }
        )
        chunks.extend(
            ChunkRecord(
                chunk_id=chunk["chunk_id"],
                chunk_index=chunk["chunk_index"],
                source=chunk["source"],
                source_path=chunk["source_path"],
                text=chunk["text"],
                clean_text=chunk["clean_text"],
                page_start=chunk["page_start"],
                page_end=chunk["page_end"],
                section=chunk["section"],
                word_count=chunk["word_count"],
                metadata=chunk["metadata"],
            )
            for chunk in document["chunks"]
        )

    return {
        "documents": documents,
        "chunks": [chunk.to_dict() for chunk in chunks],
        "texts": [chunk.text for chunk in chunks],
        "metadata": [chunk.to_dict()["metadata"] for chunk in chunks],
    }


def save_corpus(
    output_path: str | Path,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    min_words: int = DEFAULT_MIN_WORDS,
    max_words: int = DEFAULT_MAX_WORDS,
    overlap_words: int = DEFAULT_OVERLAP_WORDS,
) -> Path:
    corpus = ingest_corpus(
        data_dir=data_dir,
        min_words=min_words,
        max_words=max_words,
        overlap_words=overlap_words,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(corpus, indent=2, ensure_ascii=True), encoding="utf-8")
    return output_path


def iter_chunk_texts(corpus: dict[str, Any]) -> Iterable[str]:
    return corpus.get("texts", [])


def iter_chunk_metadata(corpus: dict[str, Any]) -> Iterable[dict[str, Any]]:
    return corpus.get("metadata", [])


__all__ = [
    "ChunkRecord",
    "DEFAULT_DATA_DIR",
    "DEFAULT_MAX_WORDS",
    "DEFAULT_MIN_WORDS",
    "DEFAULT_OVERLAP_WORDS",
    "EXPECTED_HANDBOOK_FILES",
    "PageRecord",
    "chunk_pages",
    "clean_extracted_text",
    "discover_handbook_files",
    "ingest_corpus",
    "ingest_document",
    "iter_chunk_metadata",
    "iter_chunk_texts",
    "read_document_pages",
    "read_pdf_pages",
    "read_text_pages",
    "save_corpus",
    "tokenize",
]
