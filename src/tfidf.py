"""
TF-IDF Baseline Retriever
=========================
Implements exact (non-approximate) retrieval using TF-IDF term weighting and
cosine similarity.  This serves as the *baseline* against which the LSH-based
approximate methods are compared in the experimental analysis.

Integration
-----------
- Accepts the same corpus dict produced by ``ingestion.ingest_corpus()``
  (keys: "texts", "metadata", "chunks").
- Returns ``retrieval.SimilarityResult`` objects so callers can swap this
  retriever in and out alongside ``retrieval.HybridRetriever``.

Theory (for report / presentation)
------------------------------------
TF-IDF (Term Frequency – Inverse Document Frequency):

    tf(t, d)  = count of term t in document d
    idf(t)    = log((1 + N) / (1 + df(t))) + 1   [sklearn "smooth" variant]
    tfidf(t,d)= tf(t,d) * idf(t)

Cosine similarity between query vector q and document vector d:

    cosine(q, d) = (q · d) / (||q|| * ||d||)

This is an *exact* method: every document is explicitly scored against the
query, giving correct top-k results at the cost of O(N * V) computation where
N = number of chunks and V = vocabulary size.
"""

from __future__ import annotations

import math
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# SimilarityResult is defined in retrieval.py — import it so both retrievers
# return identical objects and the rest of the pipeline stays uniform.
from retrieval import SimilarityResult


# ---------------------------------------------------------------------------
# Low-level TF-IDF helpers (implemented from scratch for academic clarity)
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    """
    Lowercase alphanumeric tokenizer (mirrors ingestion.tokenize).
    Returns unigrams only; bigrams are added at the vectoriser level.
    """
    return re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?", text.lower())


def _build_ngrams(tokens: List[str], ngram_range: Tuple[int, int]) -> List[str]:
    """
    Expand a token list into unigrams and / or bigrams based on *ngram_range*.

    Parameters
    ----------
    tokens      : list of word tokens
    ngram_range : (min_n, max_n) — (1, 1) = unigrams only,
                                   (1, 2) = unigrams + bigrams
    """
    min_n, max_n = ngram_range
    result: List[str] = []
    for n in range(min_n, max_n + 1):
        result.extend(
            " ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)
        )
    return result


# ---------------------------------------------------------------------------
# Core TF-IDF Vectoriser (from scratch)
# ---------------------------------------------------------------------------

class TFIDFVectorizer:
    """
    Sparse TF-IDF vectoriser built from scratch.

    Implements the sklearn "smooth" IDF formula:
        idf(t) = log((1 + N) / (1 + df(t))) + 1

    Documents are stored as sparse dicts {term: tfidf_weight} so memory
    usage scales with vocabulary density, not N * V.
    """

    def __init__(
        self,
        ngram_range: Tuple[int, int] = (1, 2),
        max_features: int = 60_000,
        min_df: int = 1,
        sublinear_tf: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        ngram_range   : (min_n, max_n) for n-gram extraction.
        max_features  : keep only the top-k most frequent terms.
        min_df        : minimum document frequency for a term to be kept.
        sublinear_tf  : if True, replace tf with 1 + log(tf) (dampens
                        the effect of high-frequency terms).
        """
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.min_df = min_df
        self.sublinear_tf = sublinear_tf

        # Populated during fit()
        self.vocabulary_: Dict[str, int] = {}      # term -> column index
        self.idf_: Dict[str, float] = {}           # term -> idf weight
        self._n_docs: int = 0
        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    def _get_terms(self, text: str) -> List[str]:
        tokens = _tokenize(text)
        return _build_ngrams(tokens, self.ngram_range)

    def _term_freq(self, term_counts: Counter) -> Dict[str, float]:
        """Apply sublinear TF scaling if enabled."""
        if self.sublinear_tf:
            return {t: 1.0 + math.log(c) for t, c in term_counts.items() if c > 0}
        return dict(term_counts)

    # ------------------------------------------------------------------
    def fit(self, texts: List[str]) -> "TFIDFVectorizer":
        """
        Learn vocabulary and IDF weights from *texts*.

        Steps
        -----
        1. Count document frequency (df) for each term across all documents.
        2. Prune vocabulary to ``max_features`` highest-df terms (≥ min_df).
        3. Compute smooth IDF for each kept term.
        """
        N = len(texts)
        self._n_docs = N

        # --- Step 1: document frequency ---
        df: Counter = Counter()
        for text in texts:
            terms = set(self._get_terms(text))   # set → count each term once per doc
            df.update(terms)

        # --- Step 2: prune vocabulary ---
        # Filter by min_df, then keep top max_features by df count
        kept = [(term, count) for term, count in df.items() if count >= self.min_df]
        kept.sort(key=lambda x: x[1], reverse=True)
        kept = kept[: self.max_features]

        self.vocabulary_ = {term: idx for idx, (term, _) in enumerate(kept)}

        # --- Step 3: IDF ---
        # Smooth IDF: log((1 + N) / (1 + df(t))) + 1
        self.idf_ = {
            term: math.log((1.0 + N) / (1.0 + count)) + 1.0
            for term, count in kept
        }

        self._is_fitted = True
        return self

    def transform(self, text: str) -> Dict[str, float]:
        """
        Convert *text* to a sparse TF-IDF vector (dict of term → weight).

        The vector is L2-normalised so cosine similarity reduces to a dot
        product.
        """
        if not self._is_fitted:
            raise RuntimeError("Vectorizer must be fit() before transform().")

        terms = self._get_terms(text)
        counts: Counter = Counter(t for t in terms if t in self.vocabulary_)

        if not counts:
            return {}

        # Apply TF weighting
        tf_weights = self._term_freq(counts)

        # Multiply by IDF
        tfidf: Dict[str, float] = {
            term: tf_weights[term] * self.idf_[term]
            for term in tf_weights
        }

        # L2 normalise
        norm = math.sqrt(sum(w * w for w in tfidf.values()))
        if norm > 0:
            tfidf = {t: w / norm for t, w in tfidf.items()}

        return tfidf

    def fit_transform(self, texts: List[str]) -> List[Dict[str, float]]:
        """Fit on *texts* and return all document vectors."""
        self.fit(texts)
        return [self.transform(text) for text in texts]


# ---------------------------------------------------------------------------
# Cosine similarity between sparse vectors
# ---------------------------------------------------------------------------

def _cosine_similarity(
    query_vec: Dict[str, float],
    doc_vec: Dict[str, float],
) -> float:
    """
    Dot product of two L2-normalised sparse vectors.

    Because both vectors are already L2-normalised in ``transform()``, the
    dot product equals cosine similarity directly.
    """
    if not query_vec or not doc_vec:
        return 0.0
    # Iterate over the shorter vector for efficiency
    if len(query_vec) > len(doc_vec):
        query_vec, doc_vec = doc_vec, query_vec
    return sum(w * doc_vec.get(term, 0.0) for term, w in query_vec.items())


# ---------------------------------------------------------------------------
# High-level TFIDFRetriever — mirrors HybridRetriever's public interface
# ---------------------------------------------------------------------------

@dataclass
class _IndexedChunk:
    """Internal representation of an indexed chunk."""
    chunk_id: str
    text: str
    tfidf_vector: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)


class TFIDFRetriever:
    """
    Exact retriever using TF-IDF + cosine similarity.

    Usage
    -----
    ::

        from ingestion import ingest_corpus
        from tfidf import TFIDFRetriever

        corpus = ingest_corpus()
        retriever = TFIDFRetriever()
        retriever.fit_corpus(corpus)

        results = retriever.search("What is the minimum GPA requirement?", top_k=5)
        for r in results:
            print(r.chunk_id, r.similarity_score, r.text[:120])

    The ``search()`` method returns a list of ``retrieval.SimilarityResult``
    objects, identical in shape to what ``HybridRetriever.search()`` returns,
    so both can be used interchangeably in the rest of the pipeline.
    """

    METHOD_NAME = "tfidf"

    def __init__(
        self,
        ngram_range: Tuple[int, int] = (1, 2),
        max_features: int = 60_000,
        min_df: int = 1,
        sublinear_tf: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        ngram_range   : (min_n, max_n) — (1,1) unigrams, (1,2) also bigrams.
        max_features  : vocabulary cap.
        min_df        : drop terms appearing in fewer than this many documents.
        sublinear_tf  : apply log-TF scaling (recommended for long documents).
        """
        self.vectorizer = TFIDFVectorizer(
            ngram_range=ngram_range,
            max_features=max_features,
            min_df=min_df,
            sublinear_tf=sublinear_tf,
        )
        self._chunks: List[_IndexedChunk] = []
        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    # Building the index
    # ------------------------------------------------------------------

    def fit_corpus(self, corpus: Dict[str, Any]) -> "TFIDFRetriever":
        """
        Build the TF-IDF index from a corpus dict returned by
        ``ingestion.ingest_corpus()`` or ``ingestion.ingest_document()``.

        The corpus dict is expected to have at least:
          - ``"texts"``    : List[str]   — one entry per chunk
          - ``"metadata"`` : List[dict]  — one entry per chunk
          - ``"chunks"``   : List[dict]  — full ChunkRecord dicts

        Parameters
        ----------
        corpus : dict produced by ingestion module
        """
        texts: List[str] = corpus.get("texts", [])
        metadata_list: List[Dict[str, Any]] = corpus.get("metadata", [])
        chunks_list: List[Dict[str, Any]] = corpus.get("chunks", [])

        if not texts:
            raise ValueError("Corpus contains no texts. Run ingest_corpus() first.")

        # Fit vectorizer and build document vectors
        doc_vectors = self.vectorizer.fit_transform(texts)

        self._chunks = []
        for i, (text, vec) in enumerate(zip(texts, doc_vectors)):
            # chunk_id: prefer chunks_list, fall back to metadata, then index
            if i < len(chunks_list):
                chunk_id = chunks_list[i].get("chunk_id", f"chunk-{i:04d}")
            elif i < len(metadata_list):
                chunk_id = metadata_list[i].get("chunk_id", f"chunk-{i:04d}")
            else:
                chunk_id = f"chunk-{i:04d}"

            meta = metadata_list[i] if i < len(metadata_list) else {}

            self._chunks.append(
                _IndexedChunk(
                    chunk_id=chunk_id,
                    text=text,
                    tfidf_vector=vec,
                    metadata=meta,
                )
            )

        self._is_fitted = True
        return self

    def add_chunk(
        self,
        chunk_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add a single chunk to an already-fitted retriever.

        Note: The IDF weights are *not* recomputed — IDF is fixed at fit time.
        This is intentional: it matches the behaviour of sklearn's
        ``TfidfVectorizer`` and is the standard approach for incremental
        addition.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "Call fit_corpus() before add_chunk(). "
                "The vectorizer must be fitted first to compute IDF weights."
            )
        vec = self.vectorizer.transform(text)
        self._chunks.append(
            _IndexedChunk(
                chunk_id=chunk_id,
                text=text,
                tfidf_vector=vec,
                metadata=metadata or {},
            )
        )

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> List[SimilarityResult]:
        """
        Retrieve the top-k chunks most similar to *query*.

        This is an **exact** search — every indexed chunk is scored.
        Time complexity: O(N * |query_terms|).

        Parameters
        ----------
        query     : natural language question or keyword string
        top_k     : maximum number of results to return
        min_score : discard results below this cosine similarity threshold

        Returns
        -------
        List of ``SimilarityResult`` sorted by cosine similarity (highest first).
        The ``method`` field on each result is set to ``"tfidf"``.
        """
        if not self._is_fitted or not self._chunks:
            raise RuntimeError("Retriever has no indexed chunks. Call fit_corpus() first.")

        query_vec = self.vectorizer.transform(query)

        scores: List[Tuple[float, _IndexedChunk]] = []
        for chunk in self._chunks:
            score = _cosine_similarity(query_vec, chunk.tfidf_vector)
            if score >= min_score:
                scores.append((score, chunk))

        # Sort descending by score
        scores.sort(key=lambda x: x[0], reverse=True)

        return [
            SimilarityResult(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                similarity_score=round(score, 6),
                method=self.METHOD_NAME,
            )
            for score, chunk in scores[:top_k]
        ]

    def search_with_timing(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> Tuple[List[SimilarityResult], float]:
        """
        Same as ``search()`` but also returns wall-clock time in seconds.

        Useful for the experimental analysis (Exact vs Approximate comparison).

        Returns
        -------
        (results, elapsed_seconds)
        """
        t0 = time.perf_counter()
        results = self.search(query, top_k=top_k, min_score=min_score)
        elapsed = time.perf_counter() - t0
        return results, elapsed

    # ------------------------------------------------------------------
    # Introspection helpers (useful for experiments & report)
    # ------------------------------------------------------------------

    @property
    def vocabulary_size(self) -> int:
        """Number of unique terms in the fitted vocabulary."""
        return len(self.vectorizer.vocabulary_)

    @property
    def n_chunks(self) -> int:
        """Number of indexed chunks."""
        return len(self._chunks)

    def top_terms_for_query(self, query: str, top_n: int = 10) -> List[Tuple[str, float]]:
        """
        Return the top-weighted TF-IDF terms for a query.

        Handy for debugging and for explaining retrieval decisions in the demo.
        """
        vec = self.vectorizer.transform(query)
        sorted_terms = sorted(vec.items(), key=lambda x: x[1], reverse=True)
        return sorted_terms[:top_n]

    def idf_weight(self, term: str) -> float:
        """
        Return the IDF weight of a single term.

        Useful for explaining *why* certain terms are considered more
        discriminative (low df → high idf).
        """
        return self.vectorizer.idf_.get(term.lower(), 0.0)

    def memory_usage_bytes(self) -> int:
        """
        Rough estimate of index memory in bytes.

        Counts characters in chunk texts + float weights in sparse vectors.
        Used in the Exact vs Approximate experiment (memory comparison).
        """
        text_bytes = sum(len(c.text.encode()) for c in self._chunks)
        # Each (term, float) pair ≈ avg 12 chars + 8 bytes for the float
        vector_bytes = sum(
            len(t.encode()) + 8
            for c in self._chunks
            for t in c.tfidf_vector
        )
        return text_bytes + vector_bytes

    def get_chunk_vectors(self) -> Dict[str, Dict[str, float]]:
        """
        Return a mapping of ``{chunk_id: tfidf_vector}`` for every indexed chunk.

        Used by ``pagerank.build_pagerank_from_tfidf()`` to build the
        chunk similarity graph without accessing private attributes.

        Returns
        -------
        Dict where keys are chunk IDs and values are L2-normalised sparse
        TF-IDF vectors (dicts of term → weight).
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit_corpus() before get_chunk_vectors().")
        return {c.chunk_id: c.tfidf_vector for c in self._chunks}

    def get_chunk_metadata(self) -> Dict[str, Dict[str, Any]]:
        """
        Return a mapping of ``{chunk_id: metadata}`` for every indexed chunk.

        Useful for displaying page/section references in the interface.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit_corpus() before get_chunk_metadata().")
        return {c.chunk_id: c.metadata for c in self._chunks}

    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "unfitted"
        return (
            f"TFIDFRetriever({status}, "
            f"chunks={self.n_chunks}, "
            f"vocab={self.vocabulary_size}, "
            f"ngram_range={self.vectorizer.ngram_range})"
        )


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def build_tfidf_retriever(
    corpus: Dict[str, Any],
    ngram_range: Tuple[int, int] = (1, 2),
    max_features: int = 60_000,
    min_df: int = 1,
    sublinear_tf: bool = True,
) -> TFIDFRetriever:
    """
    One-shot helper: build and fit a TFIDFRetriever from a corpus dict.

    Parameters
    ----------
    corpus        : output of ``ingestion.ingest_corpus()``
    ngram_range   : (1,1) for unigrams, (1,2) for unigrams+bigrams
    max_features  : vocabulary cap
    min_df        : minimum document frequency
    sublinear_tf  : log-TF dampening

    Returns
    -------
    Fitted ``TFIDFRetriever`` ready for ``search()`` calls.

    Example
    -------
    ::

        from ingestion import ingest_corpus
        from tfidf import build_tfidf_retriever

        corpus = ingest_corpus()
        retriever = build_tfidf_retriever(corpus)
        results = retriever.search("What is the attendance policy?", top_k=5)
    """
    retriever = TFIDFRetriever(
        ngram_range=ngram_range,
        max_features=max_features,
        min_df=min_df,
        sublinear_tf=sublinear_tf,
    )
    retriever.fit_corpus(corpus)
    return retriever


# ---------------------------------------------------------------------------
# __all__
# ---------------------------------------------------------------------------

__all__ = [
    "TFIDFRetriever",
    "TFIDFVectorizer",
    "build_tfidf_retriever",
]


# ---------------------------------------------------------------------------
# Quick self-test (run: python tfidf.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== TF-IDF Retriever — self-test ===\n")

    sample_texts = [
        "The minimum GPA requirement for undergraduate students is 2.0 on a 4.0 scale.",
        "Students who fail a course must repeat it in the next available semester.",
        "Academic probation is imposed when a student's CGPA falls below the minimum threshold.",
        "Attendance policy requires students to attend at least 75% of all lectures.",
        "A course may be repeated a maximum of two times. On the third attempt the grade is final.",
        "Hostel allotment is based on merit, distance from campus, and availability of rooms.",
        "Scholarships are awarded based on academic performance and financial need.",
        "The examination schedule is published two weeks before the midterm period.",
        "Students must submit a repeat course request through the academic office.",
        "GPA below 2.0 for two consecutive semesters results in academic dismissal.",
    ]

    # Build a mini corpus dict (same shape as ingestion.ingest_corpus output)
    mini_corpus = {
        "texts": sample_texts,
        "metadata": [{"chunk_id": f"chunk-{i:04d}", "page_label": f"p.{i+1}"} for i in range(len(sample_texts))],
        "chunks": [{"chunk_id": f"chunk-{i:04d}"} for i in range(len(sample_texts))],
    }

    retriever = TFIDFRetriever(ngram_range=(1, 2), sublinear_tf=True)
    retriever.fit_corpus(mini_corpus)
    print(retriever)
    print(f"Vocabulary size : {retriever.vocabulary_size}")
    print(f"Chunks indexed  : {retriever.n_chunks}")
    print(f"Memory estimate : {retriever.memory_usage_bytes():,} bytes\n")

    queries = [
        "What is the minimum GPA requirement?",
        "What happens if a student fails a course?",
        "What is the attendance policy?",
        "How many times can a course be repeated?",
    ]

    for query in queries:
        results, elapsed = retriever.search_with_timing(query, top_k=3)
        print(f"Query : {query}")
        print(f"Time  : {elapsed*1000:.3f} ms")
        for rank, r in enumerate(results, 1):
            print(f"  [{rank}] ({r.similarity_score:.4f}) {r.text[:90]}...")
        print()
