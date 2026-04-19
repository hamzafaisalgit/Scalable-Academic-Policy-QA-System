"""
PageRank-based Section Importance Scorer
=========================================
**Competitive edge extension** — Ranks important sections of the handbook by
treating the corpus as a weighted similarity graph and running the PageRank
power-iteration algorithm from scratch.

Motivation
----------
Similarity search alone (TF-IDF cosine or MinHash Jaccard) tells us how
close a chunk is to the *query*, but not how *important* or *central* that
chunk is within the handbook as a whole.

PageRank adds a second axis:  a chunk that is *highly similar to many other
important chunks* receives a higher PageRank score.  This mirrors how the
original PageRank algorithm elevates web pages that are cited by other
authoritative pages.

In this system, edges represent **textual similarity** between chunks rather
than hyperlinks.  A chunk about "GPA requirements" that is also similar to
chunks about "academic probation", "scholarship eligibility", and "course
repetition" will have many in-edges and therefore a high PageRank — exactly
what we want.

Algorithm (power iteration)
----------------------------
Given a weighted directed graph G = (V, E, w):

    PR_0(v)  = 1 / N    for all v in V

    PR_{t+1}(v) = (1 - d) / N
               + d * Σ_{u: u→v} PR_t(u) * w(u,v) / Σ_{v'} w(u,v')

where:
    d = damping factor (default 0.85)
    N = |V|

Convergence is checked with L1 norm:  Σ |PR_{t+1}(v) - PR_t(v)| < tol

Integration points
------------------
- ``build_pagerank_from_tfidf(tfidf_retriever)``  ← fastest path; uses
  precomputed L2-normalised TF-IDF vectors from TFIDFRetriever.
- ``build_pagerank_from_lsh(hybrid_retriever)``   ← uses MinHash Jaccard
  similarities from HybridRetriever for a fully LSH-based graph.
- ``PageRankScorer.rerank(results, alpha)``        ← re-ranks any
  List[SimilarityResult] from either retriever using a convex combination of
  the raw similarity score and the normalised PageRank score.
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from retrieval import SimilarityResult


# ---------------------------------------------------------------------------
# Internal graph representation (sparse adjacency)
# ---------------------------------------------------------------------------

class _ChunkGraph:
    """
    Lightweight weighted directed graph for PageRank computation.

    Internally stores:
    - ``out_edges[u]``  → {v: weight}   (for transition-probability computation)
    - ``in_edges[v]``   → {u: weight}   (for efficient PageRank pull-model)
    """

    def __init__(self) -> None:
        self.nodes: Set[str] = set()
        self.out_edges: Dict[str, Dict[str, float]] = {}
        self.in_edges:  Dict[str, Dict[str, float]] = {}

    # ------------------------------------------------------------------
    def add_node(self, node: str) -> None:
        self.nodes.add(node)
        self.out_edges.setdefault(node, {})
        self.in_edges.setdefault(node, {})

    def add_edge(self, u: str, v: str, weight: float = 1.0) -> None:
        """Add or update a directed edge u → v with the given weight."""
        self.add_node(u)
        self.add_node(v)
        # Accumulate weights if edge already exists
        self.out_edges[u][v] = self.out_edges[u].get(v, 0.0) + weight
        self.in_edges[v][u]  = self.in_edges[v].get(u,  0.0) + weight

    def add_undirected_edge(self, u: str, v: str, weight: float = 1.0) -> None:
        """Add a symmetric (bidirectional) edge between u and v."""
        self.add_edge(u, v, weight)
        self.add_edge(v, u, weight)

    # ------------------------------------------------------------------
    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def n_edges(self) -> int:
        return sum(len(nbrs) for nbrs in self.out_edges.values())

    def out_weight_sum(self, node: str) -> float:
        """Total weight of all outgoing edges from *node*."""
        return sum(self.out_edges.get(node, {}).values())

    def __repr__(self) -> str:
        return f"_ChunkGraph(nodes={self.n_nodes}, edges={self.n_edges})"


# ---------------------------------------------------------------------------
# PageRank power iteration
# ---------------------------------------------------------------------------

def _run_pagerank(
    graph: _ChunkGraph,
    damping: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> Tuple[Dict[str, float], int, float]:
    """
    Run the PageRank power-iteration algorithm on *graph*.

    Parameters
    ----------
    graph   : _ChunkGraph — the chunk similarity graph
    damping : teleportation factor d (usually 0.85)
    max_iter: maximum number of iterations
    tol     : convergence tolerance (L1 norm of score changes)

    Returns
    -------
    (scores, iterations_run, l1_residual)
    - scores : {chunk_id: pagerank_score}  (sum ≈ 1.0)
    - iterations_run : how many iterations were needed
    - l1_residual    : final L1 norm (< tol if converged)
    """
    N = graph.n_nodes
    if N == 0:
        return {}, 0, 0.0

    nodes = list(graph.nodes)

    # --- identify dangling nodes (no outgoing edges) ---
    # Their probability mass is redistributed uniformly.
    dangling: Set[str] = {
        node for node in nodes if graph.out_weight_sum(node) == 0.0
    }

    # Initialise uniformly
    scores: Dict[str, float] = {node: 1.0 / N for node in nodes}
    base = (1.0 - damping) / N

    iterations_run = 0
    l1_residual = float("inf")

    for iteration in range(1, max_iter + 1):
        new_scores: Dict[str, float] = {}

        # Dangling-node mass is spread evenly across all nodes
        dangling_sum = damping * sum(scores[d] for d in dangling) / N

        for v in nodes:
            # Teleportation base
            rank = base + dangling_sum

            # Contributions from in-neighbours
            for u, w_uv in graph.in_edges.get(v, {}).items():
                W_u = graph.out_weight_sum(u)
                if W_u > 0:
                    rank += damping * scores[u] * (w_uv / W_u)

            new_scores[v] = rank

        # Check convergence (L1 norm)
        l1_residual = sum(abs(new_scores[v] - scores[v]) for v in nodes)
        scores = new_scores
        iterations_run = iteration

        if l1_residual < tol:
            break

    return scores, iterations_run, l1_residual


# ---------------------------------------------------------------------------
# Graph construction helpers
# ---------------------------------------------------------------------------

def _cosine_sim_sparse(
    vec_a: Dict[str, float],
    vec_b: Dict[str, float],
) -> float:
    """
    Dot product of two L2-normalised sparse dicts (= cosine similarity).
    Iterates over the shorter vector for efficiency.
    """
    if not vec_a or not vec_b:
        return 0.0
    if len(vec_a) > len(vec_b):
        vec_a, vec_b = vec_b, vec_a
    return sum(w * vec_b.get(t, 0.0) for t, w in vec_a.items())


def _build_tfidf_graph(
    chunk_ids: List[str],
    tfidf_vectors: List[Dict[str, float]],
    threshold: float,
) -> _ChunkGraph:
    """
    Build a chunk similarity graph from precomputed TF-IDF vectors.

    An undirected edge (u, v) with weight = cosine_similarity(u, v) is added
    whenever the similarity exceeds *threshold*.

    Time complexity: O(N² * avg_terms_per_vector) — acceptable for N ≤ 2 000.
    For very large corpora, replace with an LSH candidate-pair pass.

    Parameters
    ----------
    chunk_ids     : ordered list of chunk IDs
    tfidf_vectors : corresponding L2-normalised TF-IDF sparse vectors
    threshold     : minimum cosine similarity to create an edge
    """
    graph = _ChunkGraph()
    n = len(chunk_ids)
    for cid in chunk_ids:
        graph.add_node(cid)

    for i in range(n):
        for j in range(i + 1, n):
            sim = _cosine_sim_sparse(tfidf_vectors[i], tfidf_vectors[j])
            if sim >= threshold:
                graph.add_undirected_edge(chunk_ids[i], chunk_ids[j], sim)

    return graph


def _build_minhash_graph(
    chunk_ids: List[str],
    signatures: Dict[str, Any],   # {chunk_id: MinHash}
    threshold: float,
) -> _ChunkGraph:
    """
    Build a chunk similarity graph from MinHash signatures (Jaccard similarity).

    Uses the already-stored MinHash signatures from HybridRetriever / LSH index
    to avoid recomputing signatures.  Jaccard similarity is used as edge weight.

    Parameters
    ----------
    chunk_ids  : list of chunk IDs in the order they were indexed
    signatures : {chunk_id: MinHash} — from HybridRetriever.lsh.doc_signatures
    threshold  : minimum Jaccard similarity to create an edge
    """
    graph = _ChunkGraph()
    ids = [cid for cid in chunk_ids if cid in signatures]
    n = len(ids)
    for cid in ids:
        graph.add_node(cid)

    for i in range(n):
        for j in range(i + 1, n):
            sim = signatures[ids[i]].jaccard_similarity(signatures[ids[j]])
            if sim >= threshold:
                graph.add_undirected_edge(ids[i], ids[j], sim)

    return graph


# ---------------------------------------------------------------------------
# Public PageRankScorer
# ---------------------------------------------------------------------------

class PageRankScorer:
    """
    Builds and stores PageRank scores for handbook chunks and re-ranks
    retrieval results using those scores.

    Typical usage
    -------------
    ::

        from pagerank import build_pagerank_from_tfidf

        scorer = build_pagerank_from_tfidf(tfidf_retriever, threshold=0.15)
        results = tfidf_retriever.search("What is the attendance policy?", top_k=10)
        reranked = scorer.rerank(results, alpha=0.35)

    Attributes
    ----------
    scores      : raw PageRank scores {chunk_id: float}
    graph       : the _ChunkGraph used for computation
    iterations  : number of power-iteration steps until convergence
    residual    : final L1 residual at convergence
    build_time  : seconds taken to build graph + run PageRank
    """

    def __init__(
        self,
        damping: float = 0.85,
        max_iter: int = 100,
        tol: float = 1e-6,
    ) -> None:
        """
        Parameters
        ----------
        damping  : PageRank damping factor (teleportation probability = 1-d).
        max_iter : maximum power-iteration steps.
        tol      : L1 convergence tolerance.
        """
        self.damping   = damping
        self.max_iter  = max_iter
        self.tol       = tol

        self.scores:     Dict[str, float] = {}
        self.graph:      Optional[_ChunkGraph] = None
        self.iterations: int   = 0
        self.residual:   float = float("inf")
        self.build_time: float = 0.0
        self._min_score: float = 0.0
        self._max_score: float = 1.0
        self._is_fitted: bool  = False

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, graph: _ChunkGraph) -> "PageRankScorer":
        """
        Run PageRank on the given *graph* and store scores.

        Parameters
        ----------
        graph : pre-built _ChunkGraph
        """
        t0 = time.perf_counter()
        self.graph = graph
        self.scores, self.iterations, self.residual = _run_pagerank(
            graph,
            damping=self.damping,
            max_iter=self.max_iter,
            tol=self.tol,
        )
        self.build_time = time.perf_counter() - t0

        # Pre-compute min/max for normalisation
        if self.scores:
            vals = list(self.scores.values())
            self._min_score = min(vals)
            self._max_score = max(vals)

        self._is_fitted = True
        return self

    # ------------------------------------------------------------------
    # Score access
    # ------------------------------------------------------------------

    def get_score(self, chunk_id: str) -> float:
        """
        Return the raw PageRank score for *chunk_id*.
        Returns 0.0 if the chunk was not in the graph.
        """
        return self.scores.get(chunk_id, 0.0)

    def get_normalised_score(self, chunk_id: str) -> float:
        """
        Return a min-max normalised PageRank score in [0, 1].

        This is used when combining PageRank with the similarity score so
        both are on the same scale.
        """
        raw = self.get_score(chunk_id)
        span = self._max_score - self._min_score
        if span == 0.0:
            return 1.0
        return (raw - self._min_score) / span

    # ------------------------------------------------------------------
    # Re-ranking
    # ------------------------------------------------------------------

    def rerank(
        self,
        results: List[SimilarityResult],
        alpha: float = 0.30,
    ) -> List[SimilarityResult]:
        """
        Re-rank a list of ``SimilarityResult`` objects using PageRank scores.

        New score formula (convex combination):

            final_score = (1 - alpha) * similarity_score
                        + alpha      * normalised_pagerank_score

        Parameters
        ----------
        results : list of SimilarityResult (from TFIDFRetriever or HybridRetriever)
        alpha   : weight given to PageRank (0 = pure similarity, 1 = pure PageRank)
                  Typical range: 0.20 – 0.40.

        Returns
        -------
        New sorted list of SimilarityResult with updated similarity_score and
        method string ``"<original_method>+pagerank"``.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before rerank().")

        reranked: List[SimilarityResult] = []
        for r in results:
            pr_norm = self.get_normalised_score(r.chunk_id)
            new_score = (1.0 - alpha) * r.similarity_score + alpha * pr_norm
            reranked.append(
                SimilarityResult(
                    chunk_id=r.chunk_id,
                    text=r.text,
                    similarity_score=round(new_score, 6),
                    method=f"{r.method}+pagerank",
                )
            )

        reranked.sort(key=lambda x: x.similarity_score, reverse=True)
        return reranked

    # ------------------------------------------------------------------
    # Introspection / reporting
    # ------------------------------------------------------------------

    def top_sections(self, n: int = 10) -> List[Tuple[str, float]]:
        """
        Return the *n* highest-ranked chunk IDs and their raw PageRank scores.

        Useful for identifying the most *central / important* sections of the
        handbook — a key insight to present in the demo and report.
        """
        if not self.scores:
            return []
        ranked = sorted(self.scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:n]

    def stats(self) -> Dict[str, Any]:
        """
        Return a summary dict suitable for logging / experimental analysis.

        Fields
        ------
        n_nodes       : number of chunks in the graph
        n_edges       : number of edges in the graph
        iterations    : convergence iterations
        residual      : final L1 residual
        build_time_ms : graph build + PageRank computation time in ms
        min_score     : minimum raw PageRank score
        max_score     : maximum raw PageRank score
        mean_score    : mean raw PageRank score
        """
        vals = list(self.scores.values())
        return {
            "n_nodes":       self.graph.n_nodes if self.graph else 0,
            "n_edges":       self.graph.n_edges if self.graph else 0,
            "iterations":    self.iterations,
            "residual":      round(self.residual, 8),
            "build_time_ms": round(self.build_time * 1000, 2),
            "min_score":     round(self._min_score, 8),
            "max_score":     round(self._max_score, 8),
            "mean_score":    round(sum(vals) / len(vals), 8) if vals else 0.0,
        }

    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "unfitted"
        return (
            f"PageRankScorer({status}, "
            f"nodes={self.graph.n_nodes if self.graph else 0}, "
            f"edges={self.graph.n_edges if self.graph else 0}, "
            f"iters={self.iterations})"
        )


# ---------------------------------------------------------------------------
# Convenience factory functions
# ---------------------------------------------------------------------------

def build_pagerank_from_tfidf(
    tfidf_retriever: Any,
    threshold: float = 0.15,
    damping: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> PageRankScorer:
    """
    Build and fit a PageRankScorer from an already-fitted TFIDFRetriever.

    This is the **primary integration path**.  The TFIDFRetriever already holds
    precomputed L2-normalised TF-IDF vectors for every chunk in memory, so
    graph construction is simply an O(N²) pass of sparse dot products — no
    extra computation needed.

    Parameters
    ----------
    tfidf_retriever : fitted ``tfidf.TFIDFRetriever`` instance
    threshold       : minimum cosine similarity to add an edge (0.10 – 0.25
                      works well for handbook chunks of 200–400 words)
    damping         : PageRank damping factor
    max_iter        : maximum power-iteration steps
    tol             : L1 convergence tolerance

    Returns
    -------
    Fitted ``PageRankScorer`` ready for ``rerank()`` calls.

    Example
    -------
    ::

        from ingestion import ingest_corpus
        from tfidf import build_tfidf_retriever
        from pagerank import build_pagerank_from_tfidf

        corpus   = ingest_corpus()
        tfidf    = build_tfidf_retriever(corpus)
        scorer   = build_pagerank_from_tfidf(tfidf, threshold=0.15)

        results  = tfidf.search("What is the GPA requirement?", top_k=10)
        reranked = scorer.rerank(results, alpha=0.30)
    """
    # Pull precomputed vectors via the public API
    chunk_vec_map = tfidf_retriever.get_chunk_vectors()   # {chunk_id: sparse_vec}
    chunk_ids     = list(chunk_vec_map.keys())
    vectors       = list(chunk_vec_map.values())

    graph = _build_tfidf_graph(chunk_ids, vectors, threshold)

    scorer = PageRankScorer(damping=damping, max_iter=max_iter, tol=tol)
    scorer.fit(graph)
    return scorer


def build_pagerank_from_lsh(
    hybrid_retriever: Any,
    threshold: float = 0.15,
    damping: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> PageRankScorer:
    """
    Build and fit a PageRankScorer from a HybridRetriever using MinHash Jaccard
    similarity to define edge weights.

    This variant is interesting for the report because it shows how the
    *approximate* MinHash similarity can drive a PageRank graph, giving an
    approximate but scalable alternative to the exact TF-IDF cosine graph.

    Parameters
    ----------
    hybrid_retriever : fitted ``retrieval.HybridRetriever`` instance
                       (must have been initialised with MinHash+LSH)
    threshold        : minimum Jaccard similarity to create an edge
    damping          : PageRank damping factor
    max_iter         : maximum power-iteration steps
    tol              : L1 convergence tolerance

    Returns
    -------
    Fitted ``PageRankScorer``.

    Example
    -------
    ::

        from retrieval import HybridRetriever, SimilarityMethod
        from pagerank import build_pagerank_from_lsh

        retriever = HybridRetriever(method=SimilarityMethod.HYBRID)
        for cid, text in corpus_chunks:
            retriever.add_chunk(cid, text)

        scorer   = build_pagerank_from_lsh(retriever, threshold=0.10)
        results  = retriever.search("attendance policy", top_k=10)
        reranked = scorer.rerank(results, alpha=0.30)
    """
    if hybrid_retriever.lsh is None:
        raise ValueError(
            "HybridRetriever must be initialised with MinHash+LSH "
            "(SimilarityMethod.MINHASH_LSH or SimilarityMethod.HYBRID)."
        )

    signatures = hybrid_retriever.lsh.doc_signatures   # {chunk_id: MinHash}
    chunk_ids  = list(signatures.keys())

    graph = _build_minhash_graph(chunk_ids, signatures, threshold)

    scorer = PageRankScorer(damping=damping, max_iter=max_iter, tol=tol)
    scorer.fit(graph)
    return scorer


def build_pagerank_from_texts(
    chunk_ids: List[str],
    texts: List[str],
    threshold: float = 0.15,
    damping: float = 0.85,
    ngram_range: Tuple[int, int] = (1, 2),
    max_features: int = 40_000,
) -> PageRankScorer:
    """
    Build a PageRankScorer directly from raw texts (no pre-fitted retriever
    required).  Vectorises internally, then runs graph construction + PageRank.

    Useful when you want PageRank as a standalone step without building a full
    TFIDFRetriever first.

    Parameters
    ----------
    chunk_ids    : list of unique chunk identifiers
    texts        : corresponding list of chunk texts
    threshold    : cosine similarity threshold for edges
    damping      : PageRank damping factor
    ngram_range  : n-gram range for internal TF-IDF vectorisation
    max_features : vocabulary cap

    Returns
    -------
    Fitted ``PageRankScorer``.
    """
    # Import here to avoid circular dependency at module level
    from tfidf import TFIDFVectorizer

    vec = TFIDFVectorizer(ngram_range=ngram_range, max_features=max_features)
    vectors = vec.fit_transform(texts)

    graph = _build_tfidf_graph(chunk_ids, vectors, threshold)
    scorer = PageRankScorer(damping=damping)
    scorer.fit(graph)
    return scorer



# ---------------------------------------------------------------------------
# __all__
# ---------------------------------------------------------------------------

__all__ = [
    "PageRankScorer",
    "build_pagerank_from_tfidf",
    "build_pagerank_from_lsh",
    "build_pagerank_from_texts",
]


# ---------------------------------------------------------------------------
# Self-test (python pagerank.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 65)
    print("PageRank Scorer — self-test")
    print("=" * 65 + "\n")

    # ------------------------------------------------------------------
    # 1. Build a small synthetic corpus
    # ------------------------------------------------------------------
    sample_chunks = {
        "c00": "The minimum CGPA requirement for undergraduate students is 2.00.",
        "c01": "Students with CGPA below 2.00 are placed on academic probation.",
        "c02": "Academic probation requires students to improve their CGPA.",
        "c03": "Students on probation must meet with their academic advisor.",
        "c04": "A CGPA below 2.00 for two consecutive semesters leads to dismissal.",
        "c05": "Attendance policy requires 75% presence in all classes.",
        "c06": "Students with attendance below 75% are not allowed to sit exams.",
        "c07": "Course repetition is limited to two attempts per course.",
        "c08": "Repeated courses replace the original grade in CGPA calculation.",
        "c09": "Scholarship eligibility is based on CGPA and financial need.",
        "c10": "Hostel allotment is based on merit, distance, and availability.",
        "c11": "Hostel fees are charged per semester based on room type.",
        "c12": "Fee structure for undergraduate students is published annually.",
        "c13": "Examination schedule is released two weeks before midterm week.",
        "c14": "Students must carry their university ID during examinations.",
    }

    ids = list(sample_chunks.keys())
    txts = list(sample_chunks.values())

    # ------------------------------------------------------------------
    # 2. Build PageRank from raw texts
    # ------------------------------------------------------------------
    print("Building PageRank graph from raw texts...")
    t0 = time.perf_counter()
    scorer = build_pagerank_from_texts(ids, txts, threshold=0.05, damping=0.85)
    elapsed = time.perf_counter() - t0

    print(scorer)
    st = scorer.stats()
    print(f"  Nodes      : {st['n_nodes']}")
    print(f"  Edges      : {st['n_edges']}")
    print(f"  Iterations : {st['iterations']}")
    print(f"  Residual   : {st['residual']:.2e}")
    print(f"  Build time : {st['build_time_ms']:.1f} ms\n")

    # ------------------------------------------------------------------
    # 3. Top sections by PageRank
    # ------------------------------------------------------------------
    print("Top-5 most important sections (PageRank):")
    for chunk_id, score in scorer.top_sections(n=5):
        print(f"  {chunk_id}  PR={score:.6f}  │  {sample_chunks[chunk_id][:70]}...")
    print()

    # ------------------------------------------------------------------
    # 4. Re-ranking demo
    # ------------------------------------------------------------------
    # Fake similarity results (as if from TF-IDF / LSH search)
    fake_results = [
        SimilarityResult(chunk_id="c00", text=sample_chunks["c00"], similarity_score=0.82, method="tfidf"),
        SimilarityResult(chunk_id="c01", text=sample_chunks["c01"], similarity_score=0.78, method="tfidf"),
        SimilarityResult(chunk_id="c05", text=sample_chunks["c05"], similarity_score=0.40, method="tfidf"),
        SimilarityResult(chunk_id="c09", text=sample_chunks["c09"], similarity_score=0.35, method="tfidf"),
    ]

    print("Before re-ranking:")
    for r in fake_results:
        print(f"  {r.chunk_id}  score={r.similarity_score:.4f}  {r.text[:60]}...")

    reranked = scorer.rerank(fake_results, alpha=0.30)
    print("\nAfter re-ranking with PageRank (alpha=0.30):")
    for r in reranked:
        pr_raw = scorer.get_score(r.chunk_id)
        pr_nrm = scorer.get_normalised_score(r.chunk_id)
        print(
            f"  {r.chunk_id}  final={r.similarity_score:.4f}"
            f"  PR_raw={pr_raw:.6f}  PR_norm={pr_nrm:.4f}"
            f"  method={r.method}"
        )

    print("\n✓ PageRank self-test passed.")
