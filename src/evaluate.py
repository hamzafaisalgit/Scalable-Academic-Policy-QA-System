"""
evaluate.py — Experimental Evaluation Suite
=============================================
Covers all experiments required by the project specification:

  Experiment 1 — Exact vs Approximate Retrieval
      Compare TF-IDF (exact) against MinHash+LSH and SimHash (approximate).
      Metrics: Precision@k, Recall@k, query latency (ms), memory usage (KB).
      TF-IDF top-k results serve as ground truth for approximate methods.

  Experiment 2 — Parameter Sensitivity Analysis
      2a: MinHash — number of hash functions (num_perm: 32, 64, 128, 256)
      2b: LSH     — number of           (num_bands: 8, 16, 32, 64)
      2c: SimHash — Hamming distance threshold (10, 20, 30, 40, 50)

  Experiment 3 — Scalability Test
      Corpus duplicated to 1×, 2×, 5×, 10× original size.
      Measures index build time and query latency for both TF-IDF and LSH.

  Qualitative — 15 Sample Queries
      Per-query top-1 results from TF-IDF and LSH with text previews.

All results are printed to stdout and saved as CSV files in results/.

Usage:
    cd src/
    python evaluate.py
"""

from __future__ import annotations

import csv
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Make sure src/ is on the import path regardless of cwd
sys.path.insert(0, str(Path(__file__).parent))

from ingestion import ingest_corpus
from minhash_lsh import LocalitySensitiveHash, MinHash, shingle_text
from retrieval import HybridRetriever, SimilarityMethod, SimilarityResult
from tfidf import TFIDFRetriever

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# 15 evaluation queries (all 4 spec sample queries included)
# ---------------------------------------------------------------------------
EVAL_QUERIES: List[str] = [
    "What is the minimum GPA requirement?",
    "What happens if a student fails a course?",
    "What is the attendance policy?",
    "How many times can a course be repeated?",
    "What is academic probation?",
    "What are the hostel rules and regulations?",
    "What is the fee refund policy?",
    "How is CGPA calculated?",
    "What are the examination rules?",
    "What is the policy on plagiarism and unfair means?",
    "What scholarships are available for students?",
    "What are the grading criteria and grade points?",
    "What are the conditions for course withdrawal?",
    "What happens if a student misses an examination?",
    "What are the disciplinary actions for rule violations?",
]

TOP_K = 5  # k used throughout all Precision@k / Recall@k calculations


# ===========================================================================
# Utility helpers
# ===========================================================================

def _measure(fn, *args, **kwargs) -> Tuple[Any, float, float]:
    """
    Run fn(*args, **kwargs) and return (result, elapsed_ms, peak_mem_kb).
    Uses tracemalloc to capture Python-heap allocations during the call.
    """
    if tracemalloc.is_tracing():
        tracemalloc.stop()
    tracemalloc.start()
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, elapsed_ms, peak_bytes / 1024.0


def precision_at_k(
    ground_truth: List[SimilarityResult],
    predicted: List[SimilarityResult],
    k: int,
) -> float:
    """Fraction of predicted top-k that appear in ground-truth top-k."""
    gt_ids: Set[str] = {r.chunk_id for r in ground_truth[:k]}
    pred_ids: Set[str] = {r.chunk_id for r in predicted[:k]}
    if not pred_ids:
        return 0.0
    return len(gt_ids & pred_ids) / k


def recall_at_k(
    ground_truth: List[SimilarityResult],
    predicted: List[SimilarityResult],
    k: int,
) -> float:
    """Fraction of ground-truth top-k that appear in predicted top-k."""
    gt_ids: Set[str] = {r.chunk_id for r in ground_truth[:k]}
    pred_ids: Set[str] = {r.chunk_id for r in predicted[:k]}
    if not gt_ids:
        return 0.0
    return len(gt_ids & pred_ids) / len(gt_ids)


def _avg(lst: List[float]) -> float:
    return sum(lst) / len(lst) if lst else 0.0


def _print_table(
    headers: List[str],
    rows: List[List[Any]],
    title: str = "",
) -> None:
    if title:
        print(f"\n{'=' * 72}")
        print(f"  {title}")
        print(f"{'=' * 72}")
    col_w = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_w[i] = max(col_w[i], len(str(cell)))
    fmt = "  " + "  ".join(f"{{:<{w}}}" for w in col_w)
    sep = "  " + "  ".join("-" * w for w in col_w)
    print(fmt.format(*headers))
    print(sep)
    for row in rows:
        print(fmt.format(*[str(c) for c in row]))


def _save_csv(filename: str, headers: List[str], rows: List[List[Any]]) -> None:
    path = RESULTS_DIR / filename
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(rows)
    print(f"  → saved: {path}")


# ===========================================================================
# Corpus builders
# ===========================================================================

def _build_tfidf(corpus: Dict[str, Any]) -> TFIDFRetriever:
    r = TFIDFRetriever()
    r.fit_corpus(corpus)
    return r


def _build_hybrid_lsh(corpus: Dict[str, Any]) -> HybridRetriever:
    """Standard LSH retriever (128 perms, auto bands) via HybridRetriever."""
    r = HybridRetriever(method=SimilarityMethod.MINHASH_LSH, minhash_num_perm=128)
    texts = corpus.get("texts", [])
    chunks = corpus.get("chunks", [])
    for i, text in enumerate(texts):
        cid = chunks[i].get("chunk_id", f"chunk-{i:04d}") if i < len(chunks) else f"chunk-{i:04d}"
        r.add_chunk(cid, text)
    return r


def _build_simhash(corpus: Dict[str, Any]) -> HybridRetriever:
    r = HybridRetriever(method=SimilarityMethod.SIMHASH)
    texts = corpus.get("texts", [])
    chunks = corpus.get("chunks", [])
    for i, text in enumerate(texts):
        cid = chunks[i].get("chunk_id", f"chunk-{i:04d}") if i < len(chunks) else f"chunk-{i:04d}"
        r.add_chunk(cid, text)
    return r


# ---------------------------------------------------------------------------
# Custom LSH builder that correctly threads num_perm through
# (HybridRetriever.add_chunk calls get_minhash_signature without num_perm,
#  so for param-sensitivity experiments we build the index directly.)
# ---------------------------------------------------------------------------

def _build_lsh_custom(
    corpus: Dict[str, Any],
    num_perm: int,
    num_bands: Optional[int] = None,
) -> Tuple[LocalitySensitiveHash, Dict[str, str]]:
    """
    Build an LSH index directly, passing num_perm to both MinHash and LSH.
    Returns (lsh_index, {chunk_id: text}).
    """
    lsh_index = LocalitySensitiveHash(num_perm=num_perm, num_bands=num_bands)
    chunk_texts: Dict[str, str] = {}
    texts = corpus.get("texts", [])
    chunks = corpus.get("chunks", [])
    for i, text in enumerate(texts):
        cid = chunks[i].get("chunk_id", f"chunk-{i:04d}") if i < len(chunks) else f"chunk-{i:04d}"
        shingles = shingle_text(text)
        sig = MinHash(num_perm=num_perm)
        sig.update(shingles)
        lsh_index.add_document(cid, sig)
        chunk_texts[cid] = text
    return lsh_index, chunk_texts


def _query_lsh_custom(
    lsh_index: LocalitySensitiveHash,
    chunk_texts: Dict[str, str],
    query: str,
    num_perm: int,
    threshold: float = 0.05,
    top_k: int = TOP_K,
) -> List[SimilarityResult]:
    shingles = shingle_text(query)
    sig = MinHash(num_perm=num_perm)
    sig.update(shingles)
    raw = lsh_index.query(sig, threshold=threshold)
    return [
        SimilarityResult(
            chunk_id=doc_id,
            text=chunk_texts[doc_id],
            similarity_score=score,
            method="minhash_lsh",
        )
        for doc_id, score in raw[:top_k]
    ]


# ---------------------------------------------------------------------------
# Corpus scaling (duplicate with unique IDs)
# ---------------------------------------------------------------------------

def _scale_corpus(corpus: Dict[str, Any], factor: int) -> Dict[str, Any]:
    if factor == 1:
        return corpus
    base_texts = corpus.get("texts", [])
    base_meta = corpus.get("metadata", [])
    base_chunks = corpus.get("chunks", [])
    new_texts, new_meta, new_chunks = [], [], []
    for rep in range(factor):
        suffix = "" if rep == 0 else f"-r{rep}"
        for i, text in enumerate(base_texts):
            new_texts.append(text)
            if i < len(base_meta):
                m = dict(base_meta[i])
                m["chunk_id"] = m.get("chunk_id", f"chunk-{i:04d}") + suffix
                new_meta.append(m)
            if i < len(base_chunks):
                c = dict(base_chunks[i])
                c["chunk_id"] = c.get("chunk_id", f"chunk-{i:04d}") + suffix
                new_chunks.append(c)
    return {"texts": new_texts, "metadata": new_meta, "chunks": new_chunks}


# ===========================================================================
# EXPERIMENT 1 — Exact vs Approximate Retrieval
# ===========================================================================

def experiment_1(corpus: Dict[str, Any]) -> None:
    print("\n" + "#" * 72)
    print("# EXPERIMENT 1: Exact (TF-IDF) vs Approximate (LSH / SimHash)")
    print("# Metrics: Precision@k, Recall@k, query latency (ms), memory (KB)")
    print("#" * 72)

    print("\n  Building retrievers...")
    tfidf = _build_tfidf(corpus)
    lsh = _build_hybrid_lsh(corpus)
    simhash = _build_simhash(corpus)
    print(f"  Done — corpus size: {len(corpus.get('texts', []))} chunks\n")

    rows: List[List[Any]] = []
    tfidf_ms_all, lsh_ms_all, sh_ms_all = [], [], []
    tfidf_mem_all, lsh_mem_all = [], []
    prec_lsh_all, rec_lsh_all = [], []
    prec_sh_all, rec_sh_all = [], []

    for query in EVAL_QUERIES:
        gt, tfidf_ms, tfidf_mem = _measure(tfidf.search, query, top_k=TOP_K)

        lsh_res, lsh_ms, lsh_mem = _measure(
            lsh.search,
            query,
            method=SimilarityMethod.MINHASH_LSH,
            minhash_threshold=0.05,
            top_k=TOP_K,
        )

        sh_res, sh_ms, _ = _measure(
            simhash.search,
            query,
            method=SimilarityMethod.SIMHASH,
            simhash_max_distance=30,
            top_k=TOP_K,
        )

        p_lsh = precision_at_k(gt, lsh_res, TOP_K)
        r_lsh = recall_at_k(gt, lsh_res, TOP_K)
        p_sh = precision_at_k(gt, sh_res, TOP_K)
        r_sh = recall_at_k(gt, sh_res, TOP_K)

        tfidf_ms_all.append(tfidf_ms)
        lsh_ms_all.append(lsh_ms)
        sh_ms_all.append(sh_ms)
        tfidf_mem_all.append(tfidf_mem)
        lsh_mem_all.append(lsh_mem)
        prec_lsh_all.append(p_lsh)
        rec_lsh_all.append(r_lsh)
        prec_sh_all.append(p_sh)
        rec_sh_all.append(r_sh)

        short_q = (query[:42] + "...") if len(query) > 42 else query
        rows.append([
            short_q,
            f"{tfidf_ms:.1f}",
            f"{lsh_ms:.1f}",
            f"{sh_ms:.1f}",
            f"{p_lsh:.2f}",
            f"{r_lsh:.2f}",
            f"{p_sh:.2f}",
            f"{r_sh:.2f}",
            f"{tfidf_mem:.0f}",
            f"{lsh_mem:.0f}",
        ])

    headers = [
        "Query",
        "TFIDF_ms", "LSH_ms", "SH_ms",
        f"P@{TOP_K}_LSH", f"R@{TOP_K}_LSH",
        f"P@{TOP_K}_SH", f"R@{TOP_K}_SH",
        "TFIDF_KB", "LSH_KB",
    ]
    _print_table(headers, rows, title=f"Experiment 1 — Per-Query Results (k={TOP_K})")

    # Summary
    speedup_lsh = _avg(tfidf_ms_all) / max(_avg(lsh_ms_all), 0.001)
    speedup_sh = _avg(tfidf_ms_all) / max(_avg(sh_ms_all), 0.001)
    print(f"\n  {'─'*55}")
    print(f"  SUMMARY")
    print(f"  {'─'*55}")
    print(f"  Avg TF-IDF query latency  : {_avg(tfidf_ms_all):6.2f} ms")
    print(f"  Avg LSH query latency     : {_avg(lsh_ms_all):6.2f} ms  (speedup {speedup_lsh:.1f}×)")
    print(f"  Avg SimHash query latency : {_avg(sh_ms_all):6.2f} ms  (speedup {speedup_sh:.1f}×)")
    print(f"  Avg Precision@{TOP_K} LSH      : {_avg(prec_lsh_all):.3f}")
    print(f"  Avg Recall@{TOP_K} LSH        : {_avg(rec_lsh_all):.3f}")
    print(f"  Avg Precision@{TOP_K} SimHash  : {_avg(prec_sh_all):.3f}")
    print(f"  Avg Recall@{TOP_K} SimHash    : {_avg(rec_sh_all):.3f}")
    print(f"  Avg TF-IDF query memory   : {_avg(tfidf_mem_all):6.1f} KB")
    print(f"  Avg LSH query memory      : {_avg(lsh_mem_all):6.1f} KB")

    # Add average row and save
    avg_row: List[Any] = [
        "AVERAGE",
        f"{_avg(tfidf_ms_all):.1f}", f"{_avg(lsh_ms_all):.1f}", f"{_avg(sh_ms_all):.1f}",
        f"{_avg(prec_lsh_all):.2f}", f"{_avg(rec_lsh_all):.2f}",
        f"{_avg(prec_sh_all):.2f}", f"{_avg(rec_sh_all):.2f}",
        f"{_avg(tfidf_mem_all):.0f}", f"{_avg(lsh_mem_all):.0f}",
    ]
    _save_csv("exp1_exact_vs_approx.csv", headers, rows + [avg_row])


# ===========================================================================
# EXPERIMENT 2 — Parameter Sensitivity
# ===========================================================================

def experiment_2(corpus: Dict[str, Any]) -> None:
    print("\n" + "#" * 72)
    print("# EXPERIMENT 2: Parameter Sensitivity Analysis")
    print("#" * 72)

    probe_queries = EVAL_QUERIES[:5]  # 5 representative queries
    tfidf = _build_tfidf(corpus)

    # ── 2a: MinHash num_perm ────────────────────────────────────────────────
    print("\n  2a: MinHash — num_perm (number of hash functions)")
    print("      Higher num_perm → more accurate signatures, slower indexing.\n")

    NUM_PERMS = [32, 64, 128, 256]
    rows_2a: List[List[Any]] = []

    for np in NUM_PERMS:
        # Build index with correct num_perm threaded all the way through
        t0 = time.perf_counter()
        lsh_idx, chunk_texts = _build_lsh_custom(corpus, num_perm=np)
        idx_ms = (time.perf_counter() - t0) * 1000.0

        precs, lats = [], []
        for q in probe_queries:
            gt = tfidf.search(q, top_k=TOP_K)
            res, ms, _ = _measure(_query_lsh_custom, lsh_idx, chunk_texts, q, np, 0.05, TOP_K)
            precs.append(precision_at_k(gt, res, TOP_K))
            lats.append(ms)

        rows_2a.append([
            np,
            f"{idx_ms:.0f}",
            f"{_avg(lats):.2f}",
            f"{_avg(precs):.3f}",
            len(chunk_texts),
        ])

    _print_table(
        ["num_perm", "Index_ms", f"Avg_Query_ms", f"Avg_P@{TOP_K}", "Chunks"],
        rows_2a,
        title="2a — MinHash num_perm Sensitivity",
    )
    _save_csv(
        "exp2a_num_perm.csv",
        ["num_perm", "Index_ms", "Avg_Query_ms", f"Avg_P@{TOP_K}", "Chunks"],
        rows_2a,
    )

    # ── 2b: LSH num_bands ──────────────────────────────────────────────────
    print("\n  2b: LSH — num_bands (128 perms fixed)")
    print("      More bands → lower effective threshold, higher recall, more candidates.\n")

    # For 128 perms, valid divisors: 8, 16, 32, 64
    NUM_BANDS = [8, 16, 32, 64]
    rows_2b: List[List[Any]] = []

    for nb in NUM_BANDS:
        rows_per_band = 128 // nb
        t0 = time.perf_counter()
        lsh_idx, chunk_texts = _build_lsh_custom(corpus, num_perm=128, num_bands=nb)
        idx_ms = (time.perf_counter() - t0) * 1000.0

        precs, lats, hits = [], [], []
        for q in probe_queries:
            gt = tfidf.search(q, top_k=TOP_K)
            res, ms, _ = _measure(_query_lsh_custom, lsh_idx, chunk_texts, q, 128, 0.05, TOP_K)
            precs.append(precision_at_k(gt, res, TOP_K))
            lats.append(ms)
            hits.append(len(res))

        rows_2b.append([
            nb,
            rows_per_band,
            f"{idx_ms:.0f}",
            f"{_avg(lats):.2f}",
            f"{_avg(hits):.1f}",
            f"{_avg(precs):.3f}",
        ])

    _print_table(
        ["num_bands", "rows/band", "Index_ms", "Avg_Query_ms",
         f"Avg_Hits", f"Avg_P@{TOP_K}"],
        rows_2b,
        title="2b — LSH num_bands Sensitivity (128 perms)",
    )
    _save_csv(
        "exp2b_num_bands.csv",
        ["num_bands", "rows_per_band", "Index_ms", "Avg_Query_ms",
         "Avg_Hits", f"Avg_P@{TOP_K}"],
        rows_2b,
    )

    # ── 2c: SimHash Hamming threshold ──────────────────────────────────────
    print("\n  2c: SimHash — Hamming distance threshold")
    print("      Lower threshold → stricter matching, fewer results.\n")

    HAMMING_THRESHOLDS = [10, 20, 30, 40, 50]
    simhash_r = _build_simhash(corpus)
    rows_2c: List[List[Any]] = []

    for ht in HAMMING_THRESHOLDS:
        min_sim_pct = f"{(64 - ht) / 64 * 100:.0f}%"
        precs, lats, hits = [], [], []
        for q in probe_queries:
            gt = tfidf.search(q, top_k=TOP_K)
            res, ms, _ = _measure(
                simhash_r.search,
                q,
                method=SimilarityMethod.SIMHASH,
                simhash_max_distance=ht,
                top_k=TOP_K,
            )
            precs.append(precision_at_k(gt, res, TOP_K))
            lats.append(ms)
            hits.append(len(res))

        rows_2c.append([
            ht,
            min_sim_pct,
            f"{_avg(lats):.2f}",
            f"{_avg(hits):.1f}",
            f"{_avg(precs):.3f}",
        ])

    _print_table(
        ["Hamming_Thresh", "Min_Sim%", "Avg_Query_ms", "Avg_Hits", f"Avg_P@{TOP_K}"],
        rows_2c,
        title="2c — SimHash Hamming Threshold Sensitivity",
    )
    _save_csv(
        "exp2c_simhash_threshold.csv",
        ["Hamming_Thresh", "Min_Sim%", "Avg_Query_ms", "Avg_Hits", f"Avg_P@{TOP_K}"],
        rows_2c,
    )


# ===========================================================================
# EXPERIMENT 3 — Scalability Test
# ===========================================================================

def experiment_3(corpus: Dict[str, Any]) -> None:
    print("\n" + "#" * 72)
    print("# EXPERIMENT 3: Scalability Test (corpus duplication)")
    print("# Duplicates chunks to simulate 1×, 2×, 5×, 10× dataset size.")
    print("# Shows how TF-IDF O(N·V) and LSH sub-linear complexity diverge.")
    print("#" * 72)

    SCALE_FACTORS = [1, 2, 5, 10]
    probe_queries = EVAL_QUERIES[:3]
    rows: List[List[Any]] = []

    for factor in SCALE_FACTORS:
        scaled = _scale_corpus(corpus, factor)
        n = len(scaled.get("texts", []))
        print(f"\n  Scale {factor}× — {n} chunks")

        # Index build times
        t0 = time.perf_counter()
        tfidf = _build_tfidf(scaled)
        tfidf_idx_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        lsh_idx, chunk_texts = _build_lsh_custom(scaled, num_perm=128)
        lsh_idx_ms = (time.perf_counter() - t0) * 1000.0

        # Query latency
        tfidf_lats, lsh_lats = [], []
        for q in probe_queries:
            _, t_ms, _ = _measure(tfidf.search, q, top_k=TOP_K)
            tfidf_lats.append(t_ms)
            _, l_ms, _ = _measure(_query_lsh_custom, lsh_idx, chunk_texts, q, 128, 0.05, TOP_K)
            lsh_lats.append(l_ms)

        # Index memory (measure tracemalloc during a single search as proxy)
        _, _, tfidf_mem = _measure(tfidf.search, probe_queries[0], top_k=TOP_K)
        _, _, lsh_mem = _measure(
            _query_lsh_custom, lsh_idx, chunk_texts, probe_queries[0], 128, 0.05, TOP_K
        )

        rows.append([
            f"{factor}×",
            n,
            f"{tfidf_idx_ms:.0f}",
            f"{lsh_idx_ms:.0f}",
            f"{_avg(tfidf_lats):.2f}",
            f"{_avg(lsh_lats):.2f}",
            f"{tfidf_mem:.0f}",
            f"{lsh_mem:.0f}",
        ])
        print(f"    TF-IDF index: {tfidf_idx_ms:.0f} ms | LSH index: {lsh_idx_ms:.0f} ms")
        print(
            f"    Avg TF-IDF query: {_avg(tfidf_lats):.2f} ms | "
            f"Avg LSH query: {_avg(lsh_lats):.2f} ms"
        )

    _print_table(
        [
            "Scale", "N_Chunks",
            "TFIDF_Idx_ms", "LSH_Idx_ms",
            "TFIDF_Query_ms", "LSH_Query_ms",
            "TFIDF_Qry_KB", "LSH_Qry_KB",
        ],
        rows,
        title="Experiment 3 — Scalability: TF-IDF vs LSH",
    )
    _save_csv(
        "exp3_scalability.csv",
        [
            "Scale", "N_Chunks",
            "TFIDF_Idx_ms", "LSH_Idx_ms",
            "TFIDF_Query_ms", "LSH_Query_ms",
            "TFIDF_Qry_KB", "LSH_Qry_KB",
        ],
        rows,
    )


# ===========================================================================
# QUALITATIVE EVALUATION — 15 sample queries
# ===========================================================================

def qualitative_evaluation(corpus: Dict[str, Any]) -> None:
    print("\n" + "#" * 72)
    print("# QUALITATIVE EVALUATION — 15 Sample Queries")
    print("# Shows top-1 result from TF-IDF and LSH per query with text preview.")
    print("#" * 72)

    tfidf = _build_tfidf(corpus)
    lsh_idx, chunk_texts = _build_lsh_custom(corpus, num_perm=128)

    rows: List[List[Any]] = []

    for i, query in enumerate(EVAL_QUERIES, 1):
        gt = tfidf.search(query, top_k=TOP_K)
        lsh_res = _query_lsh_custom(lsh_idx, chunk_texts, query, 128, 0.05, TOP_K)

        top_tfidf_id = gt[0].chunk_id if gt else "—"
        top_tfidf_sc = f"{gt[0].similarity_score:.4f}" if gt else "—"
        top_lsh_id = lsh_res[0].chunk_id if lsh_res else "no results"
        top_lsh_sc = f"{lsh_res[0].similarity_score:.4f}" if lsh_res else "—"
        p3 = f"{precision_at_k(gt, lsh_res, 3):.2f}"

        print(f"\n  Q{i:02d}: {query}")
        print(f"       TF-IDF  → [{top_tfidf_id}]  score={top_tfidf_sc}")
        if gt:
            preview = gt[0].text[:180].replace("\n", " ")
            print(f"       Preview : {preview}...")
        print(f"       LSH     → [{top_lsh_id}]  score={top_lsh_sc}  P@3={p3}")

        rows.append([
            i, query,
            top_tfidf_id, top_tfidf_sc,
            top_lsh_id, top_lsh_sc,
            p3,
        ])

    _print_table(
        ["#", "Query", "TFIDF_Chunk", "TFIDF_Score", "LSH_Chunk", "LSH_Score", "P@3"],
        rows,
        title="Qualitative — Top-1 Results per Query",
    )
    _save_csv(
        "qualitative_15_queries.csv",
        ["#", "Query", "TFIDF_Chunk", "TFIDF_Score", "LSH_Chunk", "LSH_Score", "P@3"],
        rows,
    )


# ===========================================================================
# MAIN
# ===========================================================================

if __name__ == "__main__":
    print("=" * 72)
    print("  Scalable Academic Policy QA System — Evaluation Suite")
    print(f"  Queries: {len(EVAL_QUERIES)}   k: {TOP_K}   Results → {RESULTS_DIR}")
    print("=" * 72)

    print("\nLoading corpus from handbooks...")
    corpus = ingest_corpus()
    n_chunks = len(corpus.get("texts", []))
    print(f"Corpus ready: {n_chunks} chunks\n")

    experiment_1(corpus)
    experiment_2(corpus)
    experiment_3(corpus)
    qualitative_evaluation(corpus)

    print("\n" + "=" * 72)
    print("  All experiments complete.")
    print(f"  CSVs saved in: {RESULTS_DIR}")
    print("=" * 72)
