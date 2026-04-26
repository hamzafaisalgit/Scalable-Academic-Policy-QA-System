"""
evaluate.py — Experimental Evaluation Suite 
"""

from __future__ import annotations

import csv
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Make sure src/ is on the import path
sys.path.insert(0, str(Path(__file__).parent))

from ingestion import ingest_corpus
from minhash_lsh import LocalitySensitiveHash, MinHash, shingle_text
from retrieval import HybridRetriever, SimilarityMethod, SimilarityResult
from tfidf import TFIDFRetriever

# Output directory
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# 15 evaluation queries
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

TOP_K = 10


# Utility helpers

def _measure(fn, *args, **kwargs) -> Tuple[Any, float, float]:
    """Run fn and measure elapsed time (ms) and peak memory (KB)."""
    if tracemalloc.is_tracing():
        tracemalloc.stop()
    tracemalloc.start()
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, elapsed_ms, peak_bytes / 1024.0


def _avg(lst: List[float]) -> float:
    return sum(lst) / len(lst) if lst else 0.0


def precision_at_k(
    ground_truth: List[SimilarityResult],
    predicted: List[SimilarityResult],
    k: int,
) -> float:
    """
    FIXED: Properly handle empty results.
    Precision@k = (# of relevant items in top-k) / (# of predicted items in top-k)
    """
    if not predicted:
        return 0.0 
    
    if not ground_truth:
        return 0.0  
    # Get text from ground truth top-k
    gt_texts = set(r.text.lower().strip() for r in ground_truth[:k])
    
    matches = 0
    for p in predicted[:k]:
        pred_text_lower = p.text.lower().strip()
        if pred_text_lower in gt_texts:
            matches += 1
    
    # Precision = matches / number of predictions
    return matches / min(k, len(predicted))


def recall_at_k(
    ground_truth: List[SimilarityResult],
    predicted: List[SimilarityResult],
    k: int,
) -> float:
    """
     Properly handle empty results.
    Recall@k = (# of relevant items in top-k) / (total ground truth items in top-k)
    """
    if not ground_truth:
        return 0.0
    
    if not predicted:
        return 0.0
    
    # Get text from ground truth top-k
    gt_texts = set(r.text.lower().strip() for r in ground_truth[:k])
    
    # Count how many ground truth items appear in predicted top-k
    pred_texts = set(r.text.lower().strip() for r in predicted[:k])
    
    matches = len(gt_texts & pred_texts)
    
    # Recall = matches / total ground truth items
    return matches / len(gt_texts)


def _print_table(
    headers: List[str],
    rows: List[List[Any]],
    title: str = "",
) -> None:
    if title:
        print(f"\n{'=' * 80}")
        print(f"  {title}")
        print(f"{'=' * 80}")
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


# Corpus builders

def _build_tfidf(corpus: Dict[str, Any]) -> TFIDFRetriever:
    r = TFIDFRetriever()
    r.fit_corpus(corpus)
    return r


def _build_hybrid_lsh(corpus: Dict[str, Any]) -> HybridRetriever:
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


def _build_lsh_custom(
    corpus: Dict[str, Any],
    num_perm: int,
    num_bands: Optional[int] = None,
    threshold: float = 0.00001, 
) -> Tuple[LocalitySensitiveHash, Dict[str, str]]:
    """
    Build LSH index with CORRECTED threshold.
    
    FIX EXPLANATION:
    Academic text has very low word overlap because of paraphrasing:
    - Query: "minimum GPA"
    - Doc: "minimum grade point average"
    
    These share only "minimum", so Jaccard similarity is < 0.3
    To capture these, we need threshold << 0.3, hence 0.00001
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
    threshold: float = 0.00001,  
    top_k: int = TOP_K,
) -> List[SimilarityResult]:
    """
    Query LSH with EXTREMELY LOW threshold.
    
    This is the key fix: threshold controls recall vs precision tradeoff.
    Lower threshold = catch more candidates = higher recall, lower precision
    """
    shingles = shingle_text(query)
    sig = MinHash(num_perm=num_perm)
    sig.update(shingles)
    
    # DIAGNOSTIC: Print what we're querying with
    # print(f"  [DEBUG] Query shingles count: {len(shingles)}, threshold={threshold}")
    
    raw = lsh_index.query(sig, threshold=threshold)
    
    # DIAGNOSTIC: Print results
    # print(f"  [DEBUG] LSH returned {len(raw)} candidates")
    
    return [
        SimilarityResult(
            chunk_id=doc_id,
            text=chunk_texts[doc_id],
            similarity_score=score,
            method="minhash_lsh",
        )
        for doc_id, score in raw[:top_k]
    ]


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


# EXPERIMENT 1 — Exact vs Approximate Retrieval 

def experiment_1(corpus: Dict[str, Any]) -> None:
    print("\n" + "#" * 80)
    print("# EXPERIMENT 1: Exact (TF-IDF) vs Approximate (LSH / SimHash)")
    print("# FIX: LSH threshold lowered to 0.00001 (was 0.001)")
    print("#" * 80)

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

    for i, query in enumerate(EVAL_QUERIES, 1):
        print(f"  [{i}/{len(EVAL_QUERIES)}] {query[:50]}...")
        
        gt, tfidf_ms, tfidf_mem = _measure(tfidf.search, query, top_k=TOP_K)

        # FIX: Use threshold=0.00001 instead of 0.001
        lsh_res, lsh_ms, lsh_mem = _measure(
            lsh.search,
            query,
            method=SimilarityMethod.MINHASH_LSH,
            minhash_threshold=0.00001, 
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

        # Diagnostic output
        if p_lsh > 0 or len(lsh_res) > 0:
            print(f"       ✓ LSH got {len(lsh_res)} results, P@{TOP_K}={p_lsh:.2f}")
        else:
            print(f"       ✗ LSH got 0 results")

        short_q = (query[:40] + "...") if len(query) > 40 else query
        rows.append([
            short_q,
            f"{tfidf_ms:.1f}",
            f"{lsh_ms:.1f}",
            f"{sh_ms:.1f}",
            f"{p_lsh:.2f}",
            f"{r_lsh:.2f}",
            f"{p_sh:.2f}",
            f"{r_sh:.2f}",
            f"{len(lsh_res):2d}",
        ])

    headers = [
        "Query",
        "TFIDF_ms", "LSH_ms", "SH_ms",
        f"P@{TOP_K}_LSH", f"R@{TOP_K}_LSH",
        f"P@{TOP_K}_SH", f"R@{TOP_K}_SH",
        "LSH_Hits",
    ]
    _print_table(headers, rows, title=f"Experiment 1 — Per-Query Results (k={TOP_K})")

    # Summary
    print(f"\n  {'─'*70}")
    print(f"  SUMMARY ")
    print(f"  {'─'*70}")
    print(f"  Avg TF-IDF query latency  : {_avg(tfidf_ms_all):6.2f} ms")
    print(f"  Avg LSH query latency     : {_avg(lsh_ms_all):6.2f} ms")
    print(f"  Avg SimHash query latency : {_avg(sh_ms_all):6.2f} ms")
    print(f"  Avg Precision@{TOP_K} TF-IDF   : {1.0:.3f} (ground truth)")
    print(f"  Avg Precision@{TOP_K} LSH      : {_avg(prec_lsh_all):.3f} ")
    print(f"  Avg Recall@{TOP_K} LSH        : {_avg(rec_lsh_all):.3f} ")
    print(f"  Avg Precision@{TOP_K} SimHash  : {_avg(prec_sh_all):.3f}")
    print(f"  Avg Recall@{TOP_K} SimHash    : {_avg(rec_sh_all):.3f}")
    print(f"  Avg TF-IDF query memory   : {_avg(tfidf_mem_all):6.1f} KB")
    print(f"  Avg LSH query memory      : {_avg(lsh_mem_all):6.1f} KB")

    avg_row: List[Any] = [
        "AVERAGE",
        f"{_avg(tfidf_ms_all):.1f}", f"{_avg(lsh_ms_all):.1f}", f"{_avg(sh_ms_all):.1f}",
        f"{_avg(prec_lsh_all):.2f}", f"{_avg(rec_lsh_all):.2f}",
        f"{_avg(prec_sh_all):.2f}", f"{_avg(rec_sh_all):.2f}",
        f"{_avg([len(r) for r in rows]):2.0f}",
    ]
    _save_csv("exp1_exact_vs_approx.csv", headers, rows + [avg_row])


# EXPERIMENT 2 — Parameter Sensitivity 

def experiment_2(corpus: Dict[str, Any]) -> None:
    print("\n" + "#" * 80)
    print("# EXPERIMENT 2: Parameter Sensitivity Analysis")
    print("#" * 80)

    probe_queries = EVAL_QUERIES[:5]
    tfidf = _build_tfidf(corpus)

    # ── 2a: MinHash num_perm 
    print("\n  2a: MinHash — num_perm sensitivity\n")
    NUM_PERMS = [32, 64, 128, 256]
    rows_2a: List[List[Any]] = []

    for np in NUM_PERMS:
        t0 = time.perf_counter()
        lsh_idx, chunk_texts = _build_lsh_custom(corpus, num_perm=np, threshold=0.00001)
        idx_ms = (time.perf_counter() - t0) * 1000.0

        precs, lats, hits_list = [], [], []
        for q in probe_queries:
            gt = tfidf.search(q, top_k=TOP_K)
            res, ms, _ = _measure(_query_lsh_custom, lsh_idx, chunk_texts, q, np, 0.00001, TOP_K)
            precs.append(precision_at_k(gt, res, TOP_K))
            lats.append(ms)
            hits_list.append(len(res))

        rows_2a.append([
            np,
            f"{idx_ms:.0f}",
            f"{_avg(lats):.2f}",
            f"{_avg(precs):.3f}",
            f"{_avg(hits_list):.1f}",
        ])

    _print_table(
        ["num_perm", "Index_ms", f"Avg_Query_ms", f"Avg_P@{TOP_K}", "Avg_Hits"],
        rows_2a,
        title="2a — MinHash num_perm Sensitivity (threshold=0.00001)",
    )
    _save_csv(
        "exp2a_num_perm.csv",
        ["num_perm", "Index_ms", "Avg_Query_ms", f"Avg_P@{TOP_K}", "Avg_Hits"],
        rows_2a,
    )

    # ── 2b: LSH num_bands ──────────────────────────────────────────────────
    print("\n  2b: LSH — num_bands sensitivity (128 perms)\n")
    NUM_BANDS = [8, 16, 32, 64]
    rows_2b: List[List[Any]] = []

    for nb in NUM_BANDS:
        rows_per_band = 128 // nb
        t0 = time.perf_counter()
        lsh_idx, chunk_texts = _build_lsh_custom(corpus, num_perm=128, num_bands=nb, threshold=0.00001)
        idx_ms = (time.perf_counter() - t0) * 1000.0

        precs, lats, hits = [], [], []
        for q in probe_queries:
            gt = tfidf.search(q, top_k=TOP_K)
            res, ms, _ = _measure(_query_lsh_custom, lsh_idx, chunk_texts, q, 128, 0.00001, TOP_K)
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
        ["num_bands", "rows/band", "Index_ms", "Avg_Query_ms", "Avg_Hits", f"Avg_P@{TOP_K}"],
        rows_2b,
        title="2b — LSH num_bands Sensitivity (threshold=0.00001)",
    )
    _save_csv(
        "exp2b_num_bands.csv",
        ["num_bands", "rows_per_band", "Index_ms", "Avg_Query_ms", "Avg_Hits", f"Avg_P@{TOP_K}"],
        rows_2b,
    )

    # ── 2c: SimHash Hamming threshold ──────────────────────────────────────
    print("\n  2c: SimHash — Hamming threshold sensitivity\n")
    HAMMING_THRESHOLDS = [10, 20, 30, 40, 50]
    simhash_r = _build_simhash(corpus)
    rows_2c: List[List[Any]] = []

    for ht in HAMMING_THRESHOLDS:
        min_sim_pct = f"{(64 - ht) / 64 * 100:.0f}%"
        precs, lats, hits = [], [], []
        for q in probe_queries:
            gt = tfidf.search(q, top_k=TOP_K)
            res, ms, _ = _measure(
                simhash_r.search, q,
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


# EXPERIMENT 3 — Scalability

def experiment_3(corpus: Dict[str, Any]) -> None:
    print("\n" + "#" * 80)
    print("# EXPERIMENT 3: Scalability Test ")
    print("#" * 80)

    SCALE_FACTORS = [1, 2, 5, 10]
    probe_queries = EVAL_QUERIES[:3]
    rows: List[List[Any]] = []

    for factor in SCALE_FACTORS:
        scaled = _scale_corpus(corpus, factor)
        n = len(scaled.get("texts", []))
        print(f"\n  Scale {factor}× — {n} chunks")

        t0 = time.perf_counter()
        tfidf = _build_tfidf(scaled)
        tfidf_idx_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        lsh_idx, chunk_texts = _build_lsh_custom(scaled, num_perm=128, threshold=0.00001)
        lsh_idx_ms = (time.perf_counter() - t0) * 1000.0

        tfidf_lats, lsh_lats = [], []
        for q in probe_queries:
            _, t_ms, _ = _measure(tfidf.search, q, top_k=TOP_K)
            tfidf_lats.append(t_ms)
            _, l_ms, _ = _measure(_query_lsh_custom, lsh_idx, chunk_texts, q, 128, 0.00001, TOP_K)
            lsh_lats.append(l_ms)

        _, _, tfidf_mem = _measure(tfidf.search, probe_queries[0], top_k=TOP_K)
        _, _, lsh_mem = _measure(
            _query_lsh_custom, lsh_idx, chunk_texts, probe_queries[0], 128, 0.00001, TOP_K
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
        print(f"    TF-IDF query: {_avg(tfidf_lats):.2f} ms | LSH query: {_avg(lsh_lats):.2f} ms")

    _print_table(
        [
            "Scale", "N_Chunks",
            "TFIDF_Idx", "LSH_Idx",
            "TFIDF_Query", "LSH_Query",
            "TFIDF_KB", "LSH_KB",
        ],
        rows,
        title="Experiment 3 — Scalability (threshold=0.00001)",
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


# MAIN

if __name__ == "__main__":
    print("=" * 80)
    print("  EVALUATION SUITE ")
    print(f"  Queries: {len(EVAL_QUERIES)}   k: {TOP_K}")
    print(f"  Results → {RESULTS_DIR}")
    print("=" * 80)

    print("\nLoading corpus...")
    corpus = ingest_corpus()
    n_chunks = len(corpus.get("texts", []))
    print(f"Corpus ready: {n_chunks} chunks\n")

    experiment_1(corpus)
    experiment_2(corpus)
    experiment_3(corpus)

    print("\n" + "=" * 80)
    print("  ✓ All experiments complete!")
    print(f"  Saved to: {RESULTS_DIR}")
    print("=" * 80)
