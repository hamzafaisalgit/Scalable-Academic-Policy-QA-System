
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set

from minhash_lsh import (
    LocalitySensitiveHash,
    MinHash,
    get_minhash_signature,
)
from simhash_impl import SimHashIndex, get_simhash


class SimilarityMethod(Enum):
    """Available similarity detection methods."""
    MINHASH_LSH = "minhash_lsh"
    SIMHASH = "simhash"
    HYBRID = "hybrid"


@dataclass
class SimilarityResult:
    """Result of a similarity search."""
    chunk_id: str
    text: str
    similarity_score: float
    method: str  # Which method detected this match


class HybridRetriever:
    """
    Hybrid similarity and indexing system combining MinHash+LSH and SimHash.
    
    Provides efficient multi-method similarity search for document chunks,
    enabling both Jaccard-based and fingerprint-based similarity detection.
    """
    
    def __init__(
        self,
        method: SimilarityMethod = SimilarityMethod.HYBRID,
        minhash_num_perm: int = 128,
        minhash_num_bands: Optional[int] = None,
        simhash_size: int = 64,
        simhash_samples: int = 4,
    ):
        """
        Initialize hybrid retriever.
        
        Args:
            method: Which similarity method to use
            minhash_num_perm: MinHash permutations
            minhash_num_bands: MinHash bands (default: sqrt(num_perm))
            simhash_size: SimHash fingerprint size in bits
            simhash_samples: Number of samples for SimHash indexing
        """
        self.method = method
        
        # Initialize MinHash+LSH if needed
        if method in (SimilarityMethod.MINHASH_LSH, SimilarityMethod.HYBRID):
            self.lsh = LocalitySensitiveHash(
                num_perm=minhash_num_perm,
                num_bands=minhash_num_bands
            )
        else:
            self.lsh = None
        
        # Initialize SimHash if needed
        if method in (SimilarityMethod.SIMHASH, SimilarityMethod.HYBRID):
            self.simhash_index = SimHashIndex(
                fingerprint_size=simhash_size,
                num_samples=simhash_samples
            )
        else:
            self.simhash_index = None
        
        # Store document chunks
        self.chunks: Dict[str, str] = {}
    
    def add_chunk(self, chunk_id: str, text: str) -> None:
        """
        Add a text chunk to the index.
        
        Args:
            chunk_id: Unique chunk identifier
            text: Chunk text content
        """
        self.chunks[chunk_id] = text
        
        # Index with MinHash+LSH
        if self.lsh is not None:
            sig = get_minhash_signature(text)
            self.lsh.add_document(chunk_id, sig)
        
        # Index with SimHash
        if self.simhash_index is not None:
            sh = get_simhash(text)
            self.simhash_index.add_document(chunk_id, sh)
    
    def search(
        self,
        query: str,
        method: Optional[SimilarityMethod] = None,
        minhash_threshold: float = 0.1,
        simhash_max_distance: int = 40,
        top_k: int = 10,
    ) -> List[SimilarityResult]:
        """
        Search for similar chunks using configured similarity method(s).
        
        Args:
            query: Query text
            method: Override retriever's method (uses self.method if None)
            minhash_threshold: MinHash similarity threshold (0-1), default 0.1 for short queries
            simhash_max_distance: SimHash max Hamming distance, default 40 (~63% of 64 bits) for very lenient matching
            top_k: Return top k results
        
        Returns:
            List of SimilarityResult sorted by similarity (highest first)
        """
        search_method = method or self.method
        results = {}  # chunk_id -> (max_similarity, method_names)
        
        # Search using MinHash+LSH
        if search_method in (SimilarityMethod.MINHASH_LSH, SimilarityMethod.HYBRID):
            if self.lsh is None:
                raise ValueError("MinHash+LSH not initialized")
            
            query_sig = get_minhash_signature(query)
            minhash_results = self.lsh.query(query_sig, threshold=minhash_threshold)
            
            for chunk_id, score in minhash_results:
                if chunk_id not in results:
                    results[chunk_id] = (score, [])
                old_score, methods = results[chunk_id]
                results[chunk_id] = (max(old_score, score), methods + ["minhash"])
        
        # Search using SimHash
        if search_method in (SimilarityMethod.SIMHASH, SimilarityMethod.HYBRID):
            if self.simhash_index is None:
                raise ValueError("SimHash not initialized")
            
            query_sh = get_simhash(query)
            simhash_results = self.simhash_index.query(query_sh, max_distance=simhash_max_distance)
            
            for chunk_id, score in simhash_results:
                if chunk_id not in results:
                    results[chunk_id] = (score, [])
                old_score, methods = results[chunk_id]
                results[chunk_id] = (max(old_score, score), methods + ["simhash"])
        
        # Convert to SimilarityResult objects
        result_objs = [
            SimilarityResult(
                chunk_id=chunk_id,
                text=self.chunks[chunk_id],
                similarity_score=score,
                # sorted() makes the method string deterministic regardless of
                # which retrieval method found the chunk first
                method="+".join(sorted(set(methods)))
            )
            for chunk_id, (score, methods) in results.items()
        ]
        
        # Sort by similarity descending
        result_objs.sort(key=lambda x: x.similarity_score, reverse=True)
        
        return result_objs[:top_k]
    
    def get_candidates(
        self,
        query: str,
        method: Optional[SimilarityMethod] = None,
        minhash_threshold: float = 0.05,
    ) -> set:
        """
        Get candidate chunks for a query (without computing full similarity).
        
        Useful for initial filtering before more expensive operations.
        
        Args:
            query: Query text
            method: Override retriever's method
            minhash_threshold: LSH threshold for candidates (default 0.05 for lenient filtering)
        
        Returns:
            Set of candidate chunk IDs
        """
        search_method = method or self.method
        candidates = set()
        
        if search_method in (SimilarityMethod.MINHASH_LSH, SimilarityMethod.HYBRID):
            query_sig = get_minhash_signature(query)
            candidates.update(self.lsh.get_candidates(query_sig, threshold=minhash_threshold))
        
        if search_method in (SimilarityMethod.SIMHASH, SimilarityMethod.HYBRID):
            query_sh = get_simhash(query)
            # For SimHash, we approximate candidates from first sample table
            fingerprint = query_sh.get_fingerprint()
            sample_size = self.simhash_index.sample_size
            sample_val = fingerprint & ((1 << sample_size) - 1)
            candidates.update(self.simhash_index.tables[0].get(sample_val, set()))
        
        return candidates

    def get_chunk_ids(self) -> List[str]:
        """
        Return all chunk IDs currently indexed.

        Used by ``pagerank.build_pagerank_from_lsh()`` so it can iterate over
        chunks without accessing ``lsh.doc_signatures`` keys directly.
        """
        return list(self.chunks.keys())

    def get_chunk_text(self, chunk_id: str) -> str:
        """
        Return the raw text for a given *chunk_id*.
        Raises ``KeyError`` if the chunk is not in the index.
        """
        return self.chunks[chunk_id]


if __name__ == "__main__":
    # Example usage
    print("=== Hybrid Similarity & Indexing System ===\n")
    
    # Sample chunks from policy documents
    chunks = {
        "chunk1": "academic probation is imposed on students with poor performance",
        "chunk2": "students on academic probation must improve their grades",
        "chunk3": "hostel allotment is based on merit and availability",
        "chunk4": "hostel fees vary by room type and occupancy",
        "chunk5": "examination dates are announced one month in advance",
    }
    
    # Initialize hybrid retriever
    retriever = HybridRetriever(
        method=SimilarityMethod.HYBRID,
        minhash_num_perm=128,
        simhash_size=64
    )
    
    # Index chunks
    for chunk_id, text in chunks.items():
        retriever.add_chunk(chunk_id, text)
    
    # Test MinHash+LSH search
    print("--- MinHash+LSH Search ---")
    query = "academic performance and probation"
    results = retriever.search(
        query,
        method=SimilarityMethod.MINHASH_LSH,
        minhash_threshold=0.2,
        top_k=3
    )
    for result in results:
        print(f"{result.chunk_id}: {result.similarity_score:.2%}")
        print(f"  Text: {result.text[:60]}...")
        print(f"  Method: {result.method}\n")
    
    # Test SimHash search
    print("--- SimHash Search ---")
    results = retriever.search(
        query,
        method=SimilarityMethod.SIMHASH,
        simhash_max_distance=10,
        top_k=3
    )
    for result in results:
        print(f"{result.chunk_id}: {result.similarity_score:.2%}")
        print(f"  Text: {result.text[:60]}...")
        print(f"  Method: {result.method}\n")
    
    # Test hybrid search
    print("--- Hybrid Search ---")
    results = retriever.search(
        query,
        method=SimilarityMethod.HYBRID,
        top_k=3
    )
    for result in results:
        print(f"{result.chunk_id}: {result.similarity_score:.2%}")
        print(f"  Text: {result.text[:60]}...")
        print(f"  Method: {result.method}\n")
