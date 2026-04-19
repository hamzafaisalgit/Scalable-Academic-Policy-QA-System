from __future__ import annotations

from typing import List, Optional, Set, Tuple

from datasketch import MinHash as DataSketchMinHash
from datasketch import MinHashLSH


class MinHash:
    """
    Wrapper around datasketch.MinHash for Jaccard similarity.
    
    Uses industry-standard implementation for efficient approximate similarity detection.
    """
    
    def __init__(self, num_perm: int = 128, seed: int = 0):
        """
        Initialize MinHash with specified number of permutations.
        
        Args:
            num_perm: Number of hash functions (higher = more accurate)
            seed: Random seed for reproducibility
        """
        self.num_perm = num_perm
        self.seed = seed
        self._minhash = DataSketchMinHash(num_perm=num_perm, seed=seed)
        self._is_empty = True
    
    @property
    def hash_values(self) -> List[int]:
        """Get current hash values for compatibility with tests."""
        # datasketch stores hashvalues as a list internally
        return list(self._minhash.hashvalues)
    
    def update(self, items: set[str]) -> None:
        """
        Update MinHash signature with new items (set elements).
        
        Args:
            items: Set of strings representing elements (e.g., shingles)
        """
        for item in items:
            self._minhash.update(item.encode('utf-8'))
        if items:
            self._is_empty = False
    
    def jaccard_similarity(self, other: MinHash) -> float:
        """
        Estimate Jaccard similarity with another MinHash signature.
        
        Returns:
            Estimated Jaccard similarity (0-1)
        """
        return self._minhash.jaccard(other._minhash)
    
    def get_signature(self) -> bytes:
        """Get hash signature as bytes."""
        return self._minhash.digest()
    
    def __repr__(self) -> str:
        return f"MinHash(num_perm={self.num_perm})"


class LocalitySensitiveHash:
    """
    LSH wrapper using datasketch.MinHashLSH for efficient candidate retrieval.
    
    Automatically configures optimal band/row parameters for high recall.
    """
    
    def __init__(self, num_perm: int = 128, num_bands: Optional[int] = None):
        """
        Initialize LSH with band configuration.
        
        Args:
            num_perm: Number of MinHash permutations
            num_bands: Number of bands (default: auto-configured for best recall)
        """
        self.num_perm = num_perm
        
        # Auto-configure bands for high recall (more bands = more collisions)
        if num_bands is None:
            # Use more bands with fewer rows per band for better recall
            # For 128: use 64 bands × 2 rows (0.49 collision probability per band at 70% similarity)
            self.num_bands = num_perm // 2
        else:
            self.num_bands = num_bands
        
        self.rows_per_band = num_perm // self.num_bands
        
        # Validate configuration
        if self.rows_per_band * self.num_bands != num_perm:
            raise ValueError(
                f"num_perm ({num_perm}) must be divisible by num_bands ({self.num_bands})"
            )
        
        # Initialize datasketch LSH with explicit band structure.
        # Using params=(b, r) instead of threshold so the num_bands/rows_per_band
        # we computed above are the ones actually used by the index.
        self.lsh = MinHashLSH(num_perm=num_perm, params=(self.num_bands, self.rows_per_band))
        self.doc_signatures: dict[str, MinHash] = {}
    
    def add_document(self, doc_id: str, minhash: MinHash) -> None:
        """
        Add a document's MinHash signature to the LSH index.
        
        Args:
            doc_id: Unique document identifier
            minhash: MinHash signature for the document
        """
        self.doc_signatures[doc_id] = minhash
        self.lsh.insert(doc_id, minhash._minhash)
    
    def get_candidates(self, minhash: MinHash, threshold: float = 0.5) -> Set[str]:
        """
        Get candidate documents that might be similar.
        
        Args:
            minhash: Query MinHash signature
            threshold: Similarity threshold (optional, used for filtering)
        
        Returns:
            Set of candidate document IDs
        """
        # LSH query returns candidates without filtering by threshold
        return set(self.lsh.query(minhash._minhash))
    
    def query(self, minhash: MinHash, threshold: float = 0.5) -> List[Tuple[str, float]]:
        """
        Find similar documents using LSH + MinHash similarity.
        
        Args:
            minhash: Query MinHash signature
            threshold: Minimum Jaccard similarity threshold
        
        Returns:
            List of (doc_id, similarity) tuples, sorted by similarity
        """
        candidates = self.get_candidates(minhash)
        results = []
        
        for doc_id in candidates:
            similarity = minhash.jaccard_similarity(self.doc_signatures[doc_id])
            if similarity >= threshold:
                results.append((doc_id, similarity))
        
        # Sort by similarity descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results


def shingle_text(text: str, shingle_size: int = 4) -> Set[str]:
    """
    Convert text to shingles (k-grams) for set representation.
    
    Args:
        text: Input text
        shingle_size: Size of each shingle
    
    Returns:
        Set of shingles
    """
    # Normalize text
    text = text.lower().strip()
    
    # Create shingles
    shingles = set()
    for i in range(len(text) - shingle_size + 1):
        shingle = text[i:i + shingle_size]
        shingles.add(shingle)
    
    return shingles


def get_minhash_signature(text: str, num_perm: int = 128, shingle_size: int = 4) -> MinHash:
    """
    Generate MinHash signature for text.
    
    Args:
        text: Input text
        num_perm: Number of permutations for MinHash
        shingle_size: Size of shingles
    
    Returns:
        MinHash signature
    """
    shingles = shingle_text(text, shingle_size)
    minhash = MinHash(num_perm=num_perm)
    minhash.update(shingles)
    return minhash


if __name__ == "__main__":
    # Example usage
    docs = {
        "doc1": "the quick brown fox jumps over the lazy dog",
        "doc2": "the quick brown fox jumps over a lazy dog",
        "doc3": "machine learning is fascinating and powerful",
    }
    
    # Create LSH index
    lsh = LocalitySensitiveHash(num_perm=128)
    
    # Index documents
    for doc_id, text in docs.items():
        sig = get_minhash_signature(text)
        lsh.add_document(doc_id, sig)
    
    # Query
    query_text = "the quick brown fox jumps"
    query_sig = get_minhash_signature(query_text)
    results = lsh.query(query_sig, threshold=0.3)
    
    print("Query results:")
    for doc_id, similarity in results:
        print(f"  {doc_id}: {similarity:.2%}")
