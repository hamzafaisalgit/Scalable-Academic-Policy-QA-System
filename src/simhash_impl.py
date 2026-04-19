"""
SimHash implementation using simhash-py library for efficient similarity detection.

Uses industry-standard Simhash implementation for fingerprint-based similarity.
"""

from __future__ import annotations

from typing import List, Set, Tuple

# Import the simhash library (no shadowing since this file is named differently)
from simhash import Simhash as SimHashLib


class SimHash:
    """
    Wrapper around simhash-py library for fingerprint-based similarity.
    
    Creates fixed-size fingerprints enabling fast similarity comparison via Hamming distance.
    """
    
    def __init__(self, fingerprint_size: int = 64):
        """
        Initialize SimHash with specified fingerprint size.
        
        Note: fingerprint_size parameter is kept for compatibility.
        The simhash-py library uses 64-bit fingerprints by default.
        
        Args:
            fingerprint_size: Size of fingerprint in bits (default: 64, not configurable in library)
        """
        self.fingerprint_size = fingerprint_size
        self._simhash = None
        self._fingerprint = 0  # Default value before compute() is called
    
    @property
    def fingerprint(self) -> int:
        """Get fingerprint as integer for compatibility with tests."""
        if self._simhash is None:
            return self._fingerprint
        return self._simhash.value
    
    @fingerprint.setter
    def fingerprint(self, value: int) -> None:
        """Set fingerprint directly for testing purposes."""
        self._fingerprint = value
        self._simhash = None  # Clear computed simhash since we're setting manually
    
    def compute(self, text: str, shingle_size: int = 4) -> None:
        """
        Compute SimHash fingerprint for text.

        Args:
            text: Input text
            shingle_size: Size of character shingles (default: 4)
        """
        # Normalize text
        text = text.lower().strip()

        if not text:
            self._simhash = SimHashLib([])
            self._fingerprint = 0
        else:
            # Build character shingles and pass as feature list.
            # SimHashLib accepts a list[str] and treats each element as a feature,
            # so passing shingles gives true shingle-based fingerprinting.
            shingles = [
                text[i : i + shingle_size]
                for i in range(len(text) - shingle_size + 1)
            ]
            # Fall back to word tokens if text is shorter than shingle_size
            features = shingles if shingles else text.split()
            self._simhash = SimHashLib(features)
            self._fingerprint = self._simhash.value
    
    def hamming_distance(self, other: SimHash) -> int:
        """
        Calculate Hamming distance to another SimHash.
        
        Counts the number of differing bits between fingerprints.
        
        Args:
            other: Other SimHash instance
        
        Returns:
            Hamming distance (0 to 64)
        """
        if self._simhash is None or other._simhash is None:
            # If either is not computed, return max distance
            fp1 = self.fingerprint
            fp2 = other.fingerprint
            xor = fp1 ^ fp2
            # Count set bits
            return bin(xor).count('1')
        
        return self._simhash.distance(other._simhash)
    
    def similarity(self, other: SimHash) -> float:
        """
        Calculate normalized similarity (0-1) to another SimHash.
        
        Args:
            other: Other SimHash instance
        
        Returns:
            Similarity score (1.0 = identical, 0.0 = maximally different)
        """
        distance = self.hamming_distance(other)
        return 1.0 - (distance / self.fingerprint_size)
    
    def get_fingerprint(self) -> int:
        """Get fingerprint as integer."""
        if self._simhash is None:
            return 0
        return self._simhash.value
    
    def get_fingerprint_binary(self) -> str:
        """Get fingerprint as binary string."""
        return format(self.get_fingerprint(), f'0{self.fingerprint_size}b')
    
    def __repr__(self) -> str:
        return f"SimHash(fingerprint={self.get_fingerprint_binary()})"


class SimHashIndex:
    """
    Index for efficient SimHash-based similarity search.
    
    Uses bit sampling to enable sub-linear similarity search by
    partitioning fingerprints into samples, with fallback to linear scan.
    """
    
    def __init__(self, fingerprint_size: int = 64, num_samples: int = 4):
        """
        Initialize SimHash index.
        
        Args:
            fingerprint_size: Size of SimHash fingerprints (default: 64)
            num_samples: Number of bit samples for indexing (default: 4)
        """
        self.fingerprint_size = fingerprint_size
        self.num_samples = num_samples
        self.sample_size = fingerprint_size // num_samples
        
        # Hash tables for each sample
        self.tables: list[dict[int, Set[str]]] = [
            {} for _ in range(num_samples)
        ]
        self.documents: dict[str, SimHash] = {}
    
    def add_document(self, doc_id: str, simhash: SimHash) -> None:
        """
        Add document's SimHash to the index.
        
        Args:
            doc_id: Unique document identifier
            simhash: SimHash fingerprint
        """
        self.documents[doc_id] = simhash
        fingerprint = simhash.get_fingerprint()
        
        # Index by samples
        for sample_idx in range(self.num_samples):
            start = sample_idx * self.sample_size
            end = start + self.sample_size
            
            # Extract bits for this sample
            sample_mask = ((1 << (end - start)) - 1) << start
            sample_val = (fingerprint & sample_mask) >> start
            
            if sample_val not in self.tables[sample_idx]:
                self.tables[sample_idx][sample_val] = set()
            
            self.tables[sample_idx][sample_val].add(doc_id)
    
    def query(
        self,
        simhash: SimHash,
        max_distance: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Find similar documents with SimHash-based search.
        
        Uses bit sampling to find candidates efficiently, with fallback to linear scan
        if no candidates are found (handles cases where samples don't match exactly).
        
        Args:
            simhash: Query SimHash
            max_distance: Maximum Hamming distance threshold
        
        Returns:
            List of (doc_id, similarity) tuples, sorted by similarity
        """
        candidates = set()
        fingerprint = simhash.get_fingerprint()
        
        # Collect candidates from each sample table
        for sample_idx in range(self.num_samples):
            start = sample_idx * self.sample_size
            end = start + self.sample_size
            
            # Extract sample bits
            sample_mask = ((1 << (end - start)) - 1) << start
            sample_val = (fingerprint & sample_mask) >> start
            
            # Get candidates with same sample
            for doc_id in self.tables[sample_idx].get(sample_val, set()):
                candidates.add(doc_id)
        
        # Fallback: If no candidates found via sampling, do linear scan
        # This handles cases where sample bits don't match exactly
        if not candidates:
            candidates = set(self.documents.keys())
        
        # Verify candidates with actual Hamming distance
        results = []
        for doc_id in candidates:
            distance = simhash.hamming_distance(self.documents[doc_id])
            if distance <= max_distance:
                similarity = simhash.similarity(self.documents[doc_id])
                results.append((doc_id, similarity))
        
        # Sort by similarity descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results


def get_simhash(text: str, fingerprint_size: int = 64, shingle_size: int = 4) -> SimHash:
    """
    Generate SimHash fingerprint for text.
    
    Args:
        text: Input text
        fingerprint_size: Size of fingerprint in bits (default: 64)
        shingle_size: Size of shingles (default: 4, handled by simhash-py)
    
    Returns:
        SimHash instance
    """
    simhash = SimHash(fingerprint_size=fingerprint_size)
    simhash.compute(text, shingle_size=shingle_size)
    return simhash


if __name__ == "__main__":
    # Example usage
    docs = {
        "doc1": "the quick brown fox jumps over the lazy dog",
        "doc2": "the quick brown fox jumps over a lazy dog",
        "doc3": "machine learning is fascinating and powerful",
    }
    
    # Create SimHash index
    index = SimHashIndex(fingerprint_size=64, num_samples=4)
    
    # Index documents
    for doc_id, text in docs.items():
        sh = get_simhash(text)
        index.add_document(doc_id, sh)
    
    # Query
    query_text = "the quick brown fox jumps"
    query_sh = get_simhash(query_text)
    results = index.query(query_sh, max_distance=5)
    
    print("Query results:")
    for doc_id, similarity in results:
        print(f"  {doc_id}: {similarity:.2%}")

