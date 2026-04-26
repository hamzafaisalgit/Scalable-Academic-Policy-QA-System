
from __future__ import annotations

from typing import List, Optional, Set, Tuple, Dict
import re

from datasketch import MinHash as DataSketchMinHash
from datasketch import MinHashLSH


# ============================================================
# MINHASH WRAPPER
# ============================================================

class MinHash:
    def __init__(self, num_perm: int = 128, seed: int = 0):
        self.num_perm = num_perm
        self.seed = seed
        self._minhash = DataSketchMinHash(num_perm=num_perm, seed=seed)

    @property
    def hash_values(self) -> List[int]:
        return list(self._minhash.hashvalues)

    def update(self, items: Set[str]) -> None:
        for item in items:
            if item:  # safety
                self._minhash.update(item.encode("utf-8"))

    def jaccard_similarity(self, other: "MinHash") -> float:
        return self._minhash.jaccard(other._minhash)


# TEXT NORMALIZATION

def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# SHINGLING 

def shingle_text(text: str, shingle_size: int = 1) -> Set[str]:
    """
    FIX: default is WORD UNIGRAMS (critical for QA retrieval)
    """
    text = normalize_text(text)
    words = text.split()

    shingles = set()

    if len(words) == 0:
        return shingles

    if len(words) < shingle_size:
        return {" ".join(words)}

    for i in range(len(words) - shingle_size + 1):
        shingles.add(" ".join(words[i:i + shingle_size]))

    return shingles


def get_minhash_signature(text: str, num_perm: int = 128, shingle_size: int = 1) -> MinHash:
    m = MinHash(num_perm=num_perm)
    shingles = shingle_text(text, shingle_size)
    m.update(shingles)
    return m


# LSH 

class LocalitySensitiveHash:
    """
    FIXED LSH:
    - meaningful banding
    - no over-reliance on threshold
    - safe fallback behavior
    """

    def __init__(self, num_perm: int = 128, num_bands: Optional[int] = None):

        self.num_perm = num_perm

        self.num_bands = num_bands or 16
        self.rows_per_band = num_perm // self.num_bands

        self.lsh = MinHashLSH(
            num_perm=num_perm,
            params=(self.num_bands, self.rows_per_band)
        )

        self.doc_signatures: Dict[str, MinHash] = {}

    def add_document(self, doc_id: str, minhash: MinHash) -> None:
        self.doc_signatures[doc_id] = minhash
        self.lsh.insert(doc_id, minhash._minhash)

    def get_candidates(self, minhash: MinHash) -> Set[str]:
        return set(self.lsh.query(minhash._minhash))

    def query(
        self,
        minhash: MinHash,
        threshold: float = 0.2,
        top_k: int = 10
    ) -> List[Tuple[str, float]]:

        candidates = self.get_candidates(minhash)

        results = []

        if not candidates:
            # brute-force fallback
            for doc_id, mh in self.doc_signatures.items():
                sim = minhash.jaccard_similarity(mh)
                if sim >= threshold:
                    results.append((doc_id, sim))
        else:
            for doc_id in candidates:
                sim = minhash.jaccard_similarity(self.doc_signatures[doc_id])
                if sim >= threshold:
                    results.append((doc_id, sim))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]