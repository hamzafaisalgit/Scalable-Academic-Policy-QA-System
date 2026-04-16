"""
Smoke test for hybrid LSH-based similarity and indexing system.

This test verifies basic functionality of MinHash+LSH and SimHash implementations.
"""

import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent))

from minhash_lsh import MinHash, LocalitySensitiveHash, get_minhash_signature
from simhash import Simhash, SimhashIndex

def get_simhash(text, fingerprint_size=64):
    tokens = text.lower().split()
    return Simhash(tokens, f=fingerprint_size)


def simhash_similarity(sh1, sh2, bits=64):
    """Convert Hamming distance to similarity score (0–1)."""
    return 1 - (sh1.distance(sh2) / bits)



def test_minhash_basic():
    print("Testing MinHash basic functionality...")
    
    text1 = "the quick brown fox jumps over the lazy dog"
    text2 = "the quick brown fox jumps over a lazy dog"
    text3 = "machine learning is fascinating and powerful"
    
    sig1 = get_minhash_signature(text1, num_perm=128)
    sig2 = get_minhash_signature(text2, num_perm=128)
    sig3 = get_minhash_signature(text3, num_perm=128)
    
    sim_1_2 = sig1.jaccard_similarity(sig2)
    sim_1_3 = sig1.jaccard_similarity(sig3)
    
    assert sim_1_2 > sim_1_3
    assert 0 <= sim_1_2 <= 1
    assert 0 <= sim_1_3 <= 1
    
    print(f"  ✓ MinHash similarity comparison: {sim_1_2:.2%} vs {sim_1_3:.2%}")


def test_lsh_indexing():
    print("Testing LSH indexing and querying...")
    
    docs = {
        "doc1": "the quick brown fox jumps over the lazy dog",
        "doc2": "the quick brown fox jumps over a lazy dog",
        "doc3": "the fast brown fox leaps over the sleepy dog",
        "doc4": "machine learning is fascinating and powerful",
    }
    
    lsh = LocalitySensitiveHash(num_perm=128)
    for doc_id, text in docs.items():
        sig = get_minhash_signature(text)
        lsh.add_document(doc_id, sig)
    
    query_text = "the quick brown fox jumps over lazy"
    query_sig = get_minhash_signature(query_text)
    results = lsh.query(query_sig, threshold=0.01)
    
    assert len(results) > 0
    assert results[0][0] in docs
    
    similarities = [sim for _, sim in results]
    assert similarities == sorted(similarities, reverse=True)
    
    print(f"  ✓ LSH found {len(results)} similar documents")
    print(f"    Top result: {results[0][0]} ({results[0][1]:.2%})")




def test_simhash_basic():
    print("Testing SimHash basic functionality...")
    
    text1 = "the quick brown fox jumps over the lazy dog"
    text2 = "the quick brown fox jumps over a lazy dog"
    text3 = "machine learning is fascinating and powerful"
    
    sh1 = get_simhash(text1)
    sh2 = get_simhash(text2)
    sh3 = get_simhash(text3)
    
    sim_1_2 = simhash_similarity(sh1, sh2)
    sim_1_3 = simhash_similarity(sh1, sh3)
    
    dist_1_2 = sh1.distance(sh2)
    dist_1_3 = sh1.distance(sh3)
    
    assert sim_1_2 > sim_1_3
    assert dist_1_2 < dist_1_3
    
    print(f"  ✓ SimHash similarity: {sim_1_2:.2%} vs {sim_1_3:.2%}")
    print(f"    Hamming distance: {dist_1_2} vs {dist_1_3}")



def test_simhash_indexing():
    print("Testing SimHash index and querying...")
    
    docs = {
        "doc1": "the quick brown fox jumps over the lazy dog",
        "doc2": "the quick brown fox jumps over a lazy dog",
        "doc3": "the fast brown fox leaps over the sleepy dog",
        "doc4": "machine learning is fascinating and powerful",
    }
    
    # Proper format: list of (id, Simhash)
    objs = [(doc_id, get_simhash(text)) for doc_id, text in docs.items()]
    
    index = SimhashIndex(objs, k=20)  # k = max Hamming distance
    
    query_text = "the quick brown fox jumps over lazy"
    query_sh = get_simhash(query_text)
    
    result_ids = index.get_near_dups(query_sh)
    
    assert len(result_ids) > 0
    
    # Convert to (id, similarity)
    results = [(doc_id, simhash_similarity(query_sh, get_simhash(docs[doc_id]))) for doc_id in result_ids]
    results.sort(key=lambda x: x[1], reverse=True)
    
    print(f"  ✓ SimHash index found {len(results)} similar documents")
    print(f"    Top result: {results[0][0]} ({results[0][1]:.2%})")


def test_comparison():
    print("Testing MinHash vs SimHash comparison...")
    
    docs = {
        "doc1": "the quick brown fox jumps over the lazy dog",
        "doc2": "the quick brown fox jumps over a lazy dog",
        "doc3": "machine learning is fascinating and powerful",
    }
    
    # MinHash
    lsh = LocalitySensitiveHash(num_perm=128)
    for doc_id, text in docs.items():
        lsh.add_document(doc_id, get_minhash_signature(text))
    
    # SimHash
    objs = [(doc_id, get_simhash(text)) for doc_id, text in docs.items()]
    simhash_index = SimhashIndex(objs, k=25)
    
    query_text = "quick brown fox jumps over lazy"
    
    minhash_results = lsh.query(get_minhash_signature(query_text), threshold=0.05)
    simhash_ids = simhash_index.get_near_dups(get_simhash(query_text))
    
    simhash_results = [
        (doc_id, simhash_similarity(get_simhash(query_text), get_simhash(docs[doc_id])))
        for doc_id in simhash_ids
    ]
    simhash_results.sort(key=lambda x: x[1], reverse=True)
    
    print(f"  ✓ MinHash found {len(minhash_results)} results")
    print(f"    Top: {minhash_results[0][0]} ({minhash_results[0][1]:.2%})" if minhash_results else "    No results")
    
    print(f"  ✓ SimHash found {len(simhash_results)} results")
    print(f"    Top: {simhash_results[0][0]} ({simhash_results[0][1]:.2%})" if simhash_results else "    No results")



def main():
    print("\n" + "=" * 60)
    print("HYBRID LSH-BASED SIMILARITY & INDEXING - SMOKE TESTS")
    print("=" * 60 + "\n")
    
    try:
        test_minhash_basic()
        print()
        
        test_lsh_indexing()
        print()
        
        test_simhash_basic()
        print()
        
        test_simhash_indexing()
        print()
        
        test_comparison()
        print()
        
        print("=" * 60)
        print("✓ ALL SMOKE TESTS PASSED")
        print("=" * 60 + "\n")
        return 0
        
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}\n")
        return 1
    except Exception as e:
        print(f"\n✗ ERROR: {e}\n")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())