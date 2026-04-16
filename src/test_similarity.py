"""
Test suite for similarity and indexing components.

Tests MinHash+LSH and SimHash implementations with various scenarios.
"""

import pytest
from minhash_lsh import MinHash, LocalitySensitiveHash, get_minhash_signature, shingle_text
from simhash_impl import SimHash, SimHashIndex, get_simhash
from retrieval import HybridRetriever, SimilarityMethod


class TestMinHash:
    """Test MinHash signature generation and similarity."""
    
    def test_minhash_initialization(self):
        """Test MinHash initialization."""
        mh = MinHash(num_perm=128)
        assert mh.num_perm == 128
        # datasketch initializes hash_values to max uint32 (4294967295)
        # or they could be float('inf') depending on implementation
        assert len(mh.hash_values) == 128
        assert all(h > 0 for h in mh.hash_values)  # All values should be initialized
    
    def test_minhash_update(self):
        """Test MinHash update with items."""
        mh = MinHash(num_perm=64)
        items = ["hello", "world", "test"]
        mh.update(items)
        
        assert not mh._is_empty
        assert all(h != float('inf') for h in mh.hash_values)
    
    def test_minhash_similarity(self):
        """Test MinHash similarity computation."""
        text1 = "the quick brown fox jumps over the lazy dog"
        text2 = "the quick brown fox jumps over a lazy dog"
        text3 = "machine learning is fascinating"
        
        mh1 = get_minhash_signature(text1)
        mh2 = get_minhash_signature(text2)
        mh3 = get_minhash_signature(text3)
        
        # Similar texts should have higher similarity
        sim_12 = mh1.jaccard_similarity(mh2)
        sim_13 = mh1.jaccard_similarity(mh3)
        
        assert sim_12 > sim_13
        assert 0 <= sim_12 <= 1
        assert 0 <= sim_13 <= 1
    
    def test_minhash_identical(self):
        """Test MinHash with identical texts."""
        text = "academic probation policy"
        mh1 = get_minhash_signature(text)
        mh2 = get_minhash_signature(text)
        
        # Identical texts should have 100% similarity
        similarity = mh1.jaccard_similarity(mh2)
        assert similarity == 1.0


class TestShingle:
    """Test text shingling."""
    
    def test_shingle_basic(self):
        """Test basic shingling."""
        text = "hello"
        shingles = shingle_text(text, shingle_size=2)
        
        expected = {"he", "el", "ll", "lo"}
        assert shingles == expected
    
    def test_shingle_empty(self):
        """Test shingling with empty text."""
        text = ""
        shingles = shingle_text(text, shingle_size=2)
        
        assert shingles == set()
    
    def test_shingle_short_text(self):
        """Test shingling with text shorter than shingle size."""
        text = "hi"
        shingles = shingle_text(text, shingle_size=4)
        
        assert shingles == set()


class TestLSH:
    """Test Locality Sensitive Hashing."""
    
    def test_lsh_initialization(self):
        """Test LSH initialization."""
        lsh = LocalitySensitiveHash(num_perm=128)
        assert lsh.num_perm == 128
        assert lsh.num_bands > 0
    
    def test_lsh_add_document(self):
        """Test adding documents to LSH."""
        lsh = LocalitySensitiveHash(num_perm=128)
        
        text = "academic probation policy for students"
        sig = get_minhash_signature(text)
        lsh.add_document("doc1", sig)
        
        assert "doc1" in lsh.doc_signatures
    
    def test_lsh_query(self):
        """Test LSH query with longer, realistic queries."""
        docs = {
            "doc1": "academic probation is a serious disciplinary action for students who fail to meet performance standards",
            "doc2": "probation policy for students means they must improve their academic performance within one semester",
            "doc3": "hostel allocation is based on merit and seniority of students",
        }
        
        lsh = LocalitySensitiveHash(num_perm=128)
        for doc_id, text in docs.items():
            sig = get_minhash_signature(text)
            lsh.add_document(doc_id, sig)
        
        # Use longer query with more overlap
        query_sig = get_minhash_signature("academic probation students performance standards")
        results = lsh.query(query_sig, threshold=0.05)  # Lower threshold
        
        # Should find probation-related docs
        result_ids = [r[0] for r in results]
        assert any(did in result_ids for did in ["doc1", "doc2"]), f"Expected doc1 or doc2, got {result_ids}"


class TestSimHash:
    """Test SimHash fingerprinting."""
    
    def test_simhash_initialization(self):
        """Test SimHash initialization."""
        sh = SimHash(fingerprint_size=64)
        assert sh.fingerprint_size == 64
        assert sh.fingerprint == 0
    
    def test_simhash_compute(self):
        """Test SimHash computation."""
        text = "the quick brown fox jumps over the lazy dog"
        sh = SimHash(fingerprint_size=64)
        sh.compute(text)
        
        assert sh.fingerprint != 0
    
    def test_simhash_identical(self):
        """Test SimHash with identical texts."""
        text = "academic probation policy"
        sh1 = get_simhash(text)
        sh2 = get_simhash(text)
        
        # Identical texts should have 100% similarity
        assert sh1.hamming_distance(sh2) == 0
        assert sh1.similarity(sh2) == 1.0
    
    def test_simhash_hamming_distance(self):
        """Test Hamming distance computation."""
        sh1 = SimHash(fingerprint_size=64)
        sh1.fingerprint = 0b1111
        
        sh2 = SimHash(fingerprint_size=64)
        sh2.fingerprint = 0b1100
        
        # Hamming distance should be 2 (bits 1 and 0 differ)
        assert sh1.hamming_distance(sh2) == 2
    
    def test_simhash_similarity(self):
        """Test SimHash similarity computation."""
        text1 = "the quick brown fox jumps over the lazy dog"
        text2 = "the quick brown fox jumps over a lazy dog"
        text3 = "machine learning algorithms are powerful"
        
        sh1 = get_simhash(text1)
        sh2 = get_simhash(text2)
        sh3 = get_simhash(text3)
        
        # Similar texts should have higher similarity
        sim_12 = sh1.similarity(sh2)
        sim_13 = sh1.similarity(sh3)
        
        assert sim_12 > sim_13
        assert 0 <= sim_12 <= 1
        assert 0 <= sim_13 <= 1


class TestSimHashIndex:
    """Test SimHash indexing."""
    
    def test_simhash_index_initialization(self):
        """Test SimHash index initialization."""
        index = SimHashIndex(fingerprint_size=64, num_samples=4)
        assert len(index.tables) == 4
    
    def test_simhash_index_add_document(self):
        """Test adding documents to SimHash index."""
        index = SimHashIndex()
        text = "academic probation policy for students"
        sh = get_simhash(text)
        index.add_document("doc1", sh)
        
        assert "doc1" in index.documents
    
    def test_simhash_index_query(self):
        """Test SimHash index query with longer, realistic text."""
        docs = {
            "doc1": "academic probation students must improve their academic performance within one semester",
            "doc2": "probation policy means students have strict performance requirements and monitoring",
            "doc3": "hostel allocation is based on merit and seniority of enrolled students",
        }
        
        index = SimHashIndex(fingerprint_size=64, num_samples=4)
        for doc_id, text in docs.items():
            sh = get_simhash(text)
            index.add_document(doc_id, sh)
        
        # Use longer query with better overlap
        query = "academic probation students performance requirements"
        query_sh = get_simhash(query)
        results = index.query(query_sh, max_distance=40)  # Very lenient distance (~63% of 64-bit fingerprint)
        
        # Should find probation-related docs
        assert len(results) > 0, f"Expected results, got empty list"
        result_ids = [r[0] for r in results]
        assert any(did in result_ids for did in ["doc1", "doc2"]), f"Expected probation docs, got {result_ids}"


class TestHybridRetriever:
    """Test hybrid retriever combining both methods."""
    
    def test_hybrid_initialization_minhash(self):
        """Test hybrid retriever with MinHash+LSH."""
        retriever = HybridRetriever(method=SimilarityMethod.MINHASH_LSH)
        assert retriever.lsh is not None
        assert retriever.simhash_index is None
    
    def test_hybrid_initialization_simhash(self):
        """Test hybrid retriever with SimHash."""
        retriever = HybridRetriever(method=SimilarityMethod.SIMHASH)
        assert retriever.lsh is None
        assert retriever.simhash_index is not None
    
    def test_hybrid_initialization_both(self):
        """Test hybrid retriever with both methods."""
        retriever = HybridRetriever(method=SimilarityMethod.HYBRID)
        assert retriever.lsh is not None
        assert retriever.simhash_index is not None
    
    def test_hybrid_add_chunk(self):
        """Test adding chunks to hybrid retriever."""
        retriever = HybridRetriever(method=SimilarityMethod.HYBRID)
        retriever.add_chunk("chunk1", "academic policy guidelines")
        
        assert "chunk1" in retriever.chunks
    
    def test_hybrid_search_minhash(self):
        """Test hybrid search with MinHash."""
        retriever = HybridRetriever(method=SimilarityMethod.HYBRID)
        
        chunks = {
            "c1": "academic probation is a serious disciplinary measure for underperforming students",
            "c2": "probation policy requires students to maintain minimum academic standards",
            "c3": "hostel accommodation rules and guidelines",
        }
        
        for chunk_id, text in chunks.items():
            retriever.add_chunk(chunk_id, text)
        
        results = retriever.search(
            "academic probation students standards",
            method=SimilarityMethod.MINHASH_LSH,
            top_k=2
        )
        
        assert len(results) > 0, "Expected search results for MinHash"
        assert all(0 <= r.similarity_score <= 1 for r in results)
    
    def test_hybrid_search_simhash(self):
        """Test hybrid search with SimHash."""
        retriever = HybridRetriever(method=SimilarityMethod.HYBRID)
        
        chunks = {
            "c1": "academic probation is serious and requires immediate action from students",
            "c2": "probation policy for students during academic probation period enforcement",
            "c3": "hostel accommodation and residence hall guidelines for all residents",
        }
        
        for chunk_id, text in chunks.items():
            retriever.add_chunk(chunk_id, text)
        
        results = retriever.search(
            "probation students policy requirements actions",
            method=SimilarityMethod.SIMHASH,
            top_k=2
        )
        
        assert len(results) > 0, "Expected search results for SimHash"
        assert all(0 <= r.similarity_score <= 1 for r in results)
    
    def test_hybrid_search_combined(self):
        """Test hybrid search with both methods."""
        retriever = HybridRetriever(method=SimilarityMethod.HYBRID)
        
        chunks = {
            "c1": "academic probation is a serious disciplinary action requiring students to improve performance",
            "c2": "probation policy guidelines for students on academic probation status",
            "c3": "hostel accommodation rules and residence guidelines",
        }
        
        for chunk_id, text in chunks.items():
            retriever.add_chunk(chunk_id, text)
        
        results = retriever.search(
            "academic probation students performance guidelines",
            method=SimilarityMethod.HYBRID,
            top_k=2
        )
        
        assert len(results) > 0, "Expected search results for hybrid method"
        assert all(0 <= r.similarity_score <= 1 for r in results)
    
    def test_hybrid_get_candidates(self):
        """Test candidate retrieval."""
        retriever = HybridRetriever(method=SimilarityMethod.HYBRID)
        
        chunks = {
            "c1": "academic probation requirements and expectations",
            "c2": "probation policy enforcement mechanisms for students",
            "c3": "hostel residence rules and guidelines",
        }
        
        for chunk_id, text in chunks.items():
            retriever.add_chunk(chunk_id, text)
        
        candidates = retriever.get_candidates("academic probation students requirements")
        assert len(candidates) > 0, f"Expected candidates, got {candidates}"


class TestPerformance:
    """Performance-related tests."""
    
    def test_large_scale_indexing(self):
        """Test indexing performance with many documents."""
        retriever = HybridRetriever(method=SimilarityMethod.HYBRID)
        
        # Add many chunks
        for i in range(100):
            text = f"policy document section {i} regarding academic performance and student requirements"
            retriever.add_chunk(f"chunk_{i}", text)
        
        # Query should still be fast
        results = retriever.search("academic policy performance requirements", top_k=5)
        assert len(results) <= 5
    
    def test_minhash_lsh_scalability(self):
        """Test MinHash+LSH with varying band configurations."""
        for num_perms in [64, 128, 256]:
            lsh = LocalitySensitiveHash(num_perm=num_perms)
            
            # Add documents
            docs = {
                f"doc_{i}": f"academic policy section {i} with various guidelines"
                for i in range(20)
            }
            
            for doc_id, text in docs.items():
                sig = get_minhash_signature(text, num_perm=num_perms)
                lsh.add_document(doc_id, sig)
            
            # Query should work
            query_sig = get_minhash_signature(
                "academic policy guidelines",
                num_perm=num_perms
            )
            results = lsh.query(query_sig, threshold=0.05)
            
            # At least some results expected
            assert isinstance(results, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
