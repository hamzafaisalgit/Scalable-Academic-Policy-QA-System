from __future__ import annotations

import os
from pathlib import Path

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from retrieval import HybridRetriever, SimilarityMethod, SimilarityResult
from tfidf import TFIDFRetriever
from pagerank import PageRankScorer, build_pagerank_from_tfidf


@dataclass
class QueryResult:
    """
    Result of query processing with both retrieval methods.
    
    Attributes:
        query: The input user question
        lsh_results: Top-k results from LSH-based method
        tfidf_results: Top-k results from TF-IDF baseline method
        method: Which retrieval method(s) were used
    """
    query: str
    lsh_results: Optional[List[SimilarityResult]] = None
    tfidf_results: Optional[List[SimilarityResult]] = None
    method: str = "unknown"


@dataclass
class AnswerResult:
    """
    Result of LLM-based answer generation.
    
    Attributes:
        query: The original user question
        answer: The generated answer from LLM
        retrieved_chunks: List of SimilarityResult used as context
        answer_generation_method: Method used (e.g., "openai", "open-source")
        model: Model name used for generation
    """
    query: str
    answer: str
    retrieved_chunks: List[SimilarityResult]
    answer_generation_method: str = "llm"
    model: str = "gpt-3.5-turbo"
    pagerank_scores: Optional[Dict[str, float]] = None


class QueryProcessor:
    """
    Main interface for processing user queries using dual retrieval methods.
    
    This processor coordinates two independent retrieval systems:
    - LSH-based (approximate): Fast, suitable for large-scale deployments
    - TF-IDF baseline (exact): Accurate, useful for ground-truth evaluation
    
    The same corpus is indexed in both systems for fair comparison.
    """

    def __init__(
        self,
        lsh_retriever: Optional[HybridRetriever] = None,
        tfidf_retriever: Optional[TFIDFRetriever] = None,
        llm_api_key: Optional[str] = None,
        llm_model: str = "gemini-2.5-flash",
        pagerank_alpha: float = 0.30,
    ):
        """
        Initialize QueryProcessor with retrieval backends and LLM configuration.

        Args:
            lsh_retriever: HybridRetriever instance (MinHash+LSH method).
            tfidf_retriever: TFIDFRetriever instance (TF-IDF baseline).
            llm_api_key: API key for Gemini API.
            llm_model: LLM model name (default: "gemini-2.5-flash")
            pagerank_alpha: Weight given to PageRank in reranking (0=pure sim, 1=pure PR)
        """
        self.lsh_retriever = lsh_retriever
        self.tfidf_retriever = tfidf_retriever
        self._corpus_indexed = False
        self.pagerank_scorer: Optional[PageRankScorer] = None
        self.pagerank_alpha = pagerank_alpha

        # LLM Configuration
        self.llm_api_key = llm_api_key or os.getenv("GEMINI_API_KEY")
        self.llm_model = llm_model
        self._llm_available = self.llm_api_key is not None

    def index_corpus(self, corpus: Dict[str, Any]) -> None:
        """
        Index a corpus in both retrieval backends.
        
        The corpus should be produced by ingestion.ingest_corpus() and contain:
          - "texts": List[str] — chunk texts
          - "metadata": List[dict] — chunk metadata
          - "chunks": List[dict] — full chunk records
        
        Args:
            corpus: Corpus dictionary with chunks and metadata
            
        Raises:
            ValueError: If neither retriever is available
        """
        if not self.lsh_retriever and not self.tfidf_retriever:
            raise ValueError("At least one retriever must be initialized")

        # Index in LSH-based method
        if self.lsh_retriever:
            texts = corpus.get("texts", [])
            metadata_list = corpus.get("metadata", [])
            chunks_list = corpus.get("chunks", [])
            
            for i, text in enumerate(texts):
                # Determine chunk_id
                if i < len(chunks_list):
                    chunk_id = chunks_list[i].get("chunk_id", f"chunk-{i:04d}")
                elif i < len(metadata_list):
                    chunk_id = metadata_list[i].get("chunk_id", f"chunk-{i:04d}")
                else:
                    chunk_id = f"chunk-{i:04d}"
                
                self.lsh_retriever.add_chunk(chunk_id, text)

        # Index in TF-IDF baseline method
        if self.tfidf_retriever:
            self.tfidf_retriever.fit_corpus(corpus)

        self._corpus_indexed = True

        # Build PageRank scorer from TF-IDF vectors (primary integration path)
        if self.tfidf_retriever:
            try:
                self.pagerank_scorer = build_pagerank_from_tfidf(
                    self.tfidf_retriever, threshold=0.15
                )
            except Exception:
                self.pagerank_scorer = None

    def retrieve_lsh(
        self,
        query: str,
        top_k: int = 5,
        minhash_threshold: float = 0.05,
    ) -> List[SimilarityResult]:
        """
        Retrieve top-k chunks using LSH-based (approximate) method.
        
        This uses MinHash + Locality Sensitive Hashing for fast approximate
        retrieval. Suitable for large-scale deployments where speed matters
        more than exact accuracy.
        
        Args:
            query: User question / query string
            top_k: Number of top results to return
            minhash_threshold: Minimum similarity threshold for MinHash
                              (default 0.05 for short queries, more lenient)
        
        Returns:
            List of SimilarityResult sorted by similarity score (highest first)
            
        Raises:
            RuntimeError: If LSH retriever not initialized or corpus not indexed
        """
        if not self.lsh_retriever:
            raise RuntimeError("LSH retriever not initialized")
        if not self._corpus_indexed:
            raise RuntimeError("Corpus not indexed. Call index_corpus() first.")

        results = self.lsh_retriever.search(
            query=query,
            method=SimilarityMethod.MINHASH_LSH,
            minhash_threshold=minhash_threshold,
            top_k=top_k,
        )
        return results

    def retrieve_tfidf(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> List[SimilarityResult]:
        """
        Retrieve top-k chunks using TF-IDF baseline (exact) method.
        
        This uses standard TF-IDF vector space model with cosine similarity.
        Guarantees to return the globally optimal top-k results by scoring
        every chunk. Slower than LSH but suitable for evaluation and as
        ground-truth baseline.
        
        Args:
            query: User question / query string
            top_k: Number of top results to return
            min_score: Minimum cosine similarity threshold (default 0.0)
        
        Returns:
            List of SimilarityResult sorted by similarity score (highest first)
            
        Raises:
            RuntimeError: If TF-IDF retriever not initialized or corpus not indexed
        """
        if not self.tfidf_retriever:
            raise RuntimeError("TF-IDF retriever not initialized")
        if not self._corpus_indexed:
            raise RuntimeError("Corpus not indexed. Call index_corpus() first.")

        results = self.tfidf_retriever.search(
            query=query,
            top_k=top_k,
            min_score=min_score,
        )
        return results

    def retrieve_all(
        self,
        query: str,
        top_k: int = 5,
        **kwargs,
    ) -> QueryResult:
        """
        Retrieve results from both LSH and TF-IDF methods simultaneously.
        
        Useful for comparing performance and effectiveness of both methods
        on the same query.
        
        Args:
            query: User question / query string
            top_k: Number of top results per method
            **kwargs: Additional arguments passed to individual retrievers
                     (e.g., minhash_threshold, min_score)
        
        Returns:
            QueryResult containing results from both methods
            
        Raises:
            RuntimeError: If corpus not indexed
        """
        if not self._corpus_indexed:
            raise RuntimeError("Corpus not indexed. Call index_corpus() first.")

        lsh_results = None
        tfidf_results = None
        methods = []

        if self.lsh_retriever:
            lsh_results = self.retrieve_lsh(query, top_k=top_k, **kwargs)
            methods.append("lsh")

        if self.tfidf_retriever:
            tfidf_results = self.retrieve_tfidf(query, top_k=top_k, **kwargs)
            methods.append("tfidf")

        return QueryResult(
            query=query,
            lsh_results=lsh_results,
            tfidf_results=tfidf_results,
            method="+".join(methods) if methods else "none",
        )

    def compare_methods(
        self,
        query: str,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Compare LSH and TF-IDF retrieval on a single query.
        
        Provides detailed comparison including:
        - Top chunks from each method
        - Ranking differences
        - Score differences
        - Intersection of results
        
        Args:
            query: User question / query string
            top_k: Number of results to compare
        
        Returns:
            Dictionary with comparison details
        """
        if not self._corpus_indexed:
            raise RuntimeError("Corpus not indexed. Call index_corpus() first.")

        results = self.retrieve_all(query, top_k=top_k)
        
        lsh_chunks = set(r.chunk_id for r in (results.lsh_results or []))
        tfidf_chunks = set(r.chunk_id for r in (results.tfidf_results or []))
        
        intersection = lsh_chunks & tfidf_chunks
        lsh_only = lsh_chunks - tfidf_chunks
        tfidf_only = tfidf_chunks - lsh_chunks
        
        return {
            "query": query,
            "lsh_results": results.lsh_results,
            "tfidf_results": results.tfidf_results,
            "lsh_chunks": sorted(lsh_chunks),
            "tfidf_chunks": sorted(tfidf_chunks),
            "intersection": sorted(intersection),
            "lsh_only": sorted(lsh_only),
            "tfidf_only": sorted(tfidf_only),
            "intersection_count": len(intersection),
            "lsh_only_count": len(lsh_only),
            "tfidf_only_count": len(tfidf_only),
        }

    def _construct_prompt(
        self,
        query: str,
        retrieved_chunks: List[SimilarityResult],
    ) -> str:
        """
        Construct a prompt for LLM-based answer generation.
        
        The prompt includes the retrieved context and instructs the model
        to answer based ONLY on the provided information.
        
        Args:
            query: User question
            retrieved_chunks: List of retrieved SimilarityResult objects
        
        Returns:
            Formatted prompt string for LLM
        """
        # Build context from retrieved chunks
        context_parts = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            context_parts.append(f"[Document {i}] {chunk.text}")
        
        context = "\n\n".join(context_parts)
        
        prompt = f"""You are an academic policy expert assistant. Using the retrieved policy document excerpts below, provide a comprehensive and detailed answer to the student's question.

Your answer should:
- Be thorough and well-structured (minimum 4-6 sentences)
- Cover all relevant requirements, conditions, procedures, deadlines, and exceptions found in the documents
- Use clear language a student can act on
- Be based ONLY on information present in the retrieved documents
- If multiple aspects of the question are addressed in different documents, cover each one
- If the documents do not contain enough information to answer fully, say so explicitly

RETRIEVED DOCUMENTS:
{context}

STUDENT QUESTION: {query}

DETAILED ANSWER:"""
        
        return prompt

    def generate_answer_gemini(
        self,
        query: str,
        retrieved_chunks: List[SimilarityResult],
        temperature: float = 0.7,
    ) -> AnswerResult:
        """
        Generate an answer using Google's Gemini API.
        
        Constraint: Answers are based on retrieved chunks provided as context.
        The LLM is instructed to only use information from the retrieved documents.
        
        Args:
            query: User question
            retrieved_chunks: List of SimilarityResult objects for context
            temperature: LLM temperature parameter (0.0-2.0), default 0.7
        
        Returns:
            AnswerResult containing the generated answer and metadata
            
        Raises:
            RuntimeError: If Gemini API key is not configured
            ImportError: If google-genai package is not installed
        """
        if not self._llm_available:
            raise RuntimeError(
                "Gemini API key not configured. "
                "Set GEMINI_API_KEY environment variable or pass llm_api_key to __init__"
            )
        
        try:
            from google.genai import Client
            from google.genai.types import GenerateContentConfig
        except ImportError:
            raise ImportError(
                "google-genai package not installed. "
                "Install with: pip install google-genai"
            )
        
        # Initialize client
        client = Client(api_key=self.llm_api_key)
        
        # Construct prompt
        prompt = self._construct_prompt(query, retrieved_chunks)
        
        # Call Gemini API
        response = client.models.generate_content(
            model=self.llm_model,
            contents=prompt,
            config=GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=4096,
            )
        )
        answer = response.text.strip() if response.text else "Unable to generate answer"
        
        return AnswerResult(
            query=query,
            answer=answer,
            retrieved_chunks=retrieved_chunks,
            answer_generation_method="gemini",
            model=self.llm_model,
        )

    def answer_question(
        self,
        query: str,
        top_k: int = 3,
        retrieval_method: str = "tfidf",
        temperature: float = 0.7,
    ) -> AnswerResult:
        """
        Retrieve relevant chunks and generate an LLM-based answer.
        
        This is the main entry point for end-to-end QA pipeline:
        1. Retrieve top-k relevant chunks using specified method
        2. Generate answer using Gemini LLM
        
        Constraint: Answers are constrained to use only retrieved content.
        
        Args:
            query: User question
            top_k: Number of chunks to retrieve for context (default 3)
            retrieval_method: Which retrieval method to use
                            ("tfidf", "lsh", or "both" - uses tfidf results)
            temperature: LLM temperature parameter (default 0.7)
        
        Returns:
            AnswerResult with generated answer and retrieved context
            
        Raises:
            ValueError: If retrieval method is invalid
            RuntimeError: If corpus not indexed or LLM not configured
        """
        if not self._corpus_indexed:
            raise RuntimeError("Corpus not indexed. Call index_corpus() first.")
        
        if not self._llm_available:
            raise RuntimeError("LLM not configured. Provide Gemini API key.")
        
        # Retrieve chunks based on specified method
        if retrieval_method == "tfidf":
            retrieved_chunks = self.retrieve_tfidf(query, top_k=top_k)
        elif retrieval_method == "lsh":
            retrieved_chunks = self.retrieve_lsh(query, top_k=top_k)
        elif retrieval_method == "both":
            retrieved_chunks = self.retrieve_tfidf(query, top_k=top_k)
        else:
            raise ValueError(
                f"Invalid retrieval_method '{retrieval_method}'. "
                "Must be 'tfidf', 'lsh', or 'both'"
            )

        # Re-rank with PageRank if scorer is available
        if self.pagerank_scorer is not None:
            retrieved_chunks = self.pagerank_scorer.rerank(
                retrieved_chunks, alpha=self.pagerank_alpha
            )

        # Collect per-chunk PageRank scores for the UI
        pagerank_scores: Optional[Dict[str, float]] = None
        if self.pagerank_scorer is not None:
            pagerank_scores = {
                chunk.chunk_id: self.pagerank_scorer.get_normalised_score(chunk.chunk_id)
                for chunk in retrieved_chunks
            }

        # Generate answer using Gemini LLM
        answer_result = self.generate_answer_gemini(
            query=query,
            retrieved_chunks=retrieved_chunks,
            temperature=temperature,
        )
        answer_result.pagerank_scores = pagerank_scores

        return answer_result

    def get_status(self) -> Dict[str, Any]:
        """
        Get status information about the QueryProcessor.
        
        Returns:
            Dictionary with retriever status, corpus indexing state, and LLM availability
        """
        lsh_indexed = False
        tfidf_indexed = False
        lsh_chunks = 0
        tfidf_chunks = 0
        
        if self.lsh_retriever:
            lsh_chunks = len(self.lsh_retriever.get_chunk_ids())
            lsh_indexed = lsh_chunks > 0
        
        if self.tfidf_retriever:
            tfidf_chunks = len(self.tfidf_retriever._chunks)
            tfidf_indexed = tfidf_chunks > 0
        
        pr_stats = self.pagerank_scorer.stats() if self.pagerank_scorer else None

        return {
            "corpus_indexed": self._corpus_indexed,
            "lsh_available": self.lsh_retriever is not None,
            "tfidf_available": self.tfidf_retriever is not None,
            "lsh_indexed": lsh_indexed,
            "tfidf_indexed": tfidf_indexed,
            "lsh_chunk_count": lsh_chunks,
            "tfidf_chunk_count": tfidf_chunks,
            "llm_available": self._llm_available,
            "llm_model": self.llm_model if self._llm_available else None,
            "pagerank_available": self.pagerank_scorer is not None,
            "pagerank_stats": pr_stats,
        }


if __name__ == "__main__":
    print("=== Query Processor Demo with LLM-Based Answer Generation ===")
    
    from ingestion import ingest_corpus
    
    # Load corpus
    print("Loading corpus...")
    corpus = ingest_corpus()
    
    # Initialize retrievers
    print("Initializing retrievers...")
    lsh_retriever = HybridRetriever(method=SimilarityMethod.MINHASH_LSH)
    tfidf_retriever = TFIDFRetriever()
    
    # Initialize QueryProcessor with LLM support
    # Note: Requires OPENAI_API_KEY environment variable or pass llm_api_key parameter
    processor = QueryProcessor(
        lsh_retriever=lsh_retriever,
        tfidf_retriever=tfidf_retriever,
        # llm_api_key="your-openai-api-key",  # Or set OPENAI_API_KEY env var
    )
    
    # Index corpus
    print("Indexing corpus...")
    processor.index_corpus(corpus)
    
    # Print status
    status = processor.get_status()
    print(f"Status: {status}\n")
    
    # Test queries
    test_queries = [
        "What is academic probation?",
        "How do I apply for hostel accommodation?",
        "What are the fee payment deadlines?",
    ]
    
    for query in test_queries:
        print(f"\n{'='*80}")
        print(f"Query: {query}")
        print('='*80)
        
        # Retrieve using TF-IDF method
        print("\nRetrieved Chunks (TF-IDF):")
        tfidf_results = processor.retrieve_tfidf(query, top_k=3)
        for i, result in enumerate(tfidf_results, 1):
            print(f"  {i}. [{result.chunk_id}] Score: {result.similarity_score:.4f}")
            print(f"     {result.text[:80]}...")
        
        # Generate LLM-based answer
        # Note: This requires Gemini API key to be configured
        print("\nLLM-Based Answer Generation (Gemini):")
        try:
            llm_result = processor.answer_question(
                query,
                top_k=3,
                retrieval_method="tfidf",
                temperature=0.7
            )
            print(f"  {llm_result.answer}")
            print(f"  Model: {llm_result.model}")
            print(f"  Supporting chunks: {len(llm_result.retrieved_chunks)}")
        except RuntimeError as e:
            print(f"  Note: {e}")
            print("  (To enable LLM answers, set GEMINI_API_KEY environment variable)")
        except Exception as e:
            print(f"  Error: {e}")

