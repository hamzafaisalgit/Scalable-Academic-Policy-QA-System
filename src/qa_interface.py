"""
Provides a Streamlit web interface for the QA system with:
  - Retrieved document chunks display
  - Source references and metadata
  - Comparison of retrieval methods
"""

import streamlit as st
import os
import sys
from pathlib import Path

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from query_processor import QueryProcessor
from ingestion import ingest_corpus
from retrieval import HybridRetriever, SimilarityMethod
from tfidf import TFIDFRetriever


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if "processor" not in st.session_state:
        st.session_state.processor = None
    if "corpus_loaded" not in st.session_state:
        st.session_state.corpus_loaded = False
    if "query_history" not in st.session_state:
        st.session_state.query_history = []


def load_corpus():
    """Load and index the corpus."""
    with st.spinner("Loading corpus..."):
        corpus = ingest_corpus()
        
        # Initialize retrievers
        lsh_retriever = HybridRetriever(method=SimilarityMethod.MINHASH_LSH)
        tfidf_retriever = TFIDFRetriever()
        
        # Initialize QueryProcessor
        llm_api_key = os.getenv("GEMINI_API_KEY")
        processor = QueryProcessor(
            lsh_retriever=lsh_retriever,
            tfidf_retriever=tfidf_retriever,
            llm_api_key=llm_api_key,
        )
        
        # Index corpus
        processor.index_corpus(corpus)
        
        st.session_state.processor = processor
        st.session_state.corpus_loaded = True
        
        return processor


def display_answer_result(answer_result, show_sources=True):
    """Display answer result with supporting evidence."""
    # Display the answer
    st.markdown("### Answer")
    st.info(answer_result.answer)
    
    # Display metadata
    col1, col2 = st.columns(2)
    with col1:
        st.caption(f"**Method:** {answer_result.answer_generation_method}")
    with col2:
        st.caption(f"**Model:** {answer_result.model}")
    
    # Display retrieved chunks as supporting evidence
    if show_sources and answer_result.retrieved_chunks:
        st.markdown("### Supporting Evidence (Retrieved Chunks)")

        for i, chunk in enumerate(answer_result.retrieved_chunks, 1):
            pr_score = None
            if answer_result.pagerank_scores:
                pr_score = answer_result.pagerank_scores.get(chunk.chunk_id)

            pr_label = f" | PageRank: {pr_score:.3f}" if pr_score is not None else ""
            with st.expander(
                f"📄 Chunk {i} — {chunk.chunk_id} "
                f"(Relevance: {chunk.similarity_score:.2%}{pr_label})"
            ):
                st.markdown(f"**Text:**\n{chunk.text}")
                if pr_score is not None:
                    st.progress(pr_score, text=f"PageRank importance: {pr_score:.3f}")

                # Display metadata if available
                if hasattr(chunk, 'metadata') and chunk.metadata:
                    st.markdown("**Metadata:**")
                    for key, value in chunk.metadata.items():
                        st.caption(f"- {key}: {value}")


def format_chunk_info(chunk):
    """Format chunk information for display."""
    return {
        "Chunk ID": chunk.chunk_id,
        "Similarity Score": f"{chunk.similarity_score:.4f}",
        "Text Preview": chunk.text[:150] + "..." if len(chunk.text) > 150 else chunk.text,
    }


def main():
    """Main Streamlit application."""
    # Page configuration
    st.set_page_config(
        page_title="Academic Policy QA System",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    # Title and description
    st.title("📚 Academic Policy QA System")
    st.markdown(
        """
        This system answers questions about academic policies using:
        - **Retrievers**: LSH-based (fast) or TF-IDF (accurate)
        - **Answer Generation**: Gemini API (Google - fluent and contextual)
        
        All answers are based on retrieved document content with supporting evidence.
        """
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Load corpus button
        if not st.session_state.corpus_loaded:
            if st.button("📥 Load Corpus", use_container_width=True):
                load_corpus()
                st.success("✅ Corpus loaded successfully!")
        else:
            st.success("✅ Corpus loaded")
            if st.button("🔄 Reload Corpus", use_container_width=True):
                load_corpus()
                st.success("✅ Corpus reloaded!")
        
        # Retrieval method selection
        retrieval_method = st.selectbox(
            "Retrieval Method",
            options=["tfidf", "lsh", "both"],
            help="TF-IDF is more accurate, LSH is faster, 'both' compares both"
        )
        
        # Top-k selection
        top_k = st.slider(
            "Number of Retrieved Chunks",
            min_value=1,
            max_value=10,
            value=3,
            help="More chunks = more context but longer processing"
        )
        
        # Temperature for LLM
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=0.7,
            step=0.1,
            help="Higher = more creative, Lower = more deterministic"
        )
        
        # LLM availability
        llm_key = os.getenv("GEMINI_API_KEY")
        if not llm_key:
            st.warning(
                "⚠️ Gemini API key not found. "
                "Set GEMINI_API_KEY environment variable to use this system."
            )
        else:
            st.info("✅ Gemini API key configured")
        
        # Status information
        st.markdown("---")
        st.subheader("Status")
        if st.session_state.corpus_loaded:
            processor = st.session_state.processor
            status = processor.get_status()
            st.json({
                "Corpus Indexed": status["corpus_indexed"],
                "TF-IDF Available": status["tfidf_available"],
                "LSH Available": status["lsh_available"],
                "TF-IDF Chunks": status["tfidf_chunk_count"],
                "LSH Chunks": status["lsh_chunk_count"],
                "LLM Available": status["llm_available"],
                "PageRank Ready": status.get("pagerank_available", False),
            })
            if status.get("pagerank_stats"):
                ps = status["pagerank_stats"]
                st.caption(
                    f"PageRank: {ps['n_nodes']} nodes, {ps['n_edges']} edges, "
                    f"{ps['iterations']} iters, {ps['build_time_ms']:.0f} ms"
                )
    
    # Main content area
    if not st.session_state.corpus_loaded:
        st.warning("📥 Please load the corpus first using the sidebar button.")
        return
    
    # Check if Gemini is available
    llm_key = os.getenv("GEMINI_API_KEY")
    if not llm_key:
        st.error("❌ Gemini API key not configured. Please set GEMINI_API_KEY environment variable.")
        return
    
    # Query input
    st.markdown("---")
    query = st.text_input(
        "Ask a question about academic policies:",
        placeholder="e.g., What is the policy on academic probation?",
        label_visibility="collapsed"
    )
    
    # Query submission
    col1, col2, col3 = st.columns([2, 1, 1])
    with col2:
        submit_button = st.button("🔍 Search", use_container_width=True)
    with col3:
        show_comparison = st.checkbox("Compare Methods", value=False)
    
    if submit_button and query:
        processor = st.session_state.processor
        
        # Add to history
        if query not in st.session_state.query_history:
            st.session_state.query_history.append(query)
        
        try:
            # Generate answer
            with st.spinner("Processing query with Gemini API..."):
                answer_result = processor.answer_question(
                    query=query,
                    top_k=top_k,
                    retrieval_method=retrieval_method,
                    temperature=temperature,
                )
            
            # Display results
            display_answer_result(answer_result, show_sources=True)

            # Show PageRank importance panel
            if processor.pagerank_scorer is not None:
                st.markdown("---")
                with st.expander("📊 Top Handbook Sections by PageRank Importance", expanded=False):
                    st.caption(
                        "PageRank identifies the most *central* sections of the handbook — "
                        "chunks that are highly similar to many other important chunks."
                    )
                    top_sections = processor.pagerank_scorer.top_sections(n=10)
                    for rank, (chunk_id, pr_raw) in enumerate(top_sections, 1):
                        pr_norm = processor.pagerank_scorer.get_normalised_score(chunk_id)
                        try:
                            preview = processor.lsh_retriever.get_chunk_text(chunk_id)[:180]
                        except Exception:
                            preview = ""
                        col_rank, col_info = st.columns([1, 9])
                        with col_rank:
                            st.markdown(f"**#{rank}**")
                        with col_info:
                            st.markdown(f"`{chunk_id}` — score: **{pr_norm:.3f}**")
                            if preview:
                                st.caption(preview + ("…" if len(preview) == 180 else ""))
                        st.progress(pr_norm)

            # Show method comparison if requested
            if show_comparison and retrieval_method == "both":
                st.markdown("---")
                st.subheader("📊 Retrieval Method Comparison")
                
                with st.spinner("Comparing retrieval methods..."):
                    comparison = processor.compare_methods(query, top_k=top_k)
                
                # Create comparison columns
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**TF-IDF Results**")
                    if comparison["tfidf_results"]:
                        for i, result in enumerate(comparison["tfidf_results"], 1):
                            st.write(
                                f"{i}. {result.chunk_id} "
                                f"(Score: {result.similarity_score:.4f})"
                            )
                    else:
                        st.write("No results")
                
                with col2:
                    st.markdown("**LSH Results**")
                    if comparison["lsh_results"]:
                        for i, result in enumerate(comparison["lsh_results"], 1):
                            st.write(
                                f"{i}. {result.chunk_id} "
                                f"(Score: {result.similarity_score:.4f})"
                            )
                    else:
                        st.write("No results")
                
                # Show overlap statistics
                st.markdown("**Overlap Statistics**")
                overlap_data = {
                    "In Both": comparison["intersection_count"],
                    "TF-IDF Only": comparison["tfidf_only_count"],
                    "LSH Only": comparison["lsh_only_count"],
                }
                st.bar_chart(overlap_data)
        
        except RuntimeError as e:
            st.error(f"❌ Error: {str(e)}")
        except Exception as e:
            st.error(f"❌ Unexpected error: {str(e)}")
    
    # Query history
    if st.session_state.query_history:
        st.markdown("---")
        st.subheader("📋 Query History")
        for prev_query in st.session_state.query_history[-5:]:
            if st.button(prev_query, key=f"history_{prev_query}"):
                st.session_state.current_query = prev_query


if __name__ == "__main__":
    main()
