import pytest
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from backend.rag.retriever import Retriever
from backend.config import CHROMA_DB_DIR


def test_retriever_initialization():
    try:
        retriever = Retriever()
        doc_count = retriever.vector_store.get_count()
        
        assert doc_count > 0, "Vector store should contain documents"
        print(f"✓ Retriever initialized with {doc_count} documents")
    except Exception as e:
        print(f"✗ Make sure to run preprocessing pipeline first!")
        raise


def test_retrieval():
    retriever = Retriever()
    
    query = "What are the admission requirements?"
    results = retriever.retrieve(query, top_k=5)
    
    assert results['count'] > 0, "Should retrieve at least one document"
    assert len(results['documents']) <= 5, "Should not exceed top_k"
    assert all(isinstance(doc, str) for doc in results['documents'])
    
    print(f"✓ Retrieved {results['count']} documents for query")


def test_context_formatting():
    retriever = Retriever()
    
    query = "Tell me about Computer Science programs"
    context, data = retriever.retrieve_and_format(query, top_k=3)
    
    assert context is not None
    assert len(context) > 0
    assert "[Context" in context
    
    print(f"✓ Context formatted successfully ({len(context)} characters)")


def test_relevance_scoring():
    retriever = Retriever()
    
    query = "Computer Science department programs"
    results = retriever.retrieve(query, top_k=3)
    
    distances = results['distances']
    assert len(distances) > 0
    assert all(isinstance(d, (int, float)) for d in distances)
    
    if len(distances) > 1:
        assert distances[0] <= distances[-1], "Results should be sorted by relevance"
    
    print(f"✓ Relevance scoring working correctly")


if __name__ == "__main__":
    print("Running RAG tests...\n")
    print("Note: These tests require the preprocessing pipeline to be run first.\n")
    
    try:
        test_retriever_initialization()
        test_retrieval()
        test_context_formatting()
        test_relevance_scoring()
        
        print("\n" + "="*50)
        print("All RAG tests passed! ✓")
        print("="*50)
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        print("\nMake sure to run: python -m backend.preprocessing.run_pipeline")
        raise
