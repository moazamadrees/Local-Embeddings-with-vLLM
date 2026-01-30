import logging
from typing import List, Dict
from backend.preprocessing.vector_store import VectorStore
from backend.preprocessing.embedder import TextEmbedder
from backend.config import EMBEDDING_MODEL
from backend.config import CHROMA_DB_DIR, TOP_K_RETRIEVAL, APPLY_METADATA_FILTER

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)


class Retriever:
    def __init__(self, vector_store_path: str = None, top_k: int = None):
        self.vector_store_path = vector_store_path or str(CHROMA_DB_DIR)
        self.top_k = top_k or TOP_K_RETRIEVAL
        
        logger.info(f"Initializing Retriever with vector store at {self.vector_store_path}")
        
        try:
            self.vector_store = VectorStore(
                persist_directory=self.vector_store_path,
                collection_name="uet_documents"
            )
            # embedder used to compute query embeddings with same model as index
            self.embedder = TextEmbedder(model_name=EMBEDDING_MODEL)
            logger.info(f"Retriever ready with {self.vector_store.get_count()} documents")
        except Exception as e:
            logger.error(f"Error initializing Retriever: {str(e)}")
            raise

    def retrieve(self, query: str, top_k: int = None) -> List[Dict]:
        k = top_k or self.top_k
        
        try:
            expanded_query = self._expand_query(query)
            metadata_filter = self._get_metadata_filter(query) if APPLY_METADATA_FILTER else None

            # compute query embeddings using the same embedder used for indexing
            try:
                query_embedding = self.embedder.embed_text(expanded_query)
            except Exception:
                query_embedding = None
            
            logger.info(f"Retrieving top {k} documents for query: '{query}'")
            if expanded_query != query:
                logger.info(f"Expanded query: '{expanded_query}'")
            
            results = self.vector_store.query(query_text=expanded_query, n_results=k, where=metadata_filter, query_embeddings=query_embedding)
            
            documents = []
            for i, doc in enumerate(results['documents'][0]):
                documents.append({
                    'content': doc,
                    'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                    'id': results['ids'][0][i] if results['ids'] else f"doc_{i}",
                    'distance': results['distances'][0][i] if results['distances'] else 0.0
                })
            
            logger.info(f"Retrieved {len(documents)} documents")
            return documents
            
        except Exception as e:
            logger.error(f"Error during retrieval: {str(e)}")
            raise
    
    def _expand_query(self, query: str) -> str:
        """Expand query with related terms for better retrieval"""
        query_lower = query.lower()
        
        expansions = []
        if "admission" in query_lower or "requirement" in query_lower or "eligibility" in query_lower:
            expansions.append("eligibility criteria admission requirements")
        if "faculty" in query_lower or "professor" in query_lower or "staff" in query_lower:
            expansions.append("faculty members professors")
        if "program" in query_lower or "degree" in query_lower:
            expansions.append("offered programs degrees")
        
        if expansions:
            return f"{query} {' '.join(expansions)}"
        return query
    
    def _get_metadata_filter(self, query: str) -> Dict:
        """Create metadata filter based on query intent"""
        query_lower = query.lower()
        
        if any(term in query_lower for term in ["admission", "requirement", "eligibility", "criteria"]):
            return {"has_eligibility": True}
        elif any(term in query_lower for term in ["faculty", "professor", "staff", "dean", "chairman"]):
            return {"has_faculty": True}
        elif any(term in query_lower for term in ["program", "degree", "offered"]):
            return {"has_programs": True}
        
        return None

    def format_context(self, documents: List[Dict]) -> str:
        if not documents:
            logger.warning("No documents to format")
            return ""
        
        context_parts = []
        for i, doc_dict in enumerate(documents, 1):
            content = doc_dict.get('content', '')
            context_parts.append(f"[Context {i}]\n{content}\n")
        
        context = "\n".join(context_parts)
        logger.info(f"Formatted context with {len(documents)} documents ({len(context)} characters)")
        
        return context

    def retrieve_and_format(self, query: str, top_k: int = None) -> tuple[str, List[Dict]]:
        documents = self.retrieve(query, top_k)
        context = self.format_context(documents)
        return context, documents


if __name__ == "__main__":
    try:
        retriever = Retriever()
        
        test_query = "What are the admission requirements for Computer Science?"
        print(f"Test Query: {test_query}\n")
        
        context, data = retriever.retrieve_and_format(test_query, top_k=3)
        
        print(f"Retrieved {data['count']} documents:\n")
        print(context[:500])
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure to run the preprocessing pipeline first!")
