import logging
from typing import List, Dict, Optional
import chromadb
from chromadb.config import Settings
from pathlib import Path
import warnings
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore", category=DeprecationWarning, module="chromadb")


class VectorStore:
    def __init__(self, persist_directory: str, collection_name: str = "uet_documents"):
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        
        logger.info(f"Initializing ChromaDB at {persist_directory}")
        
        try:
            self.client = chromadb.PersistentClient(
                path=str(self.persist_directory),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                    is_persistent=True
                )
            )
            
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            
            logger.info(f"ChromaDB collection '{collection_name}' ready with {self.collection.count()} documents")
        except Exception as e:
            logger.error(f"Error initializing ChromaDB: {str(e)}")
            raise

    def add_documents(self, documents: List[str], metadatas: Optional[List[Dict]] = None, ids: Optional[List[str]] = None, embeddings: Optional[List[List[float]]] = None):
        try:
            if ids is None:
                ids = [f"doc_{i}" for i in range(len(documents))]
            
            if metadatas is None:
                metadatas = [{"source": "uet_document", "chunk_id": i} for i in range(len(documents))]
            
            logger.info(f"Adding {len(documents)} documents to collection")
            # If embeddings are provided, ensure they're plain python lists (Chroma expects lists)
            add_kwargs = {
                "documents": documents,
                "metadatas": metadatas,
                "ids": ids
            }
            if embeddings is not None:
                # convert numpy arrays or nested numpy arrays to lists
                if isinstance(embeddings, np.ndarray):
                    embeddings = embeddings.tolist()
                else:
                    # convert inner numpy arrays, if any
                    embeddings = [e.tolist() if isinstance(e, np.ndarray) else e for e in embeddings]

                add_kwargs["embeddings"] = embeddings

            self.collection.add(**add_kwargs)
            
            logger.info(f"Successfully added documents. Total count: {self.collection.count()}")
        except Exception as e:
            logger.error(f"Error adding documents to ChromaDB: {str(e)}")
            raise

    def query(self, query_text: str = None, n_results: int = 5, where: Dict = None, query_embeddings: Optional[List[float]] = None) -> Dict:
        try:
            logger.info(f"Querying collection for top {n_results} results")
            
            query_params = {"n_results": n_results}

            # If explicit query embeddings are provided, use them (ensures same embedder)
            if query_embeddings is not None:
                if isinstance(query_embeddings, np.ndarray):
                    query_embeddings = query_embeddings.tolist()
                query_params["query_embeddings"] = [query_embeddings]
            else:
                query_params["query_texts"] = [query_text]
            
            if where:
                query_params["where"] = where
                logger.info(f"Applying metadata filter: {where}")
            
            results = self.collection.query(**query_params)
            
            logger.info(f"Retrieved {len(results['documents'][0])} results")
            return results
        except Exception as e:
            logger.error(f"Error querying ChromaDB: {str(e)}")
            raise

    def get_all_documents(self) -> Dict:
        try:
            results = self.collection.get()
            logger.info(f"Retrieved all {len(results['documents'])} documents")
            return results
        except Exception as e:
            logger.error(f"Error retrieving all documents: {str(e)}")
            raise

    def delete_collection(self):
        try:
            self.client.delete_collection(name=self.collection_name)
            logger.info(f"Deleted collection '{self.collection_name}'")
        except Exception as e:
            logger.error(f"Error deleting collection: {str(e)}")
            raise

    def reset_collection(self):
        try:
            self.delete_collection()
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Reset collection '{self.collection_name}'")
        except Exception as e:
            logger.error(f"Error resetting collection: {str(e)}")
            raise

    def get_count(self) -> int:
        return self.collection.count()


if __name__ == "__main__":
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        store = VectorStore(persist_directory=tmpdir, collection_name="test_collection")
        
        sample_docs = [
            "The Department of Computer Science offers undergraduate programs.",
            "Admission requirements include a minimum CGPA of 3.0.",
            "The faculty includes experienced professors and researchers."
        ]
        
        store.add_documents(sample_docs)
        print(f"Added {store.get_count()} documents")
        
        results = store.query("What are the admission requirements?", n_results=2)
        print(f"\nQuery results:")
        for i, doc in enumerate(results['documents'][0]):
            print(f"{i+1}. {doc}")
