import logging
import sys
from pathlib import Path
import warnings

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from backend.config import PDF_PATH, CHROMA_DB_DIR, CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL
from backend.preprocessing.pdf_extractor import PDFExtractor
from backend.preprocessing.text_cleaner import TextCleaner
from backend.preprocessing.chunker import TextChunker
from backend.preprocessing.embedder import TextEmbedder
from backend.preprocessing.vector_store import VectorStore

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore", category=DeprecationWarning)


def run_preprocessing_pipeline():
    logger.info("=" * 80)
    logger.info("Starting UET Document Preprocessing Pipeline")
    logger.info("=" * 80)
    
    try:
        logger.info(f"Step 1: Extracting text from PDF: {PDF_PATH}")
        extractor = PDFExtractor(PDF_PATH)
        raw_text = extractor.extract_text()
        logger.info(f"Extracted {len(raw_text)} characters from PDF")
        
        logger.info("\nStep 2: Cleaning extracted text")
        cleaner = TextCleaner()
        cleaned_text = cleaner.clean(raw_text)
        logger.info(f"Cleaned text: {len(cleaned_text)} characters")
        
        logger.info(f"\nStep 3: Chunking text (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
        chunker = TextChunker(chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
        # Use section-aware sentence chunking for better semantic chunks
        chunks = chunker.chunk_by_sections(cleaned_text)
        logger.info(f"Created {len(chunks)} chunks")
        
        logger.info(f"\nStep 4: Generating embeddings using {EMBEDDING_MODEL}")
        embedder = TextEmbedder(model_name=EMBEDDING_MODEL)
        logger.info(f"Embedding dimension: {embedder.get_embedding_dimension()}")
        
        logger.info(f"\nStep 5: Storing chunks in ChromaDB at {CHROMA_DB_DIR}")
        vector_store = VectorStore(persist_directory=str(CHROMA_DB_DIR))

        vector_store.reset_collection()

        # Precompute embeddings for all chunks and pass them to Chroma
        metadatas = []
        for i, chunk in enumerate(chunks):
            metadata = chunker.extract_metadata(chunk)
            metadata["chunk_id"] = i
            metadata["source"] = "uet_document"
            metadatas.append(metadata)

        # generate embeddings in batches
        embeddings = embedder.embed_batch(chunks, batch_size=64, show_progress=True)

        vector_store.add_documents(
            documents=chunks,
            metadatas=metadatas,
            ids=[f"chunk_{i}" for i in range(len(chunks))],
            embeddings=embeddings
        )
        
        logger.info("\n" + "=" * 80)
        logger.info("Preprocessing Pipeline Completed Successfully!")
        logger.info("=" * 80)
        logger.info(f"Total chunks stored: {vector_store.get_count()}")
        logger.info(f"ChromaDB location: {CHROMA_DB_DIR}")
        
        logger.info("\nTesting retrieval...")
        test_query = "What are the admission requirements?"
        try:
            # compute query embedding with same embedder used above
            query_emb = embedder.embed_text(test_query)
            results = vector_store.query(query_text=test_query, n_results=3, query_embeddings=query_emb)
            logger.info(f"\nTest query: '{test_query}'")
            logger.info("Top 3 results:")
            for i, doc in enumerate(results['documents'][0], 1):
                logger.info(f"{i}. {doc[:200]}...")
        except Exception as e:
            logger.error(f"Test retrieval failed: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        return False


if __name__ == "__main__":
    success = run_preprocessing_pipeline()
    sys.exit(0 if success else 1)
