import pytest
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from backend.preprocessing.pdf_extractor import PDFExtractor
from backend.preprocessing.text_cleaner import TextCleaner
from backend.preprocessing.chunker import TextChunker
from backend.preprocessing.embedder import TextEmbedder
from backend.config import PDF_PATH


def test_pdf_extraction():
    extractor = PDFExtractor(PDF_PATH)
    text = extractor.extract_text()
    
    assert text is not None
    assert len(text) > 0
    assert isinstance(text, str)
    print(f"✓ PDF extraction successful: {len(text)} characters")


def test_text_cleaning():
    cleaner = TextCleaner()
    
    dirty_text = "This   has   extra   spaces  and special chars @#$%"
    cleaned = cleaner.clean(dirty_text)
    
    assert cleaned is not None
    assert len(cleaned) > 0
    assert "  " not in cleaned
    print(f"✓ Text cleaning successful")


def test_chunking():
    chunker = TextChunker(chunk_size=100, overlap=20)
    
    text = " ".join([f"word{i}" for i in range(500)])
    chunks = chunker.chunk_by_words(text)
    
    assert len(chunks) > 0
    assert all(len(chunk.split()) <= 100 for chunk in chunks)
    print(f"✓ Chunking successful: {len(chunks)} chunks created")


def test_embedding_generation():
    embedder = TextEmbedder()
    
    test_texts = [
        "Computer Science department offers various programs.",
        "Electrical Engineering has excellent faculty."
    ]
    
    embeddings = embedder.embed_batch(test_texts, show_progress=False)
    
    assert len(embeddings) == len(test_texts)
    assert embeddings[0].shape[0] == embedder.get_embedding_dimension()
    print(f"✓ Embedding generation successful: {embeddings[0].shape}")


if __name__ == "__main__":
    print("Running preprocessing tests...\n")
    
    try:
        test_pdf_extraction()
        test_text_cleaning()
        test_chunking()
        test_embedding_generation()
        
        print("\n" + "="*50)
        print("All preprocessing tests passed! ✓")
        print("="*50)
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        raise
