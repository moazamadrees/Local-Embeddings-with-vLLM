import logging
from typing import List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextChunker:
    def __init__(self, chunk_size: int = 500, overlap: int = 100):
        self.chunk_size = chunk_size
        self.overlap = overlap
        logger.info(f"Initialized TextChunker with chunk_size={chunk_size}, overlap={overlap}")

    def chunk_by_words(self, text: str) -> List[str]:
        if not text:
            logger.warning("Empty text provided for chunking")
            return []
        
        words = text.split()
        total_words = len(words)
        logger.info(f"Chunking {total_words} words into chunks of {self.chunk_size} words with {self.overlap} word overlap")
        
        chunks = []
        start = 0
        
        while start < total_words:
            end = min(start + self.chunk_size, total_words)
            chunk_words = words[start:end]
            chunk_text = ' '.join(chunk_words)
            chunks.append(chunk_text)
            
            if end >= total_words:
                break
            
            start += (self.chunk_size - self.overlap)
        
        logger.info(f"Created {len(chunks)} chunks from text")
        return chunks
    
    def extract_metadata(self, chunk: str) -> dict:
        """Extract metadata from chunk to improve retrieval"""
        metadata = {
            "has_eligibility": "eligibility" in chunk.lower() or "admission" in chunk.lower() or "requirement" in chunk.lower(),
            "has_programs": "offered programs" in chunk.lower() or "programs:" in chunk.lower(),
            "has_faculty": "faculty" in chunk.lower() or "professor" in chunk.lower() or "dean" in chunk.lower(),
            "has_introduction": "introduction:" in chunk.lower() or "established" in chunk.lower()
        }
        
        import re
        dept_match = re.search(r'Department of ([A-Z][a-z\s&]+(?:Engineering|Science|Management))', chunk)
        if dept_match:
            metadata["department"] = dept_match.group(1).strip()
        
        return metadata

    def chunk_by_sentences(self, text: str) -> List[str]:
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = []
        current_word_count = 0
        
        for sentence in sentences:
            sentence_words = sentence.split()
            sentence_word_count = len(sentence_words)
            
            if current_word_count + sentence_word_count <= self.chunk_size:
                current_chunk.append(sentence)
                current_word_count += sentence_word_count
            else:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_word_count = sentence_word_count
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        logger.info(f"Created {len(chunks)} sentence-based chunks")
        return chunks

    def chunk_by_sections(self, text: str) -> List[str]:
        """Split document into logical sections (heading-aware) then chunk by sentences.

        Splits on occurrences like 'Department of ...' and numbered section headings
        to keep sections coherent before applying sentence-based chunking.
        """
        import re

        # Try to split into department/section blocks while keeping headers
        sections = re.split(r'(?=(?:\bDepartment of\b|^\d+\.|^\d+\.|\n[A-Z][A-Z\s]{5,}\n))', text, flags=re.M)

        # If split produced nothing useful, fall back to sentence chunking of whole text
        if not sections or len(sections) == 1:
            return self.chunk_by_sentences(text)

        chunks = []
        for sec in sections:
            sec = sec.strip()
            if not sec:
                continue
            # chunk each section by sentences to desired size
            # temporarily set text to section and reuse sentence-based logic
            section_chunks = self.chunk_by_sentences(sec)
            for c in section_chunks:
                chunks.append(c)

        logger.info(f"Created {len(chunks)} section-aware sentence chunks")
        return chunks


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    chunker = TextChunker(chunk_size=chunk_size, overlap=overlap)
    # prefer sentence/section-based chunking for better retrieval
    return chunker.chunk_by_sections(text)


if __name__ == "__main__":
    sample_text = " ".join([f"Word{i}" for i in range(1000)])
    
    chunks = chunk_text(sample_text, chunk_size=100, overlap=20)
    print(f"Created {len(chunks)} chunks")
    print(f"First chunk ({len(chunks[0].split())} words): {chunks[0][:100]}...")
    print(f"Second chunk ({len(chunks[1].split())} words): {chunks[1][:100]}...")
