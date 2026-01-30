import logging
from pathlib import Path
from typing import Optional
import PyPDF2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PDFExtractor:
    def __init__(self, pdf_path: str):
        self.pdf_path = Path(pdf_path)
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        logger.info(f"Initialized PDFExtractor with file: {pdf_path}")

    def extract_text(self) -> str:
        try:
            text = ""
            with open(self.pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                num_pages = len(pdf_reader.pages)
                logger.info(f"Extracting text from {num_pages} pages")
                
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                    logger.debug(f"Extracted page {page_num}/{num_pages}")
            
            logger.info(f"Successfully extracted {len(text)} characters from PDF")
            return text
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            raise

    def extract_text_by_pages(self) -> list[str]:
        try:
            pages = []
            with open(self.pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                num_pages = len(pdf_reader.pages)
                
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    page_text = page.extract_text()
                    if page_text:
                        pages.append(page_text)
                    logger.debug(f"Extracted page {page_num}/{num_pages}")
            
            logger.info(f"Successfully extracted {len(pages)} pages from PDF")
            return pages
        except Exception as e:
            logger.error(f"Error extracting pages from PDF: {str(e)}")
            raise


def extract_pdf(pdf_path: str) -> str:
    extractor = PDFExtractor(pdf_path)
    return extractor.extract_text()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    else:
        from backend.config import PDF_PATH
        pdf_path = PDF_PATH
    
    text = extract_pdf(pdf_path)
    print(f"Extracted {len(text)} characters")
    print(f"First 500 characters:\n{text[:500]}")
