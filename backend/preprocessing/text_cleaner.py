import re
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextCleaner:
    def __init__(self):
        logger.info("Initialized TextCleaner")

    def clean(self, text: str) -> str:
        if not text:
            logger.warning("Empty text provided for cleaning")
            return ""
        
        original_length = len(text)
        
        cleaned_text = self._remove_special_characters(text)
        cleaned_text = self._normalize_whitespace(cleaned_text)
        cleaned_text = self._remove_extra_newlines(cleaned_text)
        cleaned_text = self._fix_common_ocr_errors(cleaned_text)
        
        logger.info(f"Cleaned text: {original_length} -> {len(cleaned_text)} characters")
        return cleaned_text.strip()

    def _remove_special_characters(self, text: str) -> str:
        text = re.sub(r'[^\w\s\.\,\;\:\?\!\-\(\)\[\]\{\}\/\'\"\%\$\#\@\&\+\=\*]', ' ', text)
        return text

    def _normalize_whitespace(self, text: str) -> str:
        text = re.sub(r'[ \t]+', ' ', text)
        return text

    def _remove_extra_newlines(self, text: str) -> str:
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        return text

    def _fix_common_ocr_errors(self, text: str) -> str:
        text = re.sub(r'(\w)-\s+(\w)', r'\1\2', text)
        text = re.sub(r'\s+([.,;:!?])', r'\1', text)
        return text


def clean_text(text: str) -> str:
    cleaner = TextCleaner()
    return cleaner.clean(text)


if __name__ == "__main__":
    sample_text = """
    This   is    a   sample    text   with    extra     spaces.
    
    
    
    It has multiple newlines and special characters like: @#$%^&*()
    
    It also has hy- phenated words that need fixing.
    """
    
    cleaned = clean_text(sample_text)
    print("Original:")
    print(sample_text)
    print("\nCleaned:")
    print(cleaned)
