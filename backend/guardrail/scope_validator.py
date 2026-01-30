import logging
import re
from typing import Tuple
from backend.config import DEPARTMENT_KEYWORDS, GUARDRAIL_MESSAGE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ScopeValidator:
    def __init__(self, keywords: list = None, threshold: float = 0.15):
        self.keywords = keywords or DEPARTMENT_KEYWORDS
        self.threshold = threshold
        self.keywords_lower = [kw.lower() for kw in self.keywords]
        logger.info(f"Initialized ScopeValidator with {len(self.keywords)} keywords and threshold={threshold}")

    def is_department_related(self, question: str) -> Tuple[bool, float, str]:
        if not question or not question.strip():
            logger.warning("Empty question provided")
            return False, 0.0, "Empty question"
        
        question_lower = question.lower()
        question_words = re.findall(r'\b\w+\b', question_lower)
        
        if not question_words:
            return False, 0.0, "No valid words in question"
        
        keyword_matches = []
        for keyword in self.keywords_lower:
            if keyword in question_lower:
                keyword_matches.append(keyword)
        
        match_score = len(keyword_matches) / len(question_words)
        
        is_related = match_score >= self.threshold or len(keyword_matches) >= 2
        
        reason = f"Matched {len(keyword_matches)} keywords: {keyword_matches[:5]}" if keyword_matches else "No keyword matches"
        
        logger.info(f"Question validation: is_related={is_related}, score={match_score:.3f}, reason={reason}")
        
        return is_related, match_score, reason

    def validate_and_respond(self, question: str) -> Tuple[bool, str]:
        is_related, score, reason = self.is_department_related(question)
        
        if not is_related:
            logger.info(f"Question rejected: {reason}")
            return False, GUARDRAIL_MESSAGE
        
        logger.info(f"Question accepted: {reason}")
        return True, ""

    def add_keywords(self, new_keywords: list):
        self.keywords.extend(new_keywords)
        self.keywords_lower = [kw.lower() for kw in self.keywords]
        logger.info(f"Added {len(new_keywords)} new keywords. Total: {len(self.keywords)}")


def validate_question(question: str) -> Tuple[bool, str]:
    validator = ScopeValidator()
    return validator.validate_and_respond(question)


if __name__ == "__main__":
    validator = ScopeValidator()
    
    test_questions = [
        "What are the admission requirements for the Computer Science department?",
        "Who is the chairman of the Electrical Engineering department?",
        "What is the fee structure for M.Sc. programs?",
        "What is the weather like today?",
        "How do I cook pasta?",
        "Tell me about Python programming.",
        "What programs does the engineering department offer?",
        "Who won the cricket match yesterday?"
    ]
    
    print("Testing Scope Validator:")
    print("=" * 80)
    
    for question in test_questions:
        is_valid, response = validator.validate_and_respond(question)
        status = "✓ ACCEPTED" if is_valid else "✗ REJECTED"
        print(f"\n{status}")
        print(f"Q: {question}")
        if not is_valid:
            print(f"R: {response}")
