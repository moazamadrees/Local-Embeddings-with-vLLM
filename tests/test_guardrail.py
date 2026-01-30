import pytest
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from backend.guardrail.scope_validator import ScopeValidator


def test_department_question_accepted():
    validator = ScopeValidator()
    
    department_questions = [
        "What programs does the Computer Science department offer?",
        "Who is the dean of Electrical Engineering?",
        "What are the admission requirements for M.Sc. programs?",
        "Tell me about the faculty in the department.",
        "What is the eligibility criteria for Ph.D. programs?"
    ]
    
    for question in department_questions:
        is_valid, response = validator.validate_and_respond(question)
        assert is_valid, f"Question should be accepted: {question}"
        assert response == "", "Response should be empty for valid questions"
    
    print(f"✓ All {len(department_questions)} department questions accepted")


def test_out_of_scope_question_rejected():
    validator = ScopeValidator()
    
    out_of_scope_questions = [
        "What is the weather today?",
        "How do I cook pasta?",
        "Tell me about Python programming.",
        "What is the capital of France?",
        "Who won the cricket match?"
    ]
    
    for question in out_of_scope_questions:
        is_valid, response = validator.validate_and_respond(question)
        assert not is_valid, f"Question should be rejected: {question}"
        assert "I only answer department information" in response
    
    print(f"✓ All {len(out_of_scope_questions)} out-of-scope questions rejected")


def test_edge_cases():
    validator = ScopeValidator()
    
    is_valid, response = validator.validate_and_respond("")
    assert not is_valid, "Empty question should be rejected"
    
    is_valid, response = validator.validate_and_respond("   ")
    assert not is_valid, "Whitespace-only question should be rejected"
    
    print("✓ Edge cases handled correctly")


def test_keyword_matching():
    validator = ScopeValidator()
    
    is_related, score, reason = validator.is_department_related(
        "What programs does the engineering department offer?"
    )
    
    assert is_related
    assert score > 0
    assert "program" in reason.lower() or "department" in reason.lower()
    
    print(f"✓ Keyword matching works correctly")


if __name__ == "__main__":
    print("Running guardrail tests...\n")
    
    try:
        test_department_question_accepted()
        test_out_of_scope_question_rejected()
        test_edge_cases()
        test_keyword_matching()
        
        print("\n" + "="*50)
        print("All guardrail tests passed! ✓")
        print("="*50)
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        raise
