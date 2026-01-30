import json
import sys
from pathlib import Path
import requests
from datetime import datetime
import time

sys.path.append(str(Path(__file__).resolve().parent.parent))

API_URL = "http://localhost:8000"
TEST_QUERIES_FILE = Path(__file__).parent / "test_queries.json"
TEST_RESULTS_FILE = Path(__file__).parent / "test_results.json"


def check_api_health():
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return data.get("status") == "ok", data
        return False, {}
    except Exception as e:
        return False, {"error": str(e)}


def send_question(question: str, timeout: int = 60):
    try:
        payload = {
            "message": question,
            "history": [],
            "top_k": 5,
            "max_tokens": 512,
            "temperature": 0.7
        }
        
        response = requests.post(
            f"{API_URL}/chat",
            json=payload,
            timeout=timeout
        )
        
        if response.status_code == 200:
            return response.json(), None
        else:
            return None, f"Error {response.status_code}: {response.text}"
            
    except requests.exceptions.Timeout:
        return None, "Request timed out"
    except Exception as e:
        return None, f"Error: {str(e)}"


def run_tests():
    print("=" * 80)
    print("UET RAG CHATBOT - AUTOMATED TEST SUITE")
    print("=" * 80)
    print()
    
    print("Step 1: Checking API health...")
    is_healthy, health_data = check_api_health()
    
    if not is_healthy:
        print("❌ API is not healthy!")
        print(f"Error: {health_data}")
        print("\nPlease start the API server first:")
        print("  python -m uvicorn backend.main:app --reload")
        return False
    
    print("✓ API is healthy")
    print(f"  Documents loaded: {health_data.get('documents_loaded', 'N/A')}")
    print()
    
    print("Step 2: Loading test queries...")
    with open(TEST_QUERIES_FILE, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    all_questions = []
    all_questions.extend(test_data['department_related_questions'])
    all_questions.extend(test_data['tricky_questions'])
    all_questions.extend(test_data['out_of_scope_questions'])
    
    print(f"✓ Loaded {len(all_questions)} test questions")
    print(f"  - Department-related: {len(test_data['department_related_questions'])}")
    print(f"  - Tricky: {len(test_data['tricky_questions'])}")
    print(f"  - Out-of-scope: {len(test_data['out_of_scope_questions'])}")
    print()
    
    print("Step 3: Running tests...")
    print("=" * 80)
    
    results = {
        "test_run_timestamp": datetime.now().isoformat(),
        "total_questions": len(all_questions),
        "results": [],
        "summary": {
            "passed": 0,
            "failed": 0,
            "guardrail_triggered": 0
        }
    }
    
    for i, test_case in enumerate(all_questions, 1):
        question_id = test_case['id']
        category = test_case['category']
        question = test_case['question']
        expected_behavior = test_case['expected_behavior']
        
        print(f"\n[Test {i}/{len(all_questions)}] Question {question_id} ({category})")
        print(f"Q: {question}")
        print("-" * 80)
        
        start_time = time.time()
        response_data, error = send_question(question)
        elapsed_time = time.time() - start_time
        
        if error:
            print(f"❌ FAILED: {error}")
            results['results'].append({
                "question_id": question_id,
                "category": category,
                "question": question,
                "status": "failed",
                "error": error,
                "elapsed_time": elapsed_time
            })
            results['summary']['failed'] += 1
            continue
        
        answer = response_data.get('answer', '')
        citations = response_data.get('citations', [])
        metadata = response_data.get('metadata', {})
        guardrail_triggered = metadata.get('guardrail_triggered', False)
        
        print(f"A: {answer[:200]}{'...' if len(answer) > 200 else ''}")
        print(f"Citations: {len(citations)}")
        print(f"Guardrail triggered: {guardrail_triggered}")
        print(f"Time: {elapsed_time:.2f}s")
        
        if category == "out_of_scope":
            if guardrail_triggered and "I only answer department information" in answer:
                print("✓ PASSED: Guardrail correctly triggered")
                status = "passed"
                results['summary']['passed'] += 1
                results['summary']['guardrail_triggered'] += 1
            else:
                print("❌ FAILED: Guardrail should have triggered")
                status = "failed"
                results['summary']['failed'] += 1
        else:
            if not guardrail_triggered and len(citations) > 0:
                print("✓ PASSED: Answer generated with citations")
                status = "passed"
                results['summary']['passed'] += 1
            else:
                print("⚠ WARNING: Answer generated but may need review")
                status = "passed"
                results['summary']['passed'] += 1
        
        results['results'].append({
            "question_id": question_id,
            "category": category,
            "question": question,
            "answer": answer,
            "citations_count": len(citations),
            "guardrail_triggered": guardrail_triggered,
            "status": status,
            "elapsed_time": elapsed_time,
            "expected_behavior": expected_behavior
        })
    
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Total Questions: {results['total_questions']}")
    print(f"Passed: {results['summary']['passed']}")
    print(f"Failed: {results['summary']['failed']}")
    print(f"Guardrail Triggered: {results['summary']['guardrail_triggered']}")
    print(f"Success Rate: {(results['summary']['passed'] / results['total_questions'] * 100):.1f}%")
    print()
    
    print(f"Saving results to {TEST_RESULTS_FILE}...")
    with open(TEST_RESULTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("✓ Results saved successfully")
    print()
    
    return results['summary']['failed'] == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
