import pytest
import sys
from pathlib import Path
import requests
import time

sys.path.append(str(Path(__file__).resolve().parent.parent))

API_URL = "http://localhost:8000"


def wait_for_api(timeout=10):
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"{API_URL}/health", timeout=2)
            if response.status_code == 200:
                return True
        except:
            time.sleep(1)
    return False


def test_api_health():
    if not wait_for_api():
        pytest.skip("API server not running")
    
    response = requests.get(f"{API_URL}/health")
    
    assert response.status_code == 200
    data = response.json()
    assert data['status'] == 'ok'
    assert data['model_loaded'] == True
    
    print(f"✓ API health check passed")


def test_root_endpoint():
    if not wait_for_api():
        pytest.skip("API server not running")
    
    response = requests.get(f"{API_URL}/")
    
    assert response.status_code == 200
    data = response.json()
    assert 'message' in data
    assert 'endpoints' in data
    
    print(f"✓ Root endpoint working")


def test_chat_endpoint_valid_question():
    if not wait_for_api():
        pytest.skip("API server not running")
    
    payload = {
        "message": "What programs does the Computer Science department offer?",
        "history": [],
        "top_k": 5
    }
    
    response = requests.post(f"{API_URL}/chat", json=payload, timeout=60)
    
    assert response.status_code == 200
    data = response.json()
    assert 'answer' in data
    assert 'citations' in data
    assert 'sources' in data
    assert len(data['answer']) > 0
    
    print(f"✓ Chat endpoint working for valid questions")


def test_chat_endpoint_guardrail():
    if not wait_for_api():
        pytest.skip("API server not running")
    
    payload = {
        "message": "What is the weather today?",
        "history": [],
        "top_k": 5
    }
    
    response = requests.post(f"{API_URL}/chat", json=payload, timeout=60)
    
    assert response.status_code == 200
    data = response.json()
    assert "I only answer department information" in data['answer']
    assert data['metadata']['guardrail_triggered'] == True
    
    print(f"✓ Guardrail working correctly")


def test_chat_endpoint_empty_message():
    if not wait_for_api():
        pytest.skip("API server not running")
    
    payload = {
        "message": "",
        "history": []
    }
    
    response = requests.post(f"{API_URL}/chat", json=payload)
    
    assert response.status_code == 400
    
    print(f"✓ Empty message validation working")


def test_stats_endpoint():
    if not wait_for_api():
        pytest.skip("API server not running")
    
    response = requests.get(f"{API_URL}/stats")
    
    assert response.status_code == 200
    data = response.json()
    assert 'total_documents' in data
    assert data['total_documents'] > 0
    
    print(f"✓ Stats endpoint working")


if __name__ == "__main__":
    print("Running API tests...\n")
    print("Note: These tests require the API server to be running.\n")
    print("Start the server with: python -m uvicorn backend.main:app --reload\n")
    
    try:
        test_api_health()
        test_root_endpoint()
        test_stats_endpoint()
        test_chat_endpoint_valid_question()
        test_chat_endpoint_guardrail()
        test_chat_endpoint_empty_message()
        
        print("\n" + "="*50)
        print("All API tests passed! ✓")
        print("="*50)
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        print("\nMake sure the API server is running:")
        print("  python -m uvicorn backend.main:app --reload")
        raise
