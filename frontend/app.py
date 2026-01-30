import streamlit as st
import requests
from typing import List, Dict
import json

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="UET Department Chatbot",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stTextInput > div > div > input {
        background-color: white;
    }
    .user-message {
        background-color: #e3f2fd;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #1976d2;
    }
    .user-message strong {
        color: #0d47a1;
    }
    .bot-message {
        background-color: #f5f5f5;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #424242;
        border: 1px solid #e0e0e0;
    }
    .bot-message strong {
        color: #212121;
    }
    .citation {
        background-color: #fff3e0;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
        font-size: 0.9em;
        border-left: 3px solid #ff9800;
    }
    .status-indicator {
        padding: 5px 10px;
        border-radius: 5px;
        font-weight: bold;
    }
    .status-ok {
        background-color: #4CAF50;
        color: white;
    }
    .status-error {
        background-color: #f44336;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)


def check_api_health():
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return data.get("status") == "ok", data
        return False, {}
    except Exception as e:
        return False, {"error": str(e)}


def send_message(message: str, history: List[Dict], top_k: int = 5):
    try:
        payload = {
            "message": message,
            "history": history,
            "top_k": top_k,
            "max_tokens": 512,
            "temperature": 0.7
        }
        
        response = requests.post(
            f"{API_URL}/chat",
            json=payload,
            timeout=60
        )
        
        if response.status_code == 200:
            return response.json(), None
        else:
            return None, f"Error {response.status_code}: {response.text}"
            
    except requests.exceptions.Timeout:
        return None, "Request timed out. The model might be loading or processing."
    except Exception as e:
        return None, f"Error: {str(e)}"


def get_stats():
    try:
        response = requests.get(f"{API_URL}/stats", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None


if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'api_status' not in st.session_state:
    st.session_state.api_status = None


st.title("🎓 UET Department Information Chatbot")
st.markdown("Ask questions about UET departments, programs, admissions, and more!")

with st.sidebar:
    st.header("⚙️ Settings")
    
    st.subheader("System Status")
    if st.button("Check API Status", use_container_width=True):
        is_healthy, health_data = check_api_health()
        st.session_state.api_status = (is_healthy, health_data)
    
    if st.session_state.api_status:
        is_healthy, health_data = st.session_state.api_status
        if is_healthy:
            st.markdown('<div class="status-indicator status-ok">✓ API Online</div>', unsafe_allow_html=True)
            st.info(f"📚 Documents loaded: {health_data.get('documents_loaded', 'N/A')}")
        else:
            st.markdown('<div class="status-indicator status-error">✗ API Offline</div>', unsafe_allow_html=True)
            st.error(health_data.get('message', 'Unknown error'))
    
    st.divider()
    
    st.subheader("Retrieval Settings")
    top_k = st.slider("Number of context chunks", min_value=1, max_value=10, value=5)
    
    st.divider()
    
    stats = get_stats()
    if stats:
        st.subheader("📊 System Info")
        st.write(f"**Total Documents:** {stats.get('total_documents', 'N/A')}")
        st.write(f"**Embedding Model:** {stats.get('embedding_model', 'N/A').split('/')[-1]}")
        st.write(f"**LLM Model:** {stats.get('llm_model', 'N/A').split('/')[-1]}")
    
    st.divider()
    
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    st.divider()
    
    with st.expander("ℹ️ About"):
        st.markdown("""
        This chatbot uses RAG (Retrieval-Augmented Generation) to answer questions about UET departments.
        
        **Features:**
        - Local embeddings with sentence-transformers
        - ChromaDB vector storage
        - TinyLlama language model
        - Guardrail for scope validation
        
        **Note:** Only department-related questions will be answered.
        """)


chat_container = st.container()

with chat_container:
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f'<div class="user-message"><strong>You:</strong><br><span style="color: #212121;">{msg["content"]}</span></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="bot-message"><strong>Assistant:</strong><br><span style="color: #212121;">{msg["content"]}</span></div>', unsafe_allow_html=True)
            
            if "citations" in msg and msg["citations"]:
                with st.expander(f"📚 View {len(msg['citations'])} Citations"):
                    for i, citation in enumerate(msg["citations"], 1):
                        st.markdown(f'<div class="citation"><strong>Citation {i}:</strong><br>{citation[:300]}...</div>', unsafe_allow_html=True)


user_input = st.chat_input("Ask a question about UET departments...")

if user_input:
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })
    
    with st.spinner("🤔 Thinking..."):
        history = [
            {"role": msg["role"], "content": msg["content"]}
            for msg in st.session_state.messages[:-1]
        ]
        
        response_data, error = send_message(user_input, history, top_k)
        
        if error:
            st.error(f"❌ {error}")
            st.session_state.messages.pop()
        else:
            answer = response_data.get("answer", "No answer generated")
            citations = response_data.get("citations", [])
            sources = response_data.get("sources", [])
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "citations": citations,
                "sources": sources
            })


st.divider()

st.markdown("**Example Questions:**")
col1, col2, col3 = st.columns(3)

with col1:
    st.info("What are the admission requirements for Computer Science?")

with col2:
    st.info("Tell me about the faculty in the Electrical Engineering department.")

with col3:
    st.info("What programs are offered by the engineering departments?")
