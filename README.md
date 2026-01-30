# UET Department RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot system for answering department-related questions from the UET Lahore prospectus. Built with local embeddings, ChromaDB vector storage, and Gemma-2 language model.

## 🎓 Project Overview

This project implements a complete RAG pipeline that:
- Extracts and processes text from UET prospectus PDF
- Generates embeddings using local sentence-transformers model
- Stores document chunks in ChromaDB vector database
- Implements guardrail layer to filter non-department questions
- Uses TinyLlama LLM for answer generation (open-source, no authentication required)
-- Provides FastAPI backend and Gradio frontend (recommended)

## 👥 Team Members & Task Division

### Khadija: Data Preprocessing Pipeline
- PDF text extraction using PyPDF2
- Text cleaning with regex
- Document chunking (500 words, 100-word overlap)
- Implementation of preprocessing modules
- FastAPI application with CORS
- REST API endpoints (/chat, /health, /stats)
- Error handling and logging
- Request/response models with Pydantic

### Moazam: RAG Engine & LLM Integration
- Vector retriever implementation
- vLLM/Transformers integration with Gemma-2
- Answer generation with citations
- Context formatting and prompt engineering
 - Gradio chat interface
- Test suite with 20 questions
- Automated testing script
- Documentation and video preparation

## 🏗️ Architecture

### Data Preprocessing Pipeline
```
PDF Document → Text Extraction → Text Cleaning → Chunking → 
Embedding Generation → Vector Storage (ChromaDB)
```

### System Architecture
```
User Query → Streamlit Frontend → FastAPI Backend → 
Guardrail Validator → Vector Retriever → LLM Generator → Response
```

**Architecture Diagrams:**
- [Preprocessing Pipeline](UET%20Department%20RAG%20System%20-%20Preprocessing.png)
- [System Architecture](UET%20Department%20RAG%20System%20-%20Architecture%20Diagram%20-%20visual%20selection.png)

## 📋 Requirements

- **OS:** Windows
- **Python:** 3.11
- **GPU:** Optional (CUDA-enabled GPU recommended for faster inference)
- **RAM:** Minimum 8GB (16GB recommended)
- **Storage:** ~5GB for models and data

## 🚀 Installation

### Step 1: Create Virtual Environment
```powershell
cd c:\Users\moaza\Downloads\nlp_final
python -m venv venv
.\venv\Scripts\activate
```

### Step 2: Install Dependencies
```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 3: Install PyTorch with GPU Support (RTX 4050)
```powershell
pip uninstall torch torchvision torchaudio -y
pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 torchaudio==2.1.0+cu118 --index-url https://download.pytorch.org/whl/cu118
```

**Models will download automatically on first run:**
- Embedding Model: sentence-transformers/all-MiniLM-L6-v2 (~90MB)
- LLM Model: TinyLlama/TinyLlama-1.1B-Chat-v1.0 (~1.1GB) - No authentication required!

## 📊 Running the Project

### Step 1: Preprocess the PDF (One-time)
```powershell
python -m backend.preprocessing.run_pipeline
```

### Step 2: Start API Server
```powershell
python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

### Step 3: Start Frontend (New Terminal)

**Gradio (Recommended)**
```powershell
.\venv\Scripts\activate
python frontend/gradio_app.py
```
Open http://localhost:7860 in your browser.

If you prefer Streamlit the project still contains `frontend/app.py` but Gradio is the maintained UI.

## 🧪 Testing

```powershell
python tests/run_tests.py
```

Tests all 20 questions and generates results in `tests/test_results.json`.

## 📁 Project Structure

```
uet-rag-chatbot/
├── backend/
│   ├── __init__.py
│   ├── main.py                      # FastAPI application
│   ├── config.py                    # Configuration settings
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── pdf_extractor.py         # PDF text extraction
│   │   ├── text_cleaner.py          # Text cleaning with regex
│   │   ├── chunker.py               # Text chunking
│   │   ├── embedder.py              # Embedding generation
│   │   ├── vector_store.py          # ChromaDB interface
│   │   └── run_pipeline.py          # Pipeline orchestration
│   ├── guardrail/
│   │   ├── __init__.py
│   │   └── scope_validator.py       # Question scope validation
│   └── rag/
│       ├── __init__.py
│       ├── retriever.py             # Vector retrieval
│       ├── llm_client.py            # LLM integration
│       └── answer_generator.py      # Answer generation
├── frontend/
│   ├── __init__.py
│   └── app.py                       # Streamlit GUI
├── data/
│   ├── raw/
│   │   └── UET lahore Document.pdf  # Source document
│   ├── processed/                   # Processed text (generated)
│   └── chroma_db/                   # Vector database (generated)
├── diagrams/
│   ├── preprocessing_pipeline.png
│   └── system_architecture.png
├── tests/
│   ├── __init__.py
│   ├── test_queries.json            # 20 test questions
│   ├── run_tests.py                 # Automated test runner
│   └── test_results.json            # Test results (generated)
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
├── VIDEO_SCRIPT.md                  # Video presentation script
├── .env.example                     # Environment variables template
└── .gitignore                       # Git ignore rules
```

## 🔧 Technical Stack

- **Python:** 3.11
- **Embeddings:** sentence-transformers/all-mpnet-base-v2 (default)
- **LLM:** TinyLlama/TinyLlama-1.1B-Chat-v1.0 (1.1GB, open-source)
- **Vector DB:** ChromaDB
- **API:** FastAPI
- **UI:** Gradio (recommended)

## 📝 License

Educational project for NLP class.
