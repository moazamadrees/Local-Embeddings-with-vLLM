from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import logging
import uvicorn
from contextlib import asynccontextmanager
import warnings

from backend.rag.answer_generator import AnswerGenerator
from backend.config import API_HOST, API_PORT, LOG_LEVEL

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore", category=DeprecationWarning)

answer_generator = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global answer_generator
    logger.info("Starting up UET RAG API...")
    try:
        answer_generator = AnswerGenerator(use_vllm=False)
        logger.info("Answer generator initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize answer generator: {str(e)}")
        raise
    
    yield
    
    logger.info("Shutting down UET RAG API...")


app = FastAPI(
    title="UET Department RAG API",
    description="RAG-based chatbot for UET department information",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Message(BaseModel):
    role: str = Field(..., description="Role of the message sender (user or assistant)")
    content: str = Field(..., description="Content of the message")


class ChatRequest(BaseModel):
    message: str = Field(..., description="User's question")
    history: Optional[List[Message]] = Field(default=[], description="Chat history")
    top_k: Optional[int] = Field(default=5, description="Number of documents to retrieve")
    max_tokens: Optional[int] = Field(default=512, description="Maximum tokens for response")
    temperature: Optional[float] = Field(default=0.7, description="Temperature for generation")


class Citation(BaseModel):
    chunk_id: int
    source: str
    relevance_score: float


class ChatResponse(BaseModel):
    answer: str
    citations: List[str]
    sources: List[Dict]
    metadata: Dict


@app.get("/")
async def root():
    return {
        "message": "UET Department RAG API",
        "version": "1.0.0",
        "endpoints": {
            "POST /chat": "Send a question and get an answer",
            "GET /health": "Check API health status",
            "GET /docs": "API documentation"
        }
    }


@app.get("/health")
async def health_check():
    try:
        if answer_generator is None:
            return {
                "status": "error",
                "model_loaded": False,
                "message": "Answer generator not initialized"
            }
        
        retriever_count = answer_generator.retriever.vector_store.get_count()
        
        return {
            "status": "ok",
            "model_loaded": True,
            "documents_loaded": retriever_count,
            "message": "API is healthy and ready to serve requests"
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "error",
            "model_loaded": False,
            "message": str(e)
        }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        if answer_generator is None:
            raise HTTPException(
                status_code=503,
                detail="Answer generator not initialized. Please check server logs."
            )
        
        if not request.message or not request.message.strip():
            raise HTTPException(
                status_code=400,
                detail="Message cannot be empty"
            )
        
        logger.info(f"Received chat request: '{request.message[:100]}...'")
        
        result = answer_generator.generate_answer(
            question=request.message,
            top_k=request.top_k,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )
        
        response = ChatResponse(
            answer=result['answer'],
            citations=result['citations'],
            sources=result['sources'],
            metadata=result['metadata']
        )
        
        logger.info(f"Successfully generated response with {len(result['citations'])} citations")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@app.get("/stats")
async def get_stats():
    try:
        if answer_generator is None:
            raise HTTPException(status_code=503, detail="Service not initialized")
        
        doc_count = answer_generator.retriever.vector_store.get_count()
        
        return {
            "total_documents": doc_count,
            "collection_name": "uet_documents",
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "llm_model": "google/gemma-2-2b-it"
        }
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(
        "backend.main:app",
        host=API_HOST,
        port=API_PORT,
        reload=True,
        log_level=LOG_LEVEL.lower()
    )
