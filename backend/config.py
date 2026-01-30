import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

PDF_PATH = os.getenv("PDF_PATH", str(BASE_DIR / "data" / "raw" / "UET lahore Document.pdf"))
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
CHROMA_DB_DIR = BASE_DIR / "data" / "chroma_db"

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")
LLM_MODEL = os.getenv("LLM_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "250"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", "3"))

# Control whether strict metadata filters are applied during retrieval.
# When False, retrieval uses pure semantic similarity which is often better
# for semi-structured prospectus documents where strict metadata may be missing.
APPLY_METADATA_FILTER = os.getenv("APPLY_METADATA_FILTER", "False").lower() in ("1","true","yes")

API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

DEPARTMENT_KEYWORDS = [
    "program", "programs", "eligibility", "faculty", "department", "departments",
    "dean", "chairman", "m.sc", "msc", "ph.d", "phd", "engineering", "admission",
    "admissions", "course", "courses", "semester", "credit", "credits", "fee", "fees",
    "requirement", "requirements", "degree", "degrees", "undergraduate", "graduate",
    "postgraduate", "bachelor", "master", "doctorate", "curriculum", "syllabus",
    "professor", "lecturer", "instructor", "staff", "hod", "head", "contact",
    "email", "phone", "office", "building", "lab", "laboratory", "research",
    "thesis", "dissertation", "cgpa", "gpa", "merit", "scholarship", "duration"
]

GUARDRAIL_MESSAGE = "I only answer department information."
