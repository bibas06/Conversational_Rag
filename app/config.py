import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS_DIR = os.path.join(BASE_DIR, "documents")
DB_DIR = os.path.join(BASE_DIR, "db", "chroma_db")

HF_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama-3.1-8b-instant"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

ALLOWED_IS_CODES = [
    "IS_456.pdf",
    "IS_800.pdf",
    "IS_875.pdf",
    "IS_1893.pdf",
    "IS_13920.pdf"
]
