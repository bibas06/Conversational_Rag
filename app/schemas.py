from pydantic import BaseModel, Field
from typing import List, Tuple, Optional, Dict, Any

class ChatRequest(BaseModel):
    query: str
    chat_history: List[Tuple[str, str]] = []

class ChatResponse(BaseModel):
    answer: str
    structured_answer: Optional[Dict[str, Any]] = None
    confidence: Optional[str] = None
    sources: Optional[List[Dict[str, Any]]] = None
    follow_up_questions: Optional[List[str]] = None