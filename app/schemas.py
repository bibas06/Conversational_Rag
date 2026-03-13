from pydantic import BaseModel, Field
from typing import List,Tuple, Optional

class ChatRequest(BaseModel):
    query: str
    chat_history: List[Tuple[str, str]] = []

