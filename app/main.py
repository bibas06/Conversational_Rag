from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException
from app.schemas import ChatRequest
from app.rag_pipeline import rebuild_vectorstore, get_conversational_rag_chain
import os, shutil

app = FastAPI(title="Civil IS Code RAG API")

UPLOAD_DIR = "documents"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/upload")
def upload_pdfs(files: List[UploadFile] = File(...)):
    saved = []

    for file in files:
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(400, "Only PDF allowed")

        with open(os.path.join(UPLOAD_DIR, file.filename), "wb") as f:
            shutil.copyfileobj(file.file, f)

        saved.append(file.filename)

    # 🚀 NO INGESTION HERE → FAST RESPONSE
    return {"status": "uploaded", "files": saved}

@app.post("/ingest")
def ingest():
    rebuild_vectorstore()
    return {"status": "ingestion completed"}

@app.post("/chat")
def chat(req: ChatRequest):
    rag_chain = get_conversational_rag_chain()  # safe

    result = rag_chain.invoke({
        "input": req.query,
        "chat_history": req.chat_history
    })

    return {"answer": result.get("answer", "")}
