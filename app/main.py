from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from app.schemas import ChatRequest, ChatResponse
from app.rag_pipeline import rebuild_vectorstore, ask_question
import os
import shutil
import logging
import asyncio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Civil IS Code RAG API")

UPLOAD_DIR = "documents"
DB_DIR = "db/chroma_db"
os.makedirs(UPLOAD_DIR, exist_ok=True)

ingestion_in_progress = False

@app.get("/")
async def root():
    return {"message": "Civil IS Code RAG API is running"}

@app.post("/upload")
async def upload_pdfs(files: List[UploadFile] = File(...)):
    """Upload PDF files - fast operation"""
    saved = []
    errors = []

    for file in files:
        if not file.filename.lower().endswith(".pdf"):
            errors.append(f"{file.filename} is not a PDF")
            continue

        try:
            file_path = os.path.join(UPLOAD_DIR, file.filename)
            
        
            with open(file_path, "wb") as f:
                content = await file.read()
                f.write(content)
                
            saved.append(file.filename)
            logger.info(f"Uploaded: {file.filename}")
        except Exception as e:
            errors.append(f"Failed to save {file.filename}: {str(e)}")

    response = {"status": "uploaded", "files": saved}
    if errors:
        response["errors"] = errors
    
    return response

@app.post("/ingest")
async def ingest(background_tasks: BackgroundTasks):
    """Trigger document ingestion - runs in background"""
    global ingestion_in_progress
    
    if ingestion_in_progress:
        return {"status": "ingestion already in progress"}
    
    ingestion_in_progress = True
    
    
    background_tasks.add_task(run_ingestion)
    
    return {
        "status": "ingestion started", 
        "message": "Processing in background. This may take a few minutes."
    }

async def run_ingestion():
    """Run ingestion in background"""
    global ingestion_in_progress
    try:
        logger.info("Starting background ingestion...")
        await asyncio.to_thread(rebuild_vectorstore)
        logger.info("Background ingestion completed")
    except Exception as e:
        logger.error(f"Background ingestion failed: {e}")
    finally:
        ingestion_in_progress = False

@app.get("/ingest/status")
async def ingestion_status():
    """Check ingestion status"""
    global ingestion_in_progress
    return {
        "in_progress": ingestion_in_progress,
        "vectorstore_exists": os.path.exists(DB_DIR)
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Chat with the RAG system"""
    try:
        logger.info(f"Chat request: {req.query[:50]}...")
        
        
        if not os.path.exists(DB_DIR):
            return ChatResponse(
                answer="Please upload and ingest documents first using the 'Ingest Documents' button.",
                structured_answer=None,
                confidence="LOW",
                sources=[],
                follow_up_questions=[]
            )
        
        
        result = ask_question(req.query, req.chat_history)
        
        
        answer = result.get("answer", "No answer generated")
        confidence = result.get("confidence", "MEDIUM")
        sources = result.get("sources", [])
        follow_ups = result.get("follow_up_questions", [])
        
        return ChatResponse(
            answer=answer,
            structured_answer=result,
            confidence=confidence,
            sources=sources,
            follow_up_questions=follow_ups
        )
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return ChatResponse(
            answer=f"Error: {str(e)}",
            structured_answer=None,
            confidence="LOW",
            sources=[],
            follow_up_questions=[]
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "vectorstore_exists": os.path.exists(DB_DIR),
        "documents_dir_exists": os.path.exists(UPLOAD_DIR),
        "document_count": len([f for f in os.listdir(UPLOAD_DIR) if f.endswith('.pdf')]) if os.path.exists(UPLOAD_DIR) else 0
    }
