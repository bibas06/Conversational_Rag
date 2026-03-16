from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.memory import ConversationBufferMemory
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from typing import List, Optional, Dict, Any, Tuple
import os
import time

# Pydantic imports
from pydantic import BaseModel, Field

from app.config import *


class SourceInfo(BaseModel):
    """Information about the source document"""
    filename: str
    page: Optional[int] = None
    relevance: str = "Supporting"

class StructuredAnswer(BaseModel):
    """Structured answer format"""
    answer: str
    confidence: str = "MEDIUM"
    main_points: List[str] = []
    sources: List[SourceInfo] = []
    follow_up_questions: List[str] = []



def rebuild_vectorstore(progress_callback=None):
    """Rebuild the vector store from PDF documents with progress tracking"""
    documents = []
    
    if not os.path.exists(DOCS_DIR):
        os.makedirs(DOCS_DIR)
        print(f"Created documents directory: {DOCS_DIR}")
        return
    
    pdf_files = [f for f in os.listdir(DOCS_DIR) if f.lower().endswith(".pdf")]
    total_files = len(pdf_files)
    
    if total_files == 0:
        print("No PDF files found to process")
        return
    
    print(f"Found {total_files} PDF files to process")
    if progress_callback:
        progress_callback(0, f"Found {total_files} PDF files")
    
    for idx, file in enumerate(pdf_files):
        file_path = os.path.join(DOCS_DIR, file)
        print(f"Loading: {file}")
        if progress_callback:
            progress_callback(
                int((idx / total_files) * 30), 
                f"Loading {file}..."
            )
        
        try:
            loader = PyPDFLoader(file_path)
            loaded_docs = loader.load()
            # Add enhanced metadata
            for i, doc in enumerate(loaded_docs):
                doc.metadata.update({
                    "source_file": file,
                    "page": doc.metadata.get("page", i + 1),
                })
            documents.extend(loaded_docs)
            print(f"  Loaded {len(loaded_docs)} pages from {file}")
        except Exception as e:
            print(f"  Error loading {file}: {e}")
    
    if not documents:
        print("No documents could be loaded")
        return
    
    print(f"\nLoaded {len(documents)} total document pages")
    if progress_callback:
        progress_callback(40, f"Loaded {len(documents)} pages, splitting into chunks...")
    
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
    )
    
    splits = splitter.split_documents(documents)
    print(f"Created {len(splits)} text chunks")
    
    if progress_callback:
        progress_callback(60, f"Created {len(splits)} chunks, initializing embeddings...")
    
    print("Initializing embeddings model...")
    embeddings = HuggingFaceEmbeddings(
        model_name=HF_EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    
    print(f"Creating vectorstore in {DB_DIR}...")
    if progress_callback:
        progress_callback(80, "Creating vectorstore (this may take a while)...")
    

    if os.path.exists(DB_DIR):
        import shutil
        shutil.rmtree(DB_DIR)
        print(f"Removed existing vectorstore at {DB_DIR}")
    
    vectorstore = Chroma.from_documents(
        splits,
        embeddings,
        persist_directory=DB_DIR,
    )
    vectorstore.persist()
    
    print(f"✅ Vectorstore persisted to {DB_DIR}")
    if progress_callback:
        progress_callback(100, "✅ Ingestion complete!")
    
    return vectorstore



def extract_sources_from_context(docs) -> List[Dict]:
    """Extract source information from retrieved documents"""
    sources = []
    seen = set()
    
    for i, doc in enumerate(docs[:3]):  
        source_file = doc.metadata.get("source_file", "Unknown")
        page = doc.metadata.get("page", 0)
        
        # Create unique key
        key = f"{source_file}_{page}"
        
        if key not in seen:
            seen.add(key)
            
           
            relevance = "Direct" if i == 0 else "Supporting"
            
            sources.append({
                "filename": source_file,
                "page": page,
                "relevance": relevance,
            })
    
    return sources

def format_docs(docs):
    """Format documents for context"""
    return "\n\n".join(doc.page_content for doc in docs)

def format_response_as_text(structured_response: Dict) -> str:
    """Format structured response as nice text without JSON"""
    lines = []
    
    lines.append(structured_response.get("answer", ""))
    lines.append("")
    
    
    confidence = structured_response.get("confidence", "MEDIUM")
    lines.append(f"📊 **Confidence:** {confidence}")
    lines.append("")
    
  
    main_points = structured_response.get("main_points", [])
    if main_points:
        lines.append("**🔑 Key Points:**")
        for point in main_points:
            lines.append(f"• {point}")
        lines.append("")
    
   
    sources = structured_response.get("sources", [])
    if sources:
        lines.append("**📚 Sources:**")
        for source in sources:
            filename = source.get('filename', 'Unknown')
            page = source.get('page', 'N/A')
            relevance = source.get('relevance', '')
            lines.append(f"• {filename} (Page {page}) - {relevance}")
        lines.append("")
    
   
    follow_ups = structured_response.get("follow_up_questions", [])
    if follow_ups:
        lines.append("**💭 You might also ask:**")
        for q in follow_ups:
            lines.append(f"• {q}")
    
    return "\n".join(lines)


_chain = None
_memory = None

def get_conversational_rag_chain():
    """Get a conversational RAG chain"""
    global _chain, _memory
    
    if _chain is not None and _memory is not None:
        return _chain, _memory
    

    if not os.path.exists(DB_DIR):
        raise ValueError(f"Vectorstore directory {DB_DIR} does not exist. Please rebuild the vectorstore first.")
    
   
    print("Initializing embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name=HF_EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    print("Loading vectorstore...")
    vectorstore = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings,
    )
    
 
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 5}
    )
    
   
    print("Initializing LLM...")
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name=LLM_MODEL,
        temperature=0.1,
        max_tokens=1024
    )
    
   
    _memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given a chat history and the latest user question, reformulate it as a standalone question."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])
    
    
    history_aware_retriever = (
        RunnablePassthrough.assign(
            chat_history=lambda x: x.get("chat_history", [])
        )
        | contextualize_q_prompt
        | llm
        | StrOutputParser()
        | retriever
    )
    
   
    system_prompt = """You are an expert assistant for Indian Civil Engineering IS codes. Answer based ONLY on the provided context.

CONTEXT:
{context}

INSTRUCTIONS:
1. Provide a clear, comprehensive answer to the question
2. If the answer isn't in the context, say "I cannot find this information in the provided documents"
3. Include 2-4 key points that support your answer
4. List the sources you used (filenames from the context)
5. Suggest 2-3 follow-up questions the user might ask


FORMAT YOUR RESPONSE LIKE THIS:

[Your detailed answer here]

🔑 KEY POINTS:
• First key point
• Second key point
• Third key point

📚 SOURCES:
• [Filename] (Page X)
• [Filename] (Page Y)


💭 YOU MIGHT ALSO ASK:
• Follow-up question 1?
• Follow-up question 2?
• Follow-up question 3?

IMPORTANT: Do NOT output JSON. Use the format above with clear section headers."""

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])
    
    
    def combine_docs(input_dict):
        docs = input_dict.get("context", [])
        formatted_docs = format_docs(docs)
        return qa_prompt.invoke({
            "context": formatted_docs,
            "chat_history": input_dict.get("chat_history", []),
            "input": input_dict.get("input", "")
        })
    
  
    retrieval_chain = (
        RunnablePassthrough.assign(
            context=history_aware_retriever
        )
        | RunnableLambda(combine_docs)
        | llm
        | StrOutputParser()
    )
    
    
    def parse_response(response_text):
        """Parse the text response into structured format"""
        result = {
            "answer": response_text,
            "confidence": "MEDIUM",
            "main_points": [],
            "sources": [],
            "follow_up_questions": []
        }
        
        lines = response_text.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
           
            if "🔑 KEY POINTS" in line or "KEY POINTS" in line:
                current_section = "key_points"
                continue
            elif "📚 SOURCES" in line or "SOURCES" in line:
                current_section = "sources"
                continue
            elif "📊 CONFIDENCE" in line or "CONFIDENCE" in line:
                current_section = "confidence"
               
                if "HIGH" in line:
                    result["confidence"] = "HIGH"
                elif "MEDIUM" in line:
                    result["confidence"] = "MEDIUM"
                elif "LOW" in line:
                    result["confidence"] = "LOW"
                continue
            elif "💭 YOU MIGHT ALSO ASK" in line or "FOLLOW-UP" in line:
                current_section = "follow_up"
                continue
            elif current_section is None and not result["answer"]:
                
                if result["answer"] == response_text:
                    result["answer"] = line
                else:
                    result["answer"] += "\n" + line
                continue
            
            
            if current_section == "key_points" and (line.startswith('•') or line.startswith('-')):
                point = line.lstrip('•- ').strip()
                if point:
                    result["main_points"].append(point)
            elif current_section == "sources" and (line.startswith('•') or line.startswith('-')):
                result["sources"].append({"filename": line.lstrip('•- ').strip()})
            elif current_section == "follow_up" and (line.startswith('•') or line.startswith('-')):
                question = line.lstrip('•- ').strip()
                if question:
                    result["follow_up_questions"].append(question)
        
        return result
    
   
    def process_response(response_text):
        parsed = parse_response(response_text)
        
        parsed["raw_text"] = response_text
        return parsed
    
    _chain = retrieval_chain | RunnableLambda(process_response)
    
    return _chain, _memory

def ask_question(question: str, chat_history: List[Tuple[str, str]] = None) -> Dict[str, Any]:
    """Ask a question and get a structured response"""
  
    chain, memory = get_conversational_rag_chain()
    
   
    if chat_history:
        for user_msg, ai_msg in chat_history:
            if user_msg:
                memory.chat_memory.add_user_message(user_msg)
            if ai_msg:
                memory.chat_memory.add_ai_message(ai_msg)
    
   
    try:
        response = chain.invoke({
            "input": question,
            "chat_history": memory.chat_memory.messages
        })
        
        
        answer_text = response.get("answer", str(response))
        memory.chat_memory.add_user_message(question)
        memory.chat_memory.add_ai_message(answer_text)
        
        
        if not response.get("sources") and hasattr(response, 'context'):
            response["sources"] = extract_sources_from_context(response.context)
        
        return response
        
    except Exception as e:
        print(f"Error in ask_question: {e}")
        return {
            "answer": f"I encountered an error: {str(e)}",
            "main_points": [],
            "sources": [],
            "follow_up_questions": []
        }
