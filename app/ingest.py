import os
import sys
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from app.config import DB_DIR, HF_EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, DOCS_DIR, ALLOWED_IS_CODES

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def ingest_documents():
    if os.path.exists(DB_DIR):
        print("✅ Chroma DB already exists. Skipping ingestion.")
        return

    documents = []

    print("📂 Loading selected IS code PDFs...")
    for file_name in ALLOWED_IS_CODES:
        file_path = os.path.join(DOCS_DIR, file_name)
        if os.path.exists(file_path):
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            documents.extend(docs)
        else:
            print(f"⚠️ Warning: {file_path} not found!")

    print(f"📄 Total pages loaded: {len(documents)}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(documents)
    print(f"✂️ Total chunks created: {len(chunks)}")

    embeddings = HuggingFaceEmbeddings(model=HF_EMBEDDING_MODEL)

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIR
    )

    print("✅ Selected IS code documents indexed successfully.")

if __name__ == "__main__":
    ingest_documents()
