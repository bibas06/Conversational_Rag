from app.rag_pipeline import rebuild_vectorstore

if __name__ == "__main__":
    rebuild_vectorstore()
    print("Manual ingestion completed successfully")
