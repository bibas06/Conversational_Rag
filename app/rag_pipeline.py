from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

from app.config import DB_DIR, HF_EMBEDDING_MODEL, LLM_MODEL, GROQ_API_KEY

def get_conversational_rag_chain():
    embeddings = HuggingFaceEmbeddings(model=HF_EMBEDDING_MODEL)

    vectorstore = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name=LLM_MODEL,
        temperature=0
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True
    )
