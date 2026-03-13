from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.memory import ConversationBufferMemory
from langchain_community.document_loaders import PyPDFLoader
from app.config import *

def rebuild_vectorstore():
    documents = []

    for file in os.listdir(DOCS_DIR):
        if file.lower().endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(DOCS_DIR, file))
            documents.extend(loader.load())

    if not documents:
        return

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    splits = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name=HF_EMBEDDING_MODEL)

    Chroma.from_documents(
        splits,
        embeddings,
        persist_directory=DB_DIR
    ).persist()
def get_conversational_rag_chain():
    
    embeddings = HuggingFaceEmbeddings(model_name=HF_EMBEDDING_MODEL)
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


    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            ("human", "Given the above conversation, generate a search query to look up to get information relevant to the conversation."),
        ]
    )


    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )


    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Answer the user's questions based on the below context:\n\n{context}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )

    
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

   
    conversational_rag_chain = create_retrieval_chain(
        history_aware_retriever, question_answer_chain
    )

    return conversational_rag_chain