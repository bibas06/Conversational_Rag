import streamlit as st
import requests
import time
import json

API = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Civil IS Code RAG",
    page_icon="🏗️",
    layout="wide"
)

st.title("🏗️ Civil IS Code Conversational AI")
st.markdown("---")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "ingestion_status" not in st.session_state:
    st.session_state.ingestion_status = None

# Sidebar
with st.sidebar:
    st.header("🔧 System Status")
    
    # Check backend health
    try:
        health = requests.get(f"{API}/health", timeout=5)
        if health.status_code == 200:
            data = health.json()
            st.success("✅ Backend Connected")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Documents", data.get('document_count', 0))
            with col2:
                status = "✅ Ready" if data.get('vectorstore_exists') else "❌ Not Ready"
                st.metric("Vectorstore", status)
        else:
            st.error("❌ Backend Error")
    except:
        st.error("❌ Backend Not Running")
        st.info("Run: uvicorn app.main:app --reload")
        st.stop()
    
    st.markdown("---")
    st.header("📤 Upload Documents")
    
    # File uploader
    files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        key="file_uploader"
    )
    
    # Upload button
    if files and st.button("Upload Files", type="primary", use_container_width=True):
        with st.spinner("Uploading..."):
            try:
                files_to_upload = []
                for file in files:
                    files_to_upload.append(
                        ("files", (file.name, file.getvalue(), "application/pdf"))
                    )
                
                res = requests.post(
                    f"{API}/upload",
                    files=files_to_upload,
                    timeout=60
                )
                if res.status_code == 200:
                    data = res.json()
                    st.success(f"✅ Uploaded {len(data.get('files', []))} files")
                    st.session_state.ingestion_status = "files_uploaded"
                else:
                    st.error(f"Upload failed: {res.text}")
            except Exception as e:
                st.error(f"Error: {e}")
    
    st.markdown("---")
    st.header("📥 Document Processing")
    
    # Check current ingestion status
    try:
        status_res = requests.get(f"{API}/ingest/status", timeout=5)
        if status_res.status_code == 200:
            status_data = status_res.json()
            ingest_in_progress = status_data.get("in_progress", False)
            vectorstore_exists = status_data.get("vectorstore_exists", False)
            
            if ingest_in_progress:
                st.warning("⏳ Ingestion in progress...")
            elif vectorstore_exists:
                st.success("✅ Vectorstore ready")
    except:
        pass
    
    # Ingest button
    if st.button("🚀 Ingest Documents", type="secondary", use_container_width=True):
        with st.spinner("Starting ingestion..."):
            try:
                res = requests.post(f"{API}/ingest", timeout=10)
                if res.status_code == 200:
                    st.success(res.json().get("message", "Ingestion started"))
                    st.info("⏳ This will take 1-5 minutes. Check status below.")
                    st.session_state.ingestion_status = "processing"
                else:
                    st.error(f"Ingestion failed: {res.text}")
            except Exception as e:
                st.error(f"Error: {e}")
    
    # Manual status check button
    if st.button("🔄 Check Status", use_container_width=True):
        try:
            res = requests.get(f"{API}/ingest/status", timeout=5)
            if res.status_code == 200:
                data = res.json()
                if data.get("in_progress"):
                    st.warning("⏳ Ingestion in progress...")
                elif data.get("vectorstore_exists"):
                    st.success("✅ Vectorstore is ready!")
                else:
                    st.info("No vectorstore found. Click 'Ingest Documents' to create one.")
        except:
            st.error("Could not check status")
    
    st.markdown("---")
    
    # Clear chat button
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# Main chat area
st.header("💬 Ask Questions About IS Codes")

# Display welcome message if no history
if not st.session_state.chat_history:
    st.info("👋 Upload PDFs, click 'Ingest Documents', then start asking questions!")

# Chat input
query = st.chat_input("Example: What is IS 456? Explain its main provisions...")

if query:
    # Add user message
    st.session_state.chat_history.append({"role": "user", "content": query})
    
    # Get response
    with st.chat_message("user"):
        st.markdown(query)
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Prepare chat history for API
                history_tuples = []
                for i in range(0, len(st.session_state.chat_history) - 1, 2):
                    if i+1 < len(st.session_state.chat_history):
                        user_msg = st.session_state.chat_history[i]["content"]
                        ai_msg = st.session_state.chat_history[i+1]["content"]
                        history_tuples.append((user_msg, ai_msg))
                
                # Call API
                res = requests.post(
                    f"{API}/chat",
                    json={
                        "query": query,
                        "chat_history": history_tuples
                    },
                    timeout=60
                )
                
                if res.status_code == 200:
                    data = res.json()
                    answer = data.get("answer", "No response")
                    structured = data.get("structured_answer", {})
                    
                    # Display answer - if we have structured data, use it for nice formatting
                    if structured and isinstance(structured, dict):
                        # Main answer
                        st.markdown(structured.get("answer", answer))
                        
                        # Key points
                        main_points = structured.get("main_points", [])
                        if main_points:
                            st.markdown("---")
                            st.markdown("#### 🔑 Key Points")
                            for point in main_points:
                                st.markdown(f"• {point}")
                        
                        # Sources
                        sources = structured.get("sources", [])
                        if sources:
                            st.markdown("---")
                            st.markdown("#### 📚 Sources")
                            for source in sources:
                                filename = source.get('filename', 'Unknown')
                                page = source.get('page', 'N/A')
                                relevance = source.get('relevance', '')
                                st.markdown(f"• **{filename}** (Page {page}) - {relevance}")
                        
                        # Confidence
                        confidence = structured.get("confidence", "MEDIUM")
                        st.markdown("---")
                        st.markdown(f"**Confidence:** {confidence}")
                        
                        # Follow-up questions
                        follow_ups = structured.get("follow_up_questions", [])
                        if follow_ups:
                            st.markdown("---")
                            st.markdown("#### 💭 You might also ask")
                            for q in follow_ups:
                                if st.button(f"📌 {q}", key=f"followup_{len(st.session_state.chat_history)}_{q[:20]}"):
                                    # This will trigger a new query
                                    st.session_state.followup_query = q
                                    st.rerun()
                    else:
                        # Just show the answer
                        st.markdown(answer)
                    
                    # Save to history
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                else:
                    error_msg = f"Error: {res.status_code}"
                    st.error(error_msg)
                    st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
                    
            except Exception as e:
                error_msg = f"Connection error: {e}"
                st.error(error_msg)
                st.session_state.chat_history.append({"role": "assistant", "content": error_msg})

# Handle follow-up button clicks
if "followup_query" in st.session_state:
    query = st.session_state.followup_query
    del st.session_state.followup_query
    # This will trigger the chat input processing on next rerun
    st.rerun()

# Display chat history (for scrolling)
for message in st.session_state.chat_history[:-1] if query else st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])