import streamlit as st
import requests
import time

API = "http://127.0.0.1:8000"

st.set_page_config("Civil IS Code RAG", layout="wide")
st.title("🏗️ Civil IS Code Conversational AI")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -------- Upload --------
st.subheader("📤 Upload PDFs")
files = st.file_uploader(
    "Upload multiple IS code PDFs",
    type=["pdf"],
    accept_multiple_files=True
)

if files:
    try:
        res = requests.post(
            f"{API}/upload",
            files=[("files", f) for f in files],
            timeout=120
        )

        if res.status_code == 200:
            st.success("Uploaded files:")
            for f in res.json().get("files", []):
                st.write("•", f)
        else:
            st.error(res.text)

    except Exception as e:
        st.error(f"Backend not running: {e}")

# ---------------- Ingest Button ----------------
st.subheader("📥 Ingest Uploaded Documents")

if st.button("Ingest Documents"):
    try:
        res = requests.post(f"{API}/ingest", timeout=300)
        if res.status_code == 200:
            st.success(res.json()["status"])
        else:
            st.error(res.text)
    except Exception as e:
        st.error(f"Ingestion failed: {e}")

# ---------------- Chat Section ----------------
st.subheader("💬 Ask Questions")

query = st.text_input("Enter your question")

col1, col2 = st.columns(2)

if col1.button("Ask") and query.strip():
    try:
        res = requests.post(
            f"{API}/chat",
            json={
                "query": query,
                "chat_history": st.session_state.chat_history
            },
            timeout=120
        )

        if res.status_code != 200:
            st.error(res.text)
        else:
            answer = res.json()["answer"]

            # UI streaming (safe)
            placeholder = st.empty()
            streamed = ""
            for word in answer.split():
                streamed += word + " "
                placeholder.markdown(streamed)
                time.sleep(0.02)

            # Save conversation
            st.session_state.chat_history.append((query, answer))

    except Exception as e:
        st.error(f"Chat failed: {e}")

if col2.button("Stop / Reset Memory"):
    st.session_state.chat_history = []
    st.success("Conversation memory cleared")