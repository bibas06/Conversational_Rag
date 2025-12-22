import streamlit as st
from app.rag_pipeline import get_conversational_rag_chain

st.set_page_config(page_title="Civil IS Codes RAG", layout="wide")

st.title("🏗️ Civil Engineering Conversational AI")
st.write("Chat with selected IS Codes using memory-enabled RAG")

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = get_conversational_rag_chain()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

query = st.text_input("Ask a question about IS codes")

if st.button("Ask"):
    if query.strip():
        with st.spinner("Analyzing IS codes..."):
            result = st.session_state.rag_chain({"question": query})
            answer = result["answer"]

            st.session_state.chat_history.append(("You", query))
            st.session_state.chat_history.append(("AI", answer))

for role, msg in st.session_state.chat_history:
    if role == "You":
        st.markdown(f"**🧑 You:** {msg}")
    else:
        st.markdown(f"**🤖 AI:** {msg}")
