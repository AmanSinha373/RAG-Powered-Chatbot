import os
import shutil
import streamlit as st
from backend.loader import load_documents
from backend.chunker import chunk_documents
from backend.embedder import create_or_load_faiss_vectorstore
from backend.retriever import retrieve_docs
from backend.qa_chain import get_llm_response
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(page_title="RAG Chatbot")
st.title("Chat with Documents (RAG Bot)")

# Upload documents
uploaded_files = st.file_uploader(
    "Upload PDF, TXT, or DOCX files",
    type=["pdf", "txt", "docx"],
    accept_multiple_files=True
)

if uploaded_files:
    # Step 1: Clear old uploaded files
    if os.path.exists("data"):
        shutil.rmtree("data")
    os.makedirs("data", exist_ok=True)

    #  Save each uploaded file with its original name
    for uploaded_file in uploaded_files:
        file_path = os.path.join("data", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())

    # Load files
    st.info("Loading documents...")
    docs = load_documents("data")
    st.write(f" Loaded {len(docs)} documents")

    # Chunk them
    st.info("Chunking text...")
    chunks = chunk_documents(docs)
    st.write(f" Created {len(chunks)} chunks")

    # Embed and Index
    st.info("Creating vector index...")
    vectorstore = create_or_load_faiss_vectorstore(chunks)

    # Ask question
    user_query = st.text_input("Ask a question based on your documents")

    if user_query:
        with st.spinner("Thinking..."):
            relevant_docs = retrieve_docs(user_query, vectorstore)
            response = get_llm_response(user_query, relevant_docs)
            st.success(response)
