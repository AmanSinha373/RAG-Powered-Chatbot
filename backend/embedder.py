from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import os

def create_or_load_faiss_vectorstore(docs, index_path= "faiss_index"):
    # Will create FAISS vectorstore from documents or load existing one from the disk
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    if os.path.exists(os.path.join(index_path, "index.faiss")):
        print("Loading existing FAISS index. . . ")
        return FAISS.load_local(index_path, embedding, allow_dangerous_deserialization=True)
    
    if not docs:
        raise ValueError("The input document list `docs` is empty. Cannot create FAISS index.")

    print("Creating new FAISS index. . . ")
    vectorstore = FAISS.from_documents(docs, embedding)
    vectorstore.save_local(index_path)
    return vectorstore
