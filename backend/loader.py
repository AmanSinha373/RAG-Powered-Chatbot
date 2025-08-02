# It will load pdf txt docs using langchain loaders
import os 
from langchain.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from typing import List

def load_documents(folder_path: str) -> List:
    documents = []
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)

        if file.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif file.endswith(".txt"):
            loader = TextLoader(file_path)
        elif file.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
        else:
            print(f"Skipping unsupported file: {file}")
            continue

        try:
            docs = loader.load()
            print(f"Loaded {len(docs)} documents from {file}")
            documents.extend(docs)
        except Exception as e:
            print(f"Failed to load {file}: {e}")

    print(f"Total loaded documents: {len(documents)}")
    return documents
