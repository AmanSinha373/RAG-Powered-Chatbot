from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import List

def chunk_documents( documents: List[Document], chunk_size= 500, chunk_overlap=100):
    # Spiltts docuemnts into overlapping chunks to preserve context between them
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size, # number of characters per chunk
        chunk_overlap = chunk_overlap # overlap helps retain context continuity

    )
    return splitter.split_documents(documents)