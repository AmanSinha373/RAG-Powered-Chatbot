def retrieve_docs(query, vectorstore, k=3):
    # It will fetch top-k similar chunks based on query similarity
    return vectorstore.similarity_search(query, k=k)
