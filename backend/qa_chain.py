import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.chains.question_answering import load_qa_chain

# Load environment variables from .env file
load_dotenv()

def get_llm_response(query, docs):
    """
    Uses Groq's LLM to answer a query based on the provided documents.

    Args:
        query (str): The question to be answered.
        docs (list): A list of LangChain Document objects.

    Returns:
        str: The generated answer from the language model.
    """

    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY is not set. Please check your .env file.")

    # Set the environment variable for Groq
    os.environ["GROQ_API_KEY"] = groq_api_key

    # Initialize the language model
    llm = ChatGroq(model_name="llama3-8b-8192", temperature=0)

    # Load the question-answering chain using the "stuff" strategy
    chain = load_qa_chain(llm, chain_type="stuff")

    # Run the chain with the provided documents and question
    return chain.run(input_documents=docs, question=query)
