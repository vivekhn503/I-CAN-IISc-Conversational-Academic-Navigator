"""
rag_pipeline.py

This module initializes a Retrieval-Augmented Generation (RAG) pipeline using LangChain components.
The pipeline retrieves relevant documents from a vector store based on a user question and generates
answers using a chat-based language model.

Components:
- ChatOllama: Chat-based language model interface.
- OllamaEmbeddings: Embedding model for document encoding.
- FAISS: Vector store for efficient similarity search.
- RetrievalQA: Chain that performs retrieval and QA.
- PromptTemplate: Template for formatting the retrieval context and question.
- ConversationBufferMemory: Memory buffer to store chat history.

Functions:
- initialize_rag_pipeline: Sets up the RAG pipeline with specified documents.
"""

from typing import List, Tuple

# Import the chat model interface from the LangChain community package
from langchain_community.chat_models import ChatOllama

# Import the embedding model interface
from langchain_community.embeddings import OllamaEmbeddings

# Import FAISS vector store for document retrieval
from langchain_community.vectorstores import FAISS

# Import the RetrievalQA chain for question answering
from langchain.chains import RetrievalQA

# Import PromptTemplate for defining input templates
from langchain.prompts import PromptTemplate

# Import ConversationBufferMemory to maintain chat history
from langchain.memory import ConversationBufferMemory

# Import model configuration settings
from config.settings import MODEL_NAME, TEMPERATURE

# Define a template string for the QA prompt
PROMPT_TEMPLATE_STRING = """
Context:
{context}

Question:
{question}

Answer concisely based only on the provided context.
"""

# Create a PromptTemplate object that binds the template string to input variables
PROMPT_TEMPLATE = PromptTemplate(
    template=PROMPT_TEMPLATE_STRING,
    input_variables=["context", "question"]
)

def initialize_rag_pipeline(
    documents: List,
) -> Tuple[RetrievalQA, FAISS]:
    """
    Initialize and return the RAG (Retrieval-Augmented Generation) pipeline components.

    Args:
        documents (List): A list of document objects to be indexed in the vector store.

    Returns:
        qa_chain (RetrievalQA): The RetrievalQA chain with embedded memory and custom prompt.
        vector_store (FAISS): The FAISS vector store containing document embeddings.
    """

    # Instantiate the chat-based language model with specified parameters
    chat_model = ChatOllama(
        model=MODEL_NAME,
        temperature=TEMPERATURE
    )

    # Instantiate the embedding model for document encoding
    embedding_model = OllamaEmbeddings(
        model=MODEL_NAME
    )

    # Build a FAISS vector store from the provided documents and embedding model
    vector_store = FAISS.from_documents(
        documents,
        embedding_model
    )

    # Set up a conversation memory buffer to store chat history for context
    conversation_memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    # Create the RetrievalQA chain using the chat model, retriever, and memory
    qa_chain = RetrievalQA.from_chain_type(
        llm=chat_model,                      # Language model interface
        chain_type="stuff",                  # 'stuff' packs all retrieved docs into the prompt
        retriever=vector_store.as_retriever(),  # Retriever for fetching relevant docs
        memory=conversation_memory,             # Memory to keep track of chat context
        chain_type_kwargs={"prompt": PROMPT_TEMPLATE}  # Use  custom PromptTemplate
    )

    # Return the QA chain and vector store for use in application.
    return qa_chain, vector_store
