# retrieval/agent_pipeline.py

"""
retrieval/agent_pipeline.py

Defines a LangChain conversational Agent that uses FAISS-based document retrieval as a tool
and ChatOllama as the underlying language model, with conversational memory support.

Functions:
- initialize_agent_pipeline: builds and returns the Agent and its vector store.
"""

from typing import List, Tuple

from langchain.chat_models import ChatOllama
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import FAISS
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain.docstore.document import Document

# Import configuration constants
from config.settings import MODEL_NAME, TEMPERATURE


def initialize_agent_pipeline(
    documents: List[Document]
) -> Tuple[object, FAISS]:  # Agent is returned as a generic object
    """
    Instantiate a conversational Agent with FAISS retrieval and chat memory.

    Args:
        documents (List[Document]): List of text chunks to index in FAISS.

    Returns:
        agent: A LangChain Agent that will decide when to call retrieval.
        vector_store (FAISS): The FAISS store containing document embeddings.
    """
    # 1. Set up the language model and embeddings
    llm = ChatOllama(model=MODEL_NAME, temperature=TEMPERATURE)
    embeddings = OllamaEmbeddings(model=MODEL_NAME)

    # 2. Build FAISS index from supplied documents
    vector_store = FAISS.from_documents(documents, embeddings)

    # 3. Define a retrieval tool that returns concatenated top-3 documents
    def retrieve_docs(query: str) -> str:
        hits = vector_store.similarity_search_with_score(query, k=3)
        return "\n\n".join([doc.page_content for doc, _ in hits])

    retrieval_tool = Tool(
        name="faiss_retriever",
        func=retrieve_docs,
        description="Fetch the top 3 most relevant document chunks for a query."
    )

    # 4. Configure conversational memory to keep chat history
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # 5. Initialize the agent with the retrieval tool and memory
    agent = initialize_agent(
        tools=[retrieval_tool],
        llm=llm,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True

    )
    

    return agent, vector_store
