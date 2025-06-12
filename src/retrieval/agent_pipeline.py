# retrieval/agent_pipeline.py

"""
retrieval/agent_pipeline.py

Defines a LangChain conversational Agent that uses FAISS-based document retrieval as a tool
and ChatOllama/ChatOpenAI as the underlying language model, with conversational memory support.

Functions:
- initialize_agent_pipeline: builds and returns the Agent and its vector store.
"""

from datetime import datetime
import logging
import os
from typing import List, Tuple

from langchain.chat_models import ChatOllama
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import FAISS
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain.docstore.document import Document

# from langchain.chat_models import ChatOllama
# from langchain.embeddings import OllamaEmbeddings


from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings


# Import configuration constants
from config.settings import FAISS_INDEX_PATH, MODEL_NAME, TEMPERATURE 


def load_agent_pipeline() -> Tuple[object, FAISS]:
    """
    Load a persisted FAISS vector store and reconstruct the conversational Agent.

    Returns:
        agent: A LangChain Agent that uses the loaded FAISS index.
        vector_store: The FAISS store containing document embeddings.
    """
    # 1. Load embeddings and FAISS vector store
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.load_local(
        FAISS_INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True  # required due to pickle safety
    )

    # 2. Define retrieval tool
    def retrieve_docs(query: str) -> str:
        hits = vector_store.similarity_search_with_score(query, k=3)
        return "\n\n".join([doc.page_content for doc, _ in hits])

    retrieval_tool = Tool(
        name="faiss_retriever",
        func=retrieve_docs,
        description="Fetch top 3 relevant document chunks for a query."
    )

    # 3. Add a date tool (optional)
    def get_current_date(_query: str) -> str:
        return datetime.now().strftime("%Y-%m-%d")

    date_tool = Tool(
        name="get_current_date",
        func=get_current_date,
        description="Returns the current date in YYYY-MM-DD format."
    )

    # 4. Set up conversation memory
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # 5. Create agent
    llm = ChatOpenAI(model_name=MODEL_NAME, temperature=TEMPERATURE)

    agent = initialize_agent(
        tools=[retrieval_tool, date_tool],
        llm=llm,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True
    )

    return agent, vector_store

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
    #llm = ChatOllama(model=MODEL_NAME, temperature=TEMPERATURE)
    #embeddings = OllamaEmbeddings(model=MODEL_NAME)


    llm = ChatOpenAI(model_name=MODEL_NAME, temperature=TEMPERATURE)
    embeddings = OpenAIEmbeddings()


    # 2. Build FAISS index from supplied documents
    if os.path.exists(FAISS_INDEX_PATH):
        logging.info("📁 Loading existing FAISS index from disk.")
        vector_store = FAISS.load_local(
        FAISS_INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True
        )

        vector_store.add_documents(documents)
        logging.info(f"➕ Added {len(documents)} new documents to existing index.")
    else:
        logging.info("🆕 No FAISS index found. Creating new vector store.")
        vector_store = FAISS.from_documents(documents, embeddings)
        logging.info(f"✅ Created new index with {len(documents)} documents.")

    vector_store.save_local(FAISS_INDEX_PATH)
    logging.info(f"💾 Saved FAISS index to: {FAISS_INDEX_PATH}")


    # 3. Define a retrieval tool that returns concatenated top-3 documents
    def retrieve_docs(query: str) -> str:
        hits = vector_store.similarity_search_with_score(query, k=3)
        return "\n\n".join([doc.page_content for doc, _ in hits])

    retrieval_tool = Tool(
        name="faiss_retriever",
        func=retrieve_docs,
        description="Fetch the top 3 most relevant document chunks for a query."
    )

    #  Date tool 
    def get_current_date(_query: str) -> str:
        """Return today's date in YYYY-MM-DD format."""
        return datetime.now().strftime("%Y-%m-%d")

    date_tool = Tool(
        name="get_current_date",
        func=get_current_date,
        description="Returns the current date in YYYY-MM-DD format."
    )

    # 4. Configure conversational memory to keep chat history
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # 5. Initialize the agent with the retrieval tool and memory
    agent = initialize_agent(
        tools=[retrieval_tool, date_tool],
        llm=llm,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True

    )
    

    return agent, vector_store
