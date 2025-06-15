from datetime import datetime
import logging
import os
from typing import List, Tuple

from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain.docstore.document import Document
from tools.gmail_tool import send_email
from tools.calender_invite import schedule_event

from config.settings import FAISS_INDEX_PATH, MODEL_NAME, TEMPERATURE


# --- Weekly Digest Tool ---
def create_weekly_digest_tool(vector_store) -> Tool:
    def generate_digest(_query: str) -> str:
        logging.info("🧠 [Digest Tool] Generating weekly digest.")
        hits = vector_store.similarity_search("weekly update OR announcement OR deadline", k=10)
        combined = "\n\n".join([doc.page_content for doc in hits])
        prompt = f"""
Summarize the following recent academic updates into weekly digest format.
Group into: 📅 Deadlines, 📣 Announcements, 📝 Course-specific Updates.
Only include updates from the past 7 days.

TEXT:
{combined}
"""
        llm = ChatOpenAI(model_name=MODEL_NAME, temperature=0.3)
        return "📬 (via `weekly_digest_agent`) Weekly Digest:\n\n" + llm.predict(prompt)

    return Tool(
        name="weekly_digest_agent",
        func=generate_digest,
        description="Summarize weekly academic updates like deadlines and announcements."
    )


# --- Personal Planner Tool ---
def create_personal_planner_tool(vector_store) -> Tool:
    def generate_plan(query: str) -> str:
        logging.info("🧠 [Planner Tool] Creating personalized academic plan.")
        hits = vector_store.similarity_search(query, k=6)
        content = "\n\n".join([doc.page_content for doc in hits])
        prompt = f"""
You are an academic planner. Based on the user's query and the provided text, identify any upcoming deadlines, especially related to specific courses like Machine Learning.

Only include deadlines from the current or next week (today: {datetime.now().strftime('%Y-%m-%d')}).

Respond clearly and only if a deadline is found.

---

User Query:
{query}

Retrieved Text:
{content}
"""

        llm = ChatOpenAI(model_name=MODEL_NAME, temperature=0.4)
        return "📘 (via `personal_planner_agent`)\n\n" + llm.predict(prompt)

    return Tool(
        name="personal_planner_agent",
        func=generate_plan,
        description="Help students plan based on their course-specific deadlines, announcements, or progress queries."
    )


# --- Gmail Agent ---
def create_gmail_tool() -> Tool:
    def gmail_func(input: str) -> str:
        # Simple format: "to::subject::body"
        try:
            to, subject, body = input.split("::", 2)
            return send_email(to.strip(), subject.strip(), body.strip())
        except Exception as e:
            return f"Format: to::subject::body | Error: {e}"
    return Tool(
        name="send_gmail_agent",
        func=gmail_func,
        description="Send an email via Gmail. Format: to::subject::body"
    )

# --- Calendar Agent ---

def create_calendar_tool() -> Tool:
    def calendar_func(input: str) -> str:
        # Simple format: "summary::start_time::end_time::description"
        try:
            summary, start_time, end_time, description = input.split("::", 3)
            return schedule_event(summary.strip(), start_time.strip(), end_time.strip(), description.strip())
        except Exception as e:
            return f"Format: summary::start::end::description | Error: {e}"
    return Tool(
        name="schedule_calendar_event",
        func=calendar_func,
        description="Schedule a Google Calendar event. Format: summary::start_time::end_time::description"
    )

# --- Load Persistent Agent ---
def load_agent_pipeline() -> Tuple[object, FAISS]:
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.load_local(
        FAISS_INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    # Define tools
    def retrieve_docs(query: str) -> str:
        hits = vector_store.similarity_search_with_score(query, k=3)
        return "\n\n".join([doc.page_content for doc, _ in hits])

    retrieval_tool = Tool(
        name="faiss_retriever",
        func=retrieve_docs,
        description="Fetch top 3 relevant document chunks for a query."
    )

    date_tool = Tool(
        name="get_current_date",
        func=lambda _: datetime.now().strftime("%Y-%m-%d"),
        description="Returns the current date in YYYY-MM-DD format."
    )

    weekly_digest_tool = create_weekly_digest_tool(vector_store)
    personal_planner_tool = create_personal_planner_tool(vector_store)

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    llm = ChatOpenAI(model_name=MODEL_NAME, temperature=TEMPERATURE)

    agent = initialize_agent(
        tools=[retrieval_tool, date_tool, weekly_digest_tool, personal_planner_tool],
        llm=llm,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True
    )

    return agent, vector_store


# --- Initialize Agent with New Documents ---
def initialize_agent_pipeline(documents: List[Document]) -> Tuple[object, FAISS]:
    llm = ChatOpenAI(model_name=MODEL_NAME, temperature=TEMPERATURE)
    embeddings = OpenAIEmbeddings()

    if os.path.exists(FAISS_INDEX_PATH):
        logging.info("📁 Loading existing FAISS index from disk.")
        vector_store = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        vector_store.add_documents(documents)
        logging.info(f"➕ Added {len(documents)} new documents to existing index.")
    else:
        logging.info("🆕 No FAISS index found. Creating new vector store.")
        vector_store = FAISS.from_documents(documents, embeddings)
        logging.info(f"✅ Created new index with {len(documents)} documents.")

    vector_store.save_local(FAISS_INDEX_PATH)
    logging.info(f"💾 Saved FAISS index to: {FAISS_INDEX_PATH}")

    # Tools
    def retrieve_docs(query: str) -> str:
        hits = vector_store.similarity_search_with_score(query, k=3)
        return "\n\n".join([doc.page_content for doc, _ in hits])

    retrieval_tool = Tool(
        name="faiss_retriever",
        func=retrieve_docs,
        description="Fetch top 3 relevant document chunks for a query."
    )

    date_tool = Tool(
        name="get_current_date",
        func=lambda _: datetime.now().strftime("%Y-%m-%d"),
        description="Returns the current date."
    )

    gmail_tool = create_gmail_tool()
    calendar_tool = create_calendar_tool()

    weekly_digest_tool = create_weekly_digest_tool(vector_store)
    personal_planner_tool = create_personal_planner_tool(vector_store)

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    agent = initialize_agent(
        tools=[retrieval_tool, date_tool, weekly_digest_tool, personal_planner_tool, gmail_tool, calendar_tool],
        llm=llm,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True
    )

    return agent, vector_store
