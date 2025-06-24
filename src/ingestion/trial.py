import json
import os
from typing import List, Dict
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from config.settings import MODEL_NAME, TEMPERATURE
 
PROMPT_TEMPLATE_STRING = """
 
Context (extracted from official university sources):
{context}
 
Question from student:
{question}
 
"""
 
PROMPT_TEMPLATE = PromptTemplate(
    template=PROMPT_TEMPLATE_STRING,
    input_variables=["context", "question"]
)
 
def load_question_items(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
 
def save_question_items(data: List[Dict], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
 
def initialize_rag_from_vectorstore(persist_dir: str) -> RetrievalQA:
    embedding_model = OpenAIEmbeddings()
    vector_store = FAISS.load_local(
        persist_dir,
        embeddings=embedding_model,
        index_name="index",  # Use 'faiss' if your files are named faiss.faiss/pkl
        allow_dangerous_deserialization=True
    )
 
    chat_model = ChatOpenAI(
        model_name=MODEL_NAME,
        temperature=TEMPERATURE
    )
 
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
 
    qa_chain = RetrievalQA.from_chain_type(
        llm=chat_model,
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
        memory=memory,
        chain_type_kwargs={"prompt": PROMPT_TEMPLATE}
    )
 
    return qa_chain
 
def run_rag_and_replace_answers(
    input_json_path: str,
    output_json_path: str,
    faiss_dir: str
) -> None:
    qa_chain = initialize_rag_from_vectorstore(faiss_dir)
    data = load_question_items(input_json_path)
 
    for item in data:
        question = item.get("question", "")
        if question:
            response = qa_chain.invoke({"query": question})
            item["answer"] = response["result"]  # replace Gemini answer with RAG output
 
    save_question_items(data, output_json_path)
 
# Example usage
if __name__ == "__main__":
    run_rag_and_replace_answers(
        input_json_path="src/eval_scripts/random_samples.json",
        output_json_path="src/eval_scripts/retrieval_scores_rag_2.json",
        faiss_dir="src/data/faiss_index"
    )
 