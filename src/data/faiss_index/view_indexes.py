from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from config.settings import FAISS_INDEX_PATH
import os

def view_chunks():
    print("Loading FAISS vector store from disk...")
    if not os.path.exists(FAISS_INDEX_PATH):
        print("Error: ", FAISS_INDEX_PATH)
        return
    vector_store = FAISS.load_local(FAISS_INDEX_PATH, OpenAIEmbeddings(), allow_dangerous_deserialization=True)
    
    print(f"Total chunks: {len(vector_store.docstore._dict)}\n")

    for i, (doc_id, doc) in enumerate(vector_store.docstore._dict.items()):
        print(f"Chunk {i+1} [ID: {doc_id}]")
        print("Source:", doc.metadata.get("source", "Unknown"))
        print("Content Preview:")
        print(doc.page_content[:500])
        print("-" * 80)

if __name__ == "__main__":
    view_chunks()
