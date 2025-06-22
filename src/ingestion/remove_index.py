from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
import os
from config.settings import FAISS_INDEX_PATH

embedding = OpenAIEmbeddings()

# Load existing index
vs = FAISS.load_local(FAISS_INDEX_PATH, embedding, allow_dangerous_deserialization=True)

filtered_docs = []
for doc in vs.docstore._dict.values():
    if "https://iken.iisc.ac.in/mtech-online/SOI_Aug_2023.pdf" in doc.page_content:  # condition
        continue
    if "courses.iisc.ac.in" in doc.metadata.get("source", ""):
        continue
    filtered_docs.append(doc)

vs_clean = FAISS.from_documents(filtered_docs, embedding)
vs_clean.save_local(FAISS_INDEX_PATH)
print(f"Cleaned:{len(filtered_docs)} chunks.")
