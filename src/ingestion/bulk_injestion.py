from ingestion.website_processor import fetch_and_process_website
from retrieval.agent_pipeline import initialize_agent_pipeline, load_agent_pipeline
import os

import pandas as pd
from typing import List, Union
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config.settings import CHUNK_SIZE, CHUNK_OVERLAP
import fitz  


def fetch_and_process_full_csv(file_path: Union[str, object]) -> List[Document]:
    """
    Reads a CSV file and processes all columns into chunked LangChain Documents.
    """
    df = pd.read_csv(file_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    docs: List[Document] = []

    for i, row in df.iterrows():
        # Convert entire row to a readable string
        row_text = "\n".join([f"{col}: {row[col]}" for col in df.columns])
        chunks = splitter.split_text(row_text)
        chunks = [chunk for chunk in chunks if chunk.strip()]

        for chunk in chunks:
            docs.append(Document(
                page_content=chunk,
                metadata={
                    "row_index": i,
                    "source": "CSV",
                }
            ))
    if "http" in str(row.get("Website", "")):
        try:
            web_docs = fetch_and_process_website(row["Website"])
            for d in web_docs:
                d.metadata.update({"row_index": i, "source": row["Website"]})
            docs.extend(web_docs)
        except Exception as e:
            print(f"⚠️ Failed to fetch website {row['Website']}: {e}")


    return docs

def extract_text_from_pdf(file_path: str) -> List[Document]:
    """Extracts text from PDF file and returns a list of chunked Documents."""
    doc = fitz.open(file_path)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    documents = []
    for i, page in enumerate(doc):
        text = page.get_text().strip()
        if text:
            chunks = splitter.split_text(text)
            for chunk in chunks:
                documents.append(Document(page_content=chunk, metadata={"page_number": i, "source": "PDF"}))

    return documents


def ingest_file(path: str) -> List[Document]:
    """Dispatch to the correct ingestion function based on file type."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return fetch_and_process_full_csv(path)
    elif ext == ".pdf":
        return extract_text_from_pdf(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


# --- CONFIG ---
file_paths = [
    "/Users/m0j0a8a/Github/personal/I-CAN-IISc-Conversational-Academic-Navigator/src/data/faiss_index/Faculty Mentor list for Projects.csv",
]

all_docs: List[Document] = []
for path in file_paths:
    try:
        docs = ingest_file(path)
        print(f"{path} → {len(docs)} chunks")
        all_docs.extend(docs)
    except Exception as e:
        print(f"Error processing {path}: {e}")

if not all_docs:
    print("No documents to index.")
else:
    texts = [doc.page_content for doc in all_docs]
    metadatas = [doc.metadata for doc in all_docs]

    if os.path.exists("data/faiss_index"):
        agent, vs = load_agent_pipeline()
        vs.add_texts(texts, metadatas=metadatas)
        vs.save_local("data/faiss_index")
        print(f"Added {len(texts)} chunks")
    else:
        agent, vs = initialize_agent_pipeline(all_docs)
        print(f"Created new vector: {len(all_docs)} chunks.")

