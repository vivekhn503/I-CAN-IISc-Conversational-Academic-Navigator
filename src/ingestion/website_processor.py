
# ingestion/website_processor.py

"""
Fetches and processes a website URL into LangChain Document chunks.

Functions:
- fetch_and_process_website: downloads the page, extracts visible text,
  splits it into overlapping chunks, and returns a list of Document objects.
"""

import requests
from bs4 import BeautifulSoup
from typing import List

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from config.settings import CHUNK_SIZE, CHUNK_OVERLAP


def fetch_and_process_website(url: str) -> List[Document]:
    """
    Fetch the content at `url`, extract and clean the visible text,
    split into chunks of size CHUNK_SIZE with CHUNK_OVERLAP, and
    return as a list of LangChain Document objects with source metadata.

    Args:
        url: The website URL to scrape.

    Returns:
        List[Document]: Chunked page text, each Document has metadata {"source": url}.

    Raises:
        requests.HTTPError: If the HTTP request fails.
        Exception: For any parsing or processing errors.
    """
    # 1) Download the page
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()

    # 2) Parse and clean HTML
    soup = BeautifulSoup(resp.text, "html.parser")
    # Remove unwanted elements
    for tag in soup(["script", "style", "header", "footer", "nav", "form"]):
        tag.decompose()
    # Extract visible text
    text = soup.get_text(separator="\n")
    # Collapse multiple blank lines and strip whitespace
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    cleaned_text = "\n".join(lines)

    # 3) Split into overlapping chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_text(cleaned_text)

    # 4) Wrap chunks as Document objects
    docs: List[Document] = [
        Document(page_content=chunk, metadata={"source": url})
        for chunk in chunks
    ]

    return docs

