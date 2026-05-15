import os
import urllib.request
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

GUTENBERG_URL = "https://www.gutenberg.org/ebooks/100.txt.utf-8"
LOCAL_FILE = "shakespeare.txt"


def download_if_needed(url: str, path: str) -> None:
    if os.path.exists(path):
        print(f"Found existing file: {path}")
        return
    print(f"Downloading {url} ...")
    urllib.request.urlretrieve(url, path)
    print(f"Saved to {path}")


def main():
    # 1. Load: download the raw text from Project Gutenberg if we don't have it.
    download_if_needed(GUTENBERG_URL, LOCAL_FILE)

    print("Loading Shakespeare text...")
    loader = TextLoader(LOCAL_FILE, encoding="utf-8")
    docs = loader.load()
    print(f"Loaded {len(docs)} document(s), {len(docs[0].page_content)} chars")

    # 2. Split: break the giant text into overlapping chunks so we can embed
    #    and retrieve specific scenes / soliloquies instead of the whole book.
    print("Splitting into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200,
        add_start_index=True,
    )
    all_splits = text_splitter.split_documents(docs)
    print(f"Split into {len(all_splits)} chunks")

    # 3. Embed + 4. Store: turn each chunk into a vector and build a FAISS index.
    print("Building embeddings and vector store...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text-v2-moe:latest")
    vector_store = FAISS.from_documents(all_splits, embeddings)

    # 5. Persist: save the index + docstore to disk so we don't re-embed every time.
    print("Saving vector store to disk...")
    vector_store.save_local("shakespeare_index")
    print("Done. Saved to ./shakespeare_index")

if __name__ == "__main__":
    main()
