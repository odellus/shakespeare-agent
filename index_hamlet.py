import os
import re
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

HAMLET_FILE = "hamlet.txt"
INDEX_DIR = "hamlet_index"


def extract_hamlet_speeches(path: str) -> list[str]:
    """Pull out every speech spoken by HAMLET from the play text."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().replace("\r\n", "\n")

    # A speaker block: ALL CAPS NAME + period at start of line,
    # followed by everything until the next speaker block or EOF.
    blocks = re.findall(
        r"^([A-Z][A-Z\s]+)\.\n(.*?)(?=\n[A-Z][A-Z\s]+\.\n|\Z)",
        text,
        re.DOTALL | re.MULTILINE,
    )

    speeches = []
    for speaker, speech in blocks:
        if speaker.strip() == "HAMLET":
            cleaned = speech.strip()
            if cleaned:
                speeches.append(cleaned)

    return speeches


def main():
    print(f"Loading {HAMLET_FILE}...")
    speeches = extract_hamlet_speeches(HAMLET_FILE)
    print(f"  → {len(speeches)} speeches by Hamlet")

    # Join with clear separators so the splitter respects speech boundaries when possible
    raw_text = "\n\n---\n\n".join(f"HAMLET. {s}" for s in speeches)
    print(f"  → {len(raw_text):,} total characters")

    print("Chunking...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        add_start_index=True,
    )
    chunks = splitter.split_text(raw_text)
    print(f"  → {len(chunks)} chunks")

    documents = [
        Document(page_content=chunk, metadata={"play": "Hamlet", "speaker": "Hamlet", "chunk_idx": i})
        for i, chunk in enumerate(chunks)
    ]

    print("Embedding & indexing...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text-v2-moe:latest")
    vector_store = FAISS.from_documents(documents, embeddings)

    print(f"Saving to ./{INDEX_DIR} ...")
    vector_store.save_local(INDEX_DIR)
    print("Done.")


if __name__ == "__main__":
    main()
