import os
import json
from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

PLAYS_DIR = "plays"
INDEXES_DIR = "play_indexes"


def build_play_index(play_dir: str, embeddings) -> FAISS | None:
    """Read all speaker files in a play dir, chunk them, and build a FAISS index."""
    docs = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        add_start_index=True,
    )

    for speaker_file in Path(play_dir).glob("*.txt"):
        if speaker_file.name == "_play.txt":
            continue
        speaker = speaker_file.stem
        with open(speaker_file, "r", encoding="utf-8") as f:
            text = f.read()
        if len(text) < 100:
            continue  # skip tiny bit-part actors

        chunks = splitter.split_text(text)
        for i, chunk in enumerate(chunks):
            docs.append(Document(
                page_content=chunk,
                metadata={"speaker": speaker, "chunk_idx": i},
            ))

    if not docs:
        return None

    return FAISS.from_documents(docs, embeddings)


def main():
    with open(os.path.join(PLAYS_DIR, "manifest.json"), "r", encoding="utf-8") as f:
        manifest = json.load(f)

    embeddings = OllamaEmbeddings(model="nomic-embed-text-v2-moe:latest")
    os.makedirs(INDEXES_DIR, exist_ok=True)

    for slug, info in manifest.items():
        play_dir = os.path.join(PLAYS_DIR, slug)
        index_dir = os.path.join(INDEXES_DIR, slug)

        if os.path.exists(index_dir):
            print(f"Skipping {slug} (already indexed)")
            continue

        print(f"Indexing {slug} ...")
        store = build_play_index(play_dir, embeddings)
        if store:
            store.save_local(index_dir)
            print(f"  → saved {index_dir}")
        else:
            print(f"  → no documents")

    print("Done.")


if __name__ == "__main__":
    main()
