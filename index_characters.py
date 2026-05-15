import os
import json
from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

PLAYS_DIR = "plays"
INDEXES_DIR = "character_indexes"


def main():
    with open(os.path.join(PLAYS_DIR, "manifest.json"), "r", encoding="utf-8") as f:
        manifest = json.load(f)

    embeddings = OllamaEmbeddings(model="nomic-embed-text-v2-moe:latest")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        add_start_index=True,
    )

    for slug, info in manifest.items():
        play_dir = os.path.join(PLAYS_DIR, slug)
        if not os.path.exists(play_dir):
            continue

        for speaker_file in Path(play_dir).glob("*.txt"):
            if speaker_file.name == "_play.txt":
                continue

            speaker = speaker_file.stem
            index_dir = os.path.join(INDEXES_DIR, slug, speaker)

            if os.path.exists(index_dir):
                print(f"Skipping {slug}/{speaker} (already indexed)")
                continue

            with open(speaker_file, "r", encoding="utf-8") as f:
                text = f.read()

            if len(text) < 100:
                continue

            chunks = splitter.split_text(text)
            docs = [
                Document(page_content=chunk, metadata={"speaker": speaker, "chunk_idx": i})
                for i, chunk in enumerate(chunks)
            ]

            print(f"Indexing {slug}/{speaker} ({len(docs)} chunks)...")
            store = FAISS.from_documents(docs, embeddings)
            os.makedirs(index_dir, exist_ok=True)
            store.save_local(index_dir)

    print("Done.")


if __name__ == "__main__":
    main()
