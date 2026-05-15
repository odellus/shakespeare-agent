import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()


def chat_with_character(play_slug: str, speaker: str, query: str):
    embeddings = OllamaEmbeddings(model="nomic-embed-text-v2-moe:latest")
    store = FAISS.load_local(
        f"play_indexes/{play_slug}",
        embeddings,
        allow_dangerous_deserialization=True,
    )

    # Retrieve from the play, then keep only the chosen character's lines
    results = store.similarity_search(query, k=30)
    character_docs = [r for r in results if r.metadata["speaker"] == speaker]

    if not character_docs:
        print(f"No context found for {speaker} on this query.")
        return

    context = "\n\n---\n\n".join(d.page_content for d in character_docs[:4])

    model = ChatOpenAI(
        model="unsloth/Qwen3.6-35B-A3B-GGUF:Q8_0",
        api_key=os.environ["KIMI_API_KEY"],
        base_url=os.environ["KIMI_BASE_URL"],
        streaming=True,
    )

    system_prompt = (
        f"You are {speaker}, a character in a Shakespeare play. "
        "Respond to the user in character, using Elizabethan language "
        "(thee, thou, thy, and other period-appropriate speech). "
        "Ground your response in your own words from the play.\n\n"
        f"Your own words on this matter:\n{context}"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query},
    ]

    print(f"\n=== {speaker} RESPONDS ===\n")
    for chunk in model.stream(messages):
        if chunk.content:
            print(chunk.content, end="", flush=True)
    print("\n")


if __name__ == "__main__":
    chat_with_character(
        play_slug="the_tragedy_of_hamlet_prince_of_denmark",
        speaker="HAMLET",
        query="What do you think of your mother marrying your uncle so quickly?",
    )
