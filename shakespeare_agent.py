import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaEmbeddings
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from langchain_community.vectorstores import FAISS

load_dotenv()


def load_shakespeare_store():
    embeddings = OllamaEmbeddings(model="nomic-embed-text-v2-moe:latest")
    return FAISS.load_local(
        "shakespeare_index",
        embeddings,
        allow_dangerous_deserialization=True,
    )


def run_tool_agent(model, vector_store):
    @tool(response_format="content_and_artifact")
    def retrieve_shakespeare(query: str):
        """Retrieve passages from Shakespeare's works to answer questions."""
        retrieved_docs = vector_store.similarity_search(query, k=4)
        serialized = "\n\n".join(
            (f"Source: {doc.metadata}\nContent: {doc.page_content}")
            for doc in retrieved_docs
        )
        return serialized, retrieved_docs

    tools = [retrieve_shakespeare]
    prompt = (
        "You are a Shakespeare scholar with access to the complete works of William Shakespeare. "
        "Use the retrieval tool to find relevant passages and answer the user's question accurately. "
        "Quote specific lines when possible. If the context does not contain the answer, say so."
    )
    agent = create_agent(model, tools, system_prompt=prompt)

    print("\n=== Shakespeare Scholar (Tool Agent) ===")
    query = "What does Hamlet say about death and the afterlife?"
    for step in agent.stream(
        {"messages": [{"role": "user", "content": query}]},
        stream_mode="values",
    ):
        step["messages"][-1].pretty_print()


def run_character_agent(model, vector_store):
    @dynamic_prompt
    def prompt_as_character(request: ModelRequest) -> str:
        last_query = request.state["messages"][-1].text
        retrieved_docs = vector_store.similarity_search(last_query, k=3)
        docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)

        return (
            "You are a character inside one of Shakespeare's plays. "
            "You have just witnessed the events described in the retrieved context. "
            "Respond to the user's question in character, as if you are living through the scene. "
            "Speak in a dramatic, theatrical style fitting the Elizabethan era. "
            "Use thee, thou, thy, and other period-appropriate language. "
            "Stay true to the events in the context below, but react to them as a character would.\n\n"
            f"Context of the scene:\n{docs_content}"
        )

    agent = create_agent(model, tools=[], middleware=[prompt_as_character])

    print("\n=== Shakespeare Character (Middleware Agent) ===")
    query = "What do you think of the king's murder?"
    for step in agent.stream(
        {"messages": [{"role": "user", "content": query}]},
        stream_mode="values",
    ):
        step["messages"][-1].pretty_print()


def main():
    print("Loading Shakespeare vector store from disk...")
    vector_store = load_shakespeare_store()
    print("Loaded.")

    model = ChatOpenAI(
        model="unsloth/Qwen3.6-35B-A3B-GGUF:Q8_0",
        api_key=os.environ["KIMI_API_KEY"],
        base_url=os.environ["KIMI_BASE_URL"],
    )

    run_tool_agent(model, vector_store)
    run_character_agent(model, vector_store)


if __name__ == "__main__":
    main()
