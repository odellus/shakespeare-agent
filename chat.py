import os
import json
from dotenv import load_dotenv
import gradio as gr
from gradio import ChatMessage
from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaEmbeddings
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from langchain_community.vectorstores import FAISS
from langchain_core.messages import AIMessageChunk, ToolMessage

load_dotenv()

PLAYS_DIR = "plays"
INDEXES_DIR = "character_indexes"
MANIFEST_PATH = os.path.join(PLAYS_DIR, "manifest.json")


def load_manifest():
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_shakespeare_store():
    embeddings = OllamaEmbeddings(model="nomic-embed-text-v2-moe:latest")
    return FAISS.load_local(
        "shakespeare_index",
        embeddings,
        allow_dangerous_deserialization=True,
    )


def build_scholar_agent(vector_store):
    model = ChatOpenAI(
        model="unsloth/Qwen3.6-35B-A3B-GGUF:Q8_0",
        api_key=os.environ["KIMI_API_KEY"],
        base_url=os.environ["KIMI_BASE_URL"],
        streaming=True,
    )

    @tool(response_format="content_and_artifact")
    def retrieve_shakespeare(query: str):
        """Retrieve passages from Shakespeare's works to answer questions."""
        retrieved_docs = vector_store.similarity_search(query, k=4)
        serialized = "\n\n".join(
            (f"Source: {doc.metadata}\nContent: {doc.page_content}")
            for doc in retrieved_docs
        )
        return serialized, retrieved_docs

    scholar_prompt = (
        "You are a Shakespeare scholar with access to the complete works of William Shakespeare. "
        "Use the retrieval tool to find relevant passages and answer the user's question accurately. "
        "Quote specific lines when possible. If the context does not contain the answer, say so."
    )
    return create_agent(model, [retrieve_shakespeare], system_prompt=scholar_prompt)


def get_character_choices(play_slug: str) -> list[str]:
    """Return sorted list of characters that have a built index."""
    index_dir = os.path.join(INDEXES_DIR, play_slug)
    if not os.path.exists(index_dir):
        return []
    return sorted(d for d in os.listdir(index_dir) if os.path.isdir(os.path.join(index_dir, d)))


def load_character_store(play_slug: str, speaker: str):
    embeddings = OllamaEmbeddings(model="nomic-embed-text-v2-moe:latest")
    return FAISS.load_local(
        os.path.join(INDEXES_DIR, play_slug, speaker),
        embeddings,
        allow_dangerous_deserialization=True,
    )


async def stream_scholar(agent, message, history):
    history = list(history)
    history.append(ChatMessage(role="user", content=message))
    yield history

    assistant_idx = None
    tool_pending = False

    async for chunk, _meta in agent.astream(
        {"messages": [{"role": "user", "content": message}]},
        stream_mode="messages",
    ):
        if isinstance(chunk, AIMessageChunk):
            if chunk.tool_call_chunks and not tool_pending:
                tool_pending = True
                history.append(ChatMessage(
                    role="assistant",
                    content="",
                    metadata={"title": "🔍 retrieve_shakespeare", "status": "pending"},
                ))
                yield history

            if chunk.content:
                if assistant_idx is None:
                    history.append(ChatMessage(role="assistant", content=""))
                    assistant_idx = len(history) - 1
                old = history[assistant_idx]
                history[assistant_idx] = ChatMessage(
                    role="assistant",
                    content=old.content + chunk.content,
                    metadata=old.metadata,
                )
                yield history

        elif isinstance(chunk, ToolMessage):
            tool_pending = False
            assistant_idx = None
            history.append(ChatMessage(
                role="assistant",
                content=chunk.content,
                metadata={"title": "📚 Retrieved Context", "status": "done"},
            ))
            yield history


async def stream_character(play_slug: str, speaker: str, message: str, history: list):
    history = list(history)
    history.append(ChatMessage(role="user", content=message))
    yield history

    store = load_character_store(play_slug, speaker)
    character_docs = store.similarity_search(message, k=4)

    if not character_docs:
        history.append(ChatMessage(
            role="assistant",
            content=f"Alas, {speaker} hath no words upon this matter in the play.",
        ))
        yield history
        return

    context = "\n\n---\n\n".join(d.page_content for d in character_docs)

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
        {"role": "user", "content": message},
    ]

    history.append(ChatMessage(role="assistant", content=""))
    assistant_idx = len(history) - 1

    async for chunk in model.astream(messages):
        if chunk.content:
            old = history[assistant_idx]
            history[assistant_idx] = ChatMessage(
                role="assistant",
                content=old.content + chunk.content,
            )
            yield history


def main():
    manifest = load_manifest()
    # Only show plays that have at least one character index built
    play_slugs = [
        s for s in manifest
        if os.path.exists(os.path.join(INDEXES_DIR, s))
        and any(os.path.isdir(os.path.join(INDEXES_DIR, s, d)) for d in os.listdir(os.path.join(INDEXES_DIR, s)) if os.path.isdir(os.path.join(INDEXES_DIR, s, d)))
    ]
    play_labels = {s: manifest[s]["title"] for s in play_slugs}

    print("Loading Shakespeare index...")
    vector_store = load_shakespeare_store()
    scholar_agent = build_scholar_agent(vector_store)
    print("Scholar agent ready.")

    async def respond(message, history, mode, play, character):
        history = history or []
        if "Scholar" in mode:
            async for updated in stream_scholar(scholar_agent, message, history):
                yield updated
        else:
            slug = [s for s, t in play_labels.items() if t == play][0]
            async for updated in stream_character(slug, character, message, history):
                yield updated

    def update_characters(play):
        if not play:
            return gr.Dropdown(choices=[], value=None)
        slug = [s for s, t in play_labels.items() if t == play][0]
        chars = get_character_choices(slug)
        return gr.Dropdown(choices=chars, value=chars[0] if chars else None)

    with gr.Blocks() as demo:
        gr.Markdown("# 🎭 Shakespeare Chat")
        gr.Markdown("Ask the **Scholar** for facts, or speak with any **character** from any play.")

        mode = gr.Dropdown(
            ["📚 Scholar", "🎭 Character"],
            value="📚 Scholar",
            label="Mode",
        )

        play_dropdown = gr.Dropdown(
            choices=list(play_labels.values()),
            value="THE TRAGEDY OF HAMLET, PRINCE OF DENMARK",
            label="Play",
            visible=False,
        )

        character_dropdown = gr.Dropdown(
            choices=[],
            value=None,
            label="Character",
            visible=False,
            allow_custom_value=True,
        )

        chatbot = gr.Chatbot(
            label="Chat",
            avatar_images=(None, "https://em-content.zobj.net/source/twitter/53/robot-face_1f916.png"),
            height=500,
        )

        with gr.Row():
            input_box = gr.Textbox(
                placeholder="Ask about Hamlet, Macbeth, love, death, treason...",
                label="Your Question",
                lines=1,
                scale=4,
            )
            submit_btn = gr.Button("Send", scale=1)

        def toggle_visibility(mode):
            is_char = "Character" in mode
            return (
                gr.Dropdown(visible=is_char),
                gr.Dropdown(visible=is_char),
            )

        mode.change(
            toggle_visibility,
            inputs=[mode],
            outputs=[play_dropdown, character_dropdown],
        )

        play_dropdown.change(
            update_characters,
            inputs=[play_dropdown],
            outputs=[character_dropdown],
        )

        # Initialize character list for default play
        demo.load(
            update_characters,
            inputs=[play_dropdown],
            outputs=[character_dropdown],
        )

        submit_btn.click(
            respond,
            inputs=[input_box, chatbot, mode, play_dropdown, character_dropdown],
            outputs=[chatbot],
        )
        input_box.submit(
            respond,
            inputs=[input_box, chatbot, mode, play_dropdown, character_dropdown],
            outputs=[chatbot],
        )

        gr.Examples(
            examples=[
                ["What does Hamlet say about death and the afterlife?", "📚 Scholar", "THE TRAGEDY OF HAMLET, PRINCE OF DENMARK", "HAMLET"],
                ["What do you think of the king's murder?", "🎭 Character", "THE TRAGEDY OF HAMLET, PRINCE OF DENMARK", "HAMLET"],
                ["Tell me about Lady Macbeth's ambition.", "📚 Scholar", "THE TRAGEDY OF HAMLET, PRINCE OF DENMARK", "HAMLET"],
                ["How dost thou feel about thy forbidden love?", "🎭 Character", "THE TRAGEDY OF ROMEO AND JULIET", "ROMEO"],
            ],
            inputs=[input_box, mode, play_dropdown, character_dropdown],
        )

    demo.launch(server_name="0.0.0.0", server_port=7860, theme=gr.themes.Soft())


if __name__ == "__main__":
    main()
