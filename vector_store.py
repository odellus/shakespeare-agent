from langchain_ollama import OllamaEmbeddings
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from uuid import uuid4
from langchain_core.documents import Document


def build_vector_store() -> FAISS:
    embeddings = OllamaEmbeddings(model="nomic-embed-text-v2-moe:latest")
    index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))
    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )
    docs = [
        Document(page_content="I had chocolate chip pancakes and scrambled eggs for breakfast this morning.", metadata={"source": "tweet"}),
        Document(page_content="The weather forecast for tomorrow is cloudy and overcast, with a high of 62 degrees.", metadata={"source": "news"}),
        Document(page_content="Building an exciting new project with LangChain - come check it out!", metadata={"source": "tweet"}),
        Document(page_content="Robbers broke into the city bank and stole $1 million in cash.", metadata={"source": "news"}),
        Document(page_content="Wow! That was an amazing movie. I can't wait to see it again.", metadata={"source": "tweet"}),
        Document(page_content="Is the new iPhone worth the price? Read this review to find out.", metadata={"source": "website"}),
        Document(page_content="The top 10 soccer players in the world right now.", metadata={"source": "website"}),
        Document(page_content="LangGraph is the best framework for building stateful, agentic applications!", metadata={"source": "tweet"}),
        Document(page_content="The stock market is down 500 points today due to fears of a recession.", metadata={"source": "news"}),
        Document(page_content="I have a bad feeling I am going to get deleted :(", metadata={"source": "tweet"}),
    ]
    uuids = [str(uuid4()) for _ in range(len(docs))]
    vector_store.add_documents(documents=docs, ids=uuids)
    return vector_store
