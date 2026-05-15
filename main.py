from vector_store import build_vector_store
from langchain_core.documents import Document


def main():
    print("Hello from faiss-langchain!")
    vector_store = build_vector_store()

    results = vector_store.similarity_search(
        "LangChain provides abstractions to make working with LLMs easy",
        k=2,
        filter={"source": "tweet"},
    )
    for res in results:
        print(f"* {res.page_content} [{res.metadata}]")

    print("Advanced filtering")
    results = vector_store.similarity_search(
        "LangChain provides abstractions to make working with LLMs easy",
        k=2,
        filter={"source": {"$eq": "tweet"}},
    )
    for res in results:
        print(f"* {res.page_content} [{res.metadata}]")

    print("With scores!")
    results = vector_store.similarity_search_with_score(
        "Will it be hot tomorrow?", k=1, filter={"source": "news"}
    )
    for res, score in results:
        print(f"* [SIM={score:3f}] {res.page_content} [{res.metadata}]")

    print("As a retriever")
    retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 1})
    print(retriever.invoke("Stealing from the bank is a crime", filter={"source": "news"}))


if __name__ == "__main__":
    main()
