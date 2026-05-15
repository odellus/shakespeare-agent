import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaEmbeddings
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_community.vectorstores import FAISS

load_dotenv()

embeddings = OllamaEmbeddings(model='nomic-embed-text-v2-moe:latest')
vs = FAISS.load_local('shakespeare_index', embeddings, allow_dangerous_deserialization=True)

model = ChatOpenAI(
    model='unsloth/Qwen3.6-35B-A3B-GGUF:Q8_0',
    api_key=os.environ['KIMI_API_KEY'],
    base_url=os.environ['KIMI_BASE_URL'],
    streaming=True,
)

@tool(response_format='content_and_artifact')
def retrieve_shakespeare(query: str):
    '''Retrieve passages from Shakespeare.'''
    retrieved_docs = vs.similarity_search(query, k=2)
    serialized = '\n\n'.join(
        (f'Source: {doc.metadata}\nContent: {doc.page_content}')
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

agent = create_agent(model, [retrieve_shakespeare], system_prompt='You are a Shakespeare scholar.')

with open('/tmp/agent_debug_full.txt', 'w') as f:
    for i, step in enumerate(agent.stream(
        {'messages': [{'role': 'user', 'content': 'What does Hamlet say about death?'}]},
        stream_mode='values',
    )):
        msgs = step['messages']
        f.write(f'\n=== Step {i}, {len(msgs)} messages ===\n')
        for j, msg in enumerate(msgs):
            tc = getattr(msg, 'tool_calls', None)
            f.write(f'  [{j}] {type(msg).__name__}: content_len={len(msg.content)} tool_calls={tc}\n')
            f.write(f'      CONTENT: {repr(msg.content)}\n')
