import asyncio
from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(
    model="llama3.2",
)

response = embeddings.embed_query("This is a try!")

print(str(response))
