'''
Semantic search engine for given PDF documents
'''

from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

# embedding model
embeddings = OllamaEmbeddings(
    model="llama3.2",
)

# pdf files path
file_path = "../data/main.pdf"

pdf_loader = PyPDFLoader(file_path=file_path)

docs = pdf_loader.load()

# print(f"{docs[0].page_content[:500]}\n")

# setting up text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True
)

# splitting all pages from entire document
all_splits = text_splitter.split_documents(docs)

# setting up vector store
vector_store = Chroma(embedding_function=embeddings, 
                      collection_name='chroma_db',
                      persist_directory='../data/chroma_db')
# ids = vector_store.add_documents(documents=all_splits)
ids = vector_store.add_documents(documents=all_splits)

# search check
# results = vector_store.similarity_search("Close loop pulsating heat pipe working")

