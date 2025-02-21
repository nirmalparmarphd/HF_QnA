from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import getpass
import os
from dotenv import load_dotenv

load_dotenv()

# langsmith setup
os.environ['LANGSMITH_TRACING']=os.getenv('LANGSMITH_TRACING')
os.environ['LANGSMITH_API_KEY']=os.getenv('LANGSMITH_API_KEY')
os.environ['LANGSMITH_ENDPOINT']=os.getenv('LANGSMITH_ENDPOINT')
os.environ['LANGSMITH_PROJECT']=os.getenv('LANGSMITH_PROJECT')

# Define your prompt template
template = """Question: {question}

Answer: Let's think step by step."""

prompt = ChatPromptTemplate.from_template(template)

# Initialize the Ollama LLM with the desired model
model = OllamaLLM(model="llama3.2")

# Combine the prompt and model into a chain
chain = prompt | model

# Run the chain with your input question
response = chain.invoke({"question": "Can you translate to Czech language?"})

print(response)
