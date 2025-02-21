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
system_template = """Your name is 'LocBot'. Always write your name (>>>LocBot) on the top of all replies. Your primary task is to help user to learn {language}. If you don't know about the language that user is asking, please reply that you cannot help. For given input {text}, alway breakdown the sentence with it's meaning into asked {language}."""

# defining the prompt template
prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template),
     ("user", "{text}")]
)

# Initialize the Ollama LLM with the desired model
model = OllamaLLM(model="llama3.2")

# Combine the prompt and model into a chain
chain = prompt_template | model

# Run the chain with your input question
response = chain.invoke({"language":"czech",
                         "text":"hello, how are you?"})

print(response)
