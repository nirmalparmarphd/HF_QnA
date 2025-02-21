from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

# Define your prompt template
template = """Question: {question}

Answer: Let's think step by step."""

prompt = ChatPromptTemplate.from_template(template)

# Initialize the Ollama LLM with the desired model
model = OllamaLLM(model="llama3.2")

# Combine the prompt and model into a chain
chain = prompt | model

# Run the chain with your input question
response = chain.invoke({"question": "Why sky is blue?"})

print(response)
