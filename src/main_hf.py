'''
Q&A chatbot for given document
'''

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_huggingface.llms import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from huggingface_hub import login
from dotenv import load_dotenv
import os
import time

start = time.time()

load_dotenv()

# os.environ['HF_HOME'] = '/Users/nirmal/Documents/np_dsci_codes/HF_QnA/model_cache'

hf_token = os.getenv('HUGGINGFACEHUB_ACCESS_TOKEN')

if hf_token:
    login(token=hf_token)
    print("Logged in Successfully!")
else:
    print("HF token not found, Check .env file.")

llm = HuggingFaceEndpoint(
    repo_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    task='text-generation'
)

# llm = HuggingFacePipeline.from_model_id(
#     model_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
#     task='text-generation',
#     # device=-1,
#     device_map='auto',
#     pipeline_kwargs=dict(
#         temperature=0.5,
#         max_new_tokens=100
#     )
# )

model = ChatHuggingFace(llm=llm)

template = ''' Question: {question}

Answer: Give answer in five sentence
'''

prompt = PromptTemplate.from_template(template=template)

question = 'What is Machine Learning?'

chain = prompt | llm

print(chain.invoke(
    {'question': question}
))
