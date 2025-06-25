from langchain_huggingface import HuggingFaceHub
from dotenv import load_dotenv
import os

load_dotenv()

# Initialize the model
llm = HuggingFaceHub(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    model_kwargs={
        "temperature": 0.7,
        "max_length": 256
    },
    huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_TOKEN")
)

result = llm.invoke("What is the capital of India?")
print(result)