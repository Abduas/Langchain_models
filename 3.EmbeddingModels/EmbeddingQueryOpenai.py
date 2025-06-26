from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=32)

result = embedding.embed_query("Delhi is the capital of India")

print(f"Embedding length: {len(result)}")
print(f"First 5 values: {result[:5]}")
print(f"All values: {result}")