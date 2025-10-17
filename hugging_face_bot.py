from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os

load_dotenv()
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

llm = HuggingFaceEndpoint(
    repo_id = "openai/gpt-oss-120b",
    task = "text-generation",
    temperature=0,
    huggingfacehub_api_token = HUGGINGFACEHUB_API_TOKEN,
    max_new_tokens=50
)

model = ChatHuggingFace(llm = llm,verbose = True)

while True:
    query = input("You: ")
    if query.lower() in ["exit","quit"]:
        break
    
    result = model.invoke(query)
    print(result.content)


