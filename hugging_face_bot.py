from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os

load_dotenv()
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

llm = HuggingFaceEndpoint(
    repo_id = "openai/gpt-oss-120b",
    task = "text-generation",
    temperature=0,
    huggingfacehub_api_token = HUGGINGFACEHUB_API_TOKEN,
    max_new_tokens=50
)

model = ChatHuggingFace(llm = llm,verbose = True)

result = model.invoke("What are the various wars fought between India and Pakistan?")
print(result.content)
