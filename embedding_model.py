from langchain_openai import AzureOpenAIEmbeddings
from dotenv import load_dotenv
import os
load_dotenv()

#CONFIG
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_key=os.getenv("AZURE_OPENAI_API_KEY")
api_version=os.getenv('AZURE_OPENAI_API_VERSION')
deployment_name = os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME')
embedding_deployment= os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")

embedding = AzureOpenAIEmbeddings(
    azure_endpoint=azure_endpoint,
    api_key=api_key,
    openai_api_version=api_version,
    model = embedding_deployment,
)

result = embedding.embed_query("Delhi is the capital of India")

print(str(result))