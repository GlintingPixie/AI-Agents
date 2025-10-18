from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
import os
import streamlit as st
load_dotenv()

# ---- CONFIG ----
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_key=os.getenv("AZURE_OPENAI_API_KEY")
api_version=os.getenv('AZURE_OPENAI_API_VERSION')
deployment_name = os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME')

# Create the model
model = AzureChatOpenAI(
    deployment_name=deployment_name,
    temperature=0,
    api_key=api_key,
    azure_endpoint =azure_endpoint,
    api_version=api_version
)

st.header("Legal Research Assistant")

user_input = st.text_input("Write your query here")

if st.button("Answer"):
    result = model.invoke(user_input)
    st.write(result.content)
