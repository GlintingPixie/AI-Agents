from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

chat_promp_template = ChatPromptTemplate(
    [
    ("system","You are a helpful {domain} expert"),
    ("human","Explain in simple terms, {topic}")
    ],
    input_variables=["domain","topic"]
)

prompt = chat_promp_template.invoke(
    {
    "domain":"cricket",
    "topic":"What are the various types of bowlers?"
    }
)

print(prompt)

