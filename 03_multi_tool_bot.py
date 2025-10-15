from langchain_openai import AzureChatOpenAI
from langchain.agents import initialize_agent
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents import AgentType
from langchain.memory import ConversationSummaryMemory
from langchain.tools import StructuredTool
import os
from dotenv import load_dotenv
load_dotenv()

# ---- CONFIG ----
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_key=os.getenv("AZURE_OPENAI_API_KEY")
api_version=os.getenv('AZURE_OPENAI_API_VERSION')
deployment_name = os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME')
serp_api_key  = os.getenv('SERPAPI_API_KEY')
os.environ["SERPAPI_API_KEY"] = str(serp_api_key)

# Create the model
llm = AzureChatOpenAI(
    deployment_name=deployment_name,
    temperature=0,
    api_key=api_key,
    azure_endpoint =azure_endpoint,
    api_version=api_version
)

def currency_converter(query: str) -> str:
    try:
        amount, from_currency, _, to_currency = query.split()
        amount = float(amount)
        rates = {"USD": 1, "INR": 83, "EUR": 0.9}  # Static demo
        if from_currency not in rates or to_currency not in rates:
            return "Unsupported currency"
        converted = amount * rates[to_currency] / rates[from_currency]
        return f"{amount} {from_currency} = {converted:.2f} {to_currency}"
    except:
        return "Error: Use '<amount> <FROM> to <TO>'"

currency_tool = StructuredTool.from_function(
    func=currency_converter,
    name="currency_converter",  # <--- must be a string name
    description="Convert currency using the format '<amount> <from> to <to>'."
)

# ---------- Tools (Calculator + Wikipedia + Web Search + Currency Converter) ----------
tools = load_tools(["llm-math", "wikipedia", "serpapi"], llm=llm)
tools += [currency_tool]


# ---------- Smarter Memory (Summary-based) ----------
memory = ConversationSummaryMemory(llm=llm, memory_key="chat_history",return_messages=True)

# 4. Create the agent
agent = initialize_agent(
    tools,
    llm,
    agent= AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True,
    memory = memory,
    handle_parsing_errors = True,
    allow_dangerous_tools = True
)

# ---------- Conversation Loop ----------
while True:
    query = input("You: ")
    if query.lower() in ["exit", "quit"]:
        break
    response = agent.invoke({"input": query})
    print("Bot:", response["output"])

