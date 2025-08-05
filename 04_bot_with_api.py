from langchain_openai import AzureChatOpenAI
from langchain.agents import initialize_agent
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents import AgentType
from langchain.memory import ConversationSummaryMemory
from langchain.tools import StructuredTool
import os
import requests
from dotenv import load_dotenv
load_dotenv()

# ---- CONFIG ----
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_key=os.getenv("AZURE_OPENAI_API_KEY")
api_version=os.getenv('AZURE_OPENAI_API_VERSION')
deployment_name = os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME')
serp_api_key  = os.getenv('SERP_API_KEY')
os.environ["SERPAPI_API_KEY"] = serp_api_key

# Create the model
llm = AzureChatOpenAI(
    deployment_name=deployment_name,
    temperature=0,
    api_key=api_key,
    azure_endpoint =azure_endpoint,
    api_version=api_version
)

def currency_converter(query: str) -> str:
    """
    Converts currency using live exchange rates.
    Format: '<amount> <FROM> to <TO>'
    Example: '100 USD to INR'
    """
    try:
        amount, from_currency, _, to_currency = query.strip().split()
        amount = float(amount)
        from_currency, to_currency = from_currency.lower(), to_currency.lower()
        url = f"https://cdn.jsdelivr.net/npm/@fawazahmed0/currency-api@latest/v1/currencies/{from_currency}.json"
        response = requests.get(url)
        if response.status_code != 200:
            return "Error: Unable to fetch currency rates."
        data = response.json()
        rate = data[from_currency].get(to_currency)
        if rate:
            return f"{amount} {from_currency.upper()} = {amount * rate:.2f} {to_currency.upper()}"
        return "Error: Unsupported currency pair."
    except Exception as e:
        return "Error: Use '<amount> <FROM> to <TO>'"
def get_weather(city: str) -> str:
    """
    Gets current weather for a city.
    Format: Just pass the city name. Example: 'London'
    """
    try:
        url = f"https://wttr.in/{city}?format=3"
        response = requests.get(url)
        return response.text if response.status_code == 200 else "Error fetching weather."
    except:
        return "Error: Unable to get weather."

currency_tool = StructuredTool.from_function(
    func=currency_converter,
    name="currency_converter",  # <--- must be a string name
    description="Convert currency. Format: '<amount> <FROM> to <TO>'. Example: '100 USD to INR'."
)
weather_tool = StructuredTool.from_function(
    func=get_weather,
    name="weather_checker",
    description="Get current weather for a city. Format: Just pass the city name. Example: 'London'."
)

# ---------- Tools (Calculator + Wikipedia + Web Search + Currency Converter) ----------
tools = load_tools(["llm-math", "wikipedia", "serpapi"], llm=llm)
tools += [currency_tool,weather_tool]


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

