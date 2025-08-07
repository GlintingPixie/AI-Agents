from langchain_openai import AzureChatOpenAI
from langchain.agents import initialize_agent
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents import AgentType
from langchain.memory import ConversationSummaryMemory
from langchain.tools import StructuredTool
from pydantic import BaseModel, Field
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

# Input Class for Currency Converter function
class CurrencyInput(BaseModel):
    amount: float = Field(..., description="Amount to convert")
    from_currency: str = Field(..., description="3-letter currency code to convert from (e.g., USD, INR, EUR)")
    to_currency: str = Field(..., description="3-letter currency code to convert to (e.g., USD, INR, EUR)")

# Preprocessing layer for informal phases
SLANG_TO_CURRENCY = {
    "bucks": "USD",
    "quid": "GBP",
    "rupees": "INR",
    "euro": "EUR",
    "yen": "JPY",
    "down under": "AUD",
    "aussie": "AUD",
    "canadian": "CAD",
    "singaporean": "SGD",
    "yuan": "CNY"
}

# Function to convert informal phrases to currency
def normalize_currency_name(name: str) -> str:
    name = name.lower().strip()
    return SLANG_TO_CURRENCY.get(name, name.upper())  # fallback to original

# Currency Converter Function
def currency_converter_pydantic(amount: float, from_currency: str, to_currency: str) -> str:
    from_currency = normalize_currency_name(from_currency).lower()
    to_currency = normalize_currency_name(to_currency).lower()
    
    url = f"https://cdn.jsdelivr.net/npm/@fawazahmed0/currency-api@latest/v1/currencies/{from_currency}.json"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        if to_currency in data[from_currency]:
            rate = data[from_currency][to_currency]
            converted = amount * rate
            return f"{amount} {from_currency.upper()} = {converted:.2f} {to_currency.upper()}"
        else:
            return f"Conversion rate for {to_currency.upper()} not available."
    return "API Error: Could not fetch conversion rate."

# Weather function
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
    func=currency_converter_pydantic,
    name="currency_converter",  # <--- must be a string name
    args_schema=CurrencyInput,
    description="Converts currency using live rates. Provide amount, from_currency, and to_currency. Example: {\"amount\": 100, \"from_currency\": \"USD\", \"to_currency\": \"INR\"}"
)

weather_tool = StructuredTool.from_function(
    func=get_weather,
    name="weather_checker",
    description="Get current weather for a city. Format: Just pass the city name. Example: 'London'."
)

# ----- Tools (Calculator + Wikipedia + Web Search + Currency Converter + Weather) ----------
tools = load_tools(["llm-math", "wikipedia", "serpapi"], llm=llm)
tools += [currency_tool,weather_tool]


# ---------- Smarter Memory (Summary-based) ----------
memory = ConversationSummaryMemory(llm=llm, memory_key="chat_history",return_messages=True)

# 4. Create the agent
agent = initialize_agent(
    tools,
    llm,
    agent= AgentType.OPENAI_FUNCTIONS,
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

