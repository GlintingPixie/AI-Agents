from langchain_openai import AzureChatOpenAI
from langchain.agents import initialize_agent
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents import AgentType
from langchain.memory import ConversationBufferMemory
from langchain.tools import StructuredTool, Tool
from langchain_core.messages import SystemMessage
from pydantic import BaseModel, Field, field_validator
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
    api_version=api_version,
    model_kwargs = {
        "messages": [{"role": "system", "content": "You are a helpful assistant. Always try to understand informal terms like 'bucks' as USD or 'down under' as AUD when converting currencies."}]
    }
)

# Input Class for Currency Converter function
class CurrencyInput(BaseModel):
    amount: float = Field(..., description="Amount to convert")
    from_currency: str = Field(..., description="3-letter currency code to convert from (e.g., USD, INR, EUR)")
    to_currency: str = Field(..., description="3-letter currency code to convert to (e.g., USD, INR, EUR)")

    @field_validator('from_currency', 'to_currency')
    @classmethod
    def check_currency_code(cls, v):
        v = v.strip().upper()
        if len(v) != 3 or not v.isalpha():
            raise ValueError("Currency codes must be 3-letter alphabetical codes like USD or INR.")
        return v
    
    @field_validator('amount')
    @classmethod
    def check_amount(cls,a):
        if a < 0:
            raise ValueError("Amount should be positive")
        return a

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

def currency_converter_logged(*args, **kwargs):
    print("🛠️ [Currency Tool] Called with:", args, kwargs)
    result = currency_converter_pydantic(*args, **kwargs)
    print("✅ [Currency Tool] Result:", result)
    return result

def handle_currency_errors(error: Exception) -> str:
    return f"⚠️ Oops! There was an error: {str(error)}. Please try again with correct inputs."
currency_tool = StructuredTool.from_function(
    func=currency_converter_logged,
    name="currency_converter",  # <--- must be a string name
    args_schema=CurrencyInput,
    description="Converts currency using live rates. Provide amount, from_currency, and to_currency. Example: {\"amount\": 100, \"from_currency\": \"USD\", \"to_currency\": \"INR\"}",
    handle_tool_error = handle_currency_errors
)

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

weather_tool = StructuredTool.from_function(
    func=get_weather,
    name="weather_checker",
    description="Get current weather for a city. Format: Just pass the city name. Example: 'London'."
)

# ----- Tools (Calculator + Wikipedia + Web Search + Currency Converter + Weather) ----------
tools = load_tools(["llm-math", "wikipedia", "serpapi"], llm=llm)
tools += [currency_tool,weather_tool]


# ---------- Smarter Memory (Summary-based) ----------
research_memory = ConversationBufferMemory(
    llm=llm, 
    memory_key="chat_history",
    return_messages=True,
    input_key = "input"
)

research_prompt = SystemMessage(content="You are a research assistant that helps with factual queries from the web or Wikipedia.")

# Research agent
research_agent = initialize_agent(
    [tool for tool in tools if tool.name in ['wikipedia', 'serpapi']],
    llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True,
    memory=research_memory,
    handle_parsing_errors = True,
    allow_dangerous_tools = True,
    agent_kwargs={"system_message": research_prompt}
)

utility_prompt = SystemMessage(content="You are a utility assistant that helps with currency conversion, weather, and similar tasks.")

utility_memory = ConversationBufferMemory(
    llm=llm, 
    memory_key="chat_history",
    return_messages=True,
    input_key = "input"
)
# Utility agent
utility_agent = initialize_agent(
    [tool for tool in tools if tool.name in ['currency_converter', 'weather_checker']],
    llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
    memory = utility_memory,
    handle_parsing_errors = True,
    allow_dangerous_tools = True,
    agent_kwargs={"system_message": utility_prompt}
)

router_tools = [
    Tool(
        name="research_agent",
        func=research_agent.run,
        description="Good for information gathering and questions about people, places, or current events"
    ),
    Tool(
        name="utility_agent",
        func=utility_agent.run,
        description="Good for currency conversion, weather info, or math calculations"
    )
]

system_msg = SystemMessage(
    content="Your job is to route queries to the correct expert agent based on context and user intent. Remember prior conversation history."
)
shared_memory = ConversationBufferMemory(
    llm=llm, 
    memory_key="chat_history",
    return_messages=True,
    input_key = "input"
)

router_agent = initialize_agent(
    tools=router_tools,
    llm=llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True,
    memory = shared_memory,
    handle_parsing_errors = True,
    allow_dangerous_tools = True,
    agent_kwargs={
        "system_message":system_msg
    }
)

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    result = router_agent.invoke({"input":user_input})
    print("Bot:", result["output"])


