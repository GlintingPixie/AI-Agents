from langchain_openai import AzureChatOpenAI
from langchain.agents import initialize_agent
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents import AgentType
from langchain.memory import ConversationBufferMemory
import os
from dotenv import load_dotenv
load_dotenv()

# ---- CONFIG ----
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_key=os.getenv("AZURE_OPENAI_API_KEY")
api_version=os.getenv('AZURE_OPENAI_API_VERSION')
deployment_name = os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME')

# Create the model
llm = AzureChatOpenAI(
    deployment_name=deployment_name,
    temperature=0,
    api_key=api_key,
    azure_endpoint =azure_endpoint,
    api_version=api_version
)

# 2. Load tools (calculator)
tools = load_tools(["llm-math"], llm=llm)

# 3. Add memory (to store conversation)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# 4. Create the agent
agent = initialize_agent(
    tools,
    llm,
    agent='chat-conversational-react-description',
    verbose=True,
    memory = memory
)

while True:
    user_input = input("How can i help you today. Type 'stop' to exit\n")
    if user_input == 'stop':
        break

    print(agent.invoke({'input':f"${user_input}"})['output'])

