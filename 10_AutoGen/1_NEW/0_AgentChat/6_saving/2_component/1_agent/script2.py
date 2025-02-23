import os
import joblib
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())  # read local .env file

# Create an agent that uses the OpenAI GPT-4o model.
model_client = OpenAIChatCompletionClient(
    model="gpt-4o",
    api_key=os.getenv("OPENAI_API_KEY")
)
my_agent = AssistantAgent(
    name="assistant",
    model_client=model_client,
    handoffs=["flights_refunder", "user"],
    # tools=[], # serializing tools is not yet supported
    system_message="Use tools to solve tasks.",
)

# Deserialize the agent component
my_new_agent_config = joblib.load("my_agent_config.pkl")

# Load the agent component
my_new_agent = AssistantAgent.load_component(my_new_agent_config)

print(my_new_agent)