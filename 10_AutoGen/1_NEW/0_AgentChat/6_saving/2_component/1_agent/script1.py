import os
import joblib
from autogen_agentchat.agents import AssistantAgent
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

# Save the agent component
my_agent_config = my_agent.dump_component()
print(my_agent_config.model_dump_json())

# Serialize the agent component
joblib.dump(my_agent_config, "my_agent_config.pkl")