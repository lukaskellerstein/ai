from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient

# Create an agent that uses the OpenAI GPT-4o model.
model_client = OpenAIChatCompletionClient(
    model="gpt-4o",
    # api_key="YOUR_API_KEY",
)
agent = AssistantAgent(
    name="assistant",
    model_client=model_client,
    handoffs=["flights_refunder", "user"],
    # tools=[], # serializing tools is not yet supported
    system_message="Use tools to solve tasks.",
)

team = RoundRobinGroupChat(participants=[agent], termination_condition=MaxMessageTermination(2))

team_config = team.dump_component()  # dump component
print(team_config.model_dump_json())