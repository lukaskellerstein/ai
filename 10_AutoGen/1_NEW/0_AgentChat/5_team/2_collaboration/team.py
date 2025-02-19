from autogen_agentchat.ui import Console
from autogen_agentchat.teams import RoundRobinGroupChat
from agent_coder import my_agent_coder
from agent_researcher import my_agent_researcher

# Team of agents
my_team = RoundRobinGroupChat(
    [my_agent_researcher, my_agent_coder], 
    max_turns=10 
)

