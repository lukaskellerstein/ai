import os
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import Swarm
from agent_coder import my_agent_coder
from agent_financial_analyst import my_agent_financial_analyst
from agent_news_analyst import my_agent_news_analyst
from agent_writer import my_agent_writer
from agent_planner import my_agent_planner


text_termination = TextMentionTermination("TERMINATE")
termination = text_termination

my_team = Swarm(
    participants=[my_agent_planner, my_agent_writer, my_agent_news_analyst, my_agent_financial_analyst, my_agent_coder], 
    termination_condition=termination
)


