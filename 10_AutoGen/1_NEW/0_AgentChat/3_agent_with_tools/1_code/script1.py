from pprint import pprint
from autogen_agentchat.agents import AssistantAgent, CodeExecutorAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient, AzureOpenAIChatCompletionClient
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor
from autogen_ext.tools.code_execution import PythonCodeExecutionTool
from autogen_agentchat.messages import TextMessage
from dotenv import load_dotenv, find_dotenv
import asyncio
import os
from pathlib import Path

load_dotenv(find_dotenv())  # read local .env file

llm = OpenAIChatCompletionClient(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY")
)


# Define a tool
work_dir = Path("coding")
work_dir.mkdir(exist_ok=True)
local_executor = LocalCommandLineCodeExecutor(work_dir=work_dir)
python_tool = PythonCodeExecutionTool(LocalCommandLineCodeExecutor(work_dir="coding"))

# Main function
async def main() -> None:
    agent = CodeExecutorAgent(
        name="my_assistant", 
        code_executor=local_executor
    )

    message = TextMessage(content='''Here is some code
```python
print('Hello world')
```
''', source="xxx")

    # Question 1 (run) ---
    result = await agent.run(task=message)
    # pprint(result)
    print(result.messages[len(result.messages) - 1].content)


# NOTE: if running this inside a Python script you'll need to use asyncio.run(main()).
asyncio.run(main())