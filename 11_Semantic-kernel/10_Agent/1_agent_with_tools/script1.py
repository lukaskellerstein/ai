import asyncio
from pprint import pprint
from semantic_kernel import Kernel
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.contents import ChatHistory
from semantic_kernel.functions import kernel_function
from semantic_kernel.functions import KernelArguments
from semantic_kernel.connectors.ai.prompt_execution_settings import PromptExecutionSettings
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior

kernel = Kernel()
kernel.add_service(OpenAIChatCompletion(service_id="llm"))

# Tools (as plugin)
class MyPlugin:
    """Description: My custom plugin"""

    @kernel_function(description="Get weather")
    def get_weather (self, city: str) -> str:
        return f"The weather in {city} is 73 degrees and Sunny."

    @kernel_function(description="Get price for Stock ticker")
    def get_stock_price (self, ticker: str) -> str:
        return f"The price of {ticker} is $100"

kernel.add_plugin(MyPlugin(), plugin_name="MyPlugin")

# Arguments
arguments = KernelArguments(
    settings=PromptExecutionSettings(
            function_choice_behavior=FunctionChoiceBehavior.Auto(),
    )
)

# Define the agent with name and instructions
agent = ChatCompletionAgent(
    name="my_assistant", 
    kernel=kernel, 
    service_id="llm",
    arguments=arguments,
)

async def main():
    # Define the chat history
    chat_history = ChatHistory()
    chat_history.add_system_message("You are a helpful AI assistant.")

    # No reason to share with the agent the chat history
    # chat_history.add_user_message("Tell me a joke.")
    # chat_history.add_assistant_message("1+1=3")
    
    chat_history.add_user_message("What is weather in Prague?")
    
    # Invoke the agent to get a response
    async for content in agent.invoke(chat_history):
        print("Agent response:")
        pprint(content)


if __name__ == "__main__":
    asyncio.run(main())
