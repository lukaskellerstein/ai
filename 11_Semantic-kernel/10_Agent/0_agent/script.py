import asyncio
from pprint import pprint
from semantic_kernel import Kernel
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.contents import ChatHistory

kernel = Kernel()
kernel.add_service(OpenAIChatCompletion(service_id="llm"))

# Define the agent with name and instructions
agent = ChatCompletionAgent(
    name="my_assistant", 
    kernel=kernel, 
    service_id="llm"
)

async def main():
    # Define the chat history
    chat_history = ChatHistory()
    chat_history.add_system_message("You are a helpful AI assistant.")

    # No reason to share with the agent the chat history
    # chat_history.add_user_message("Tell me a joke.")
    # chat_history.add_assistant_message("1+1=3")
    
    chat_history.add_user_message("That's not a joke. Tell me a new one.")
    
    # Invoke the agent to get a response
    async for content in agent.invoke(chat_history):
        print("Agent response:")
        pprint(content)


if __name__ == "__main__":
    asyncio.run(main())
