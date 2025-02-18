from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.connectors.ai.open_ai import OpenAIChatPromptExecutionSettings
from semantic_kernel.prompt_template import InputVariable, PromptTemplateConfig
from semantic_kernel.contents import ChatHistory
from semantic_kernel.functions import KernelArguments
import asyncio

async def main():
    kernel = Kernel()

    # LLM (AKA Service) ----------------------
    service1 = OpenAIChatCompletion(
            service_id="my-openai-service"
        )
    kernel.add_service(service1)

    # Prompt (AKA function) ----------------------
    prompt = """
History:
=====
{{$history}}
=====

User: {{$user_input}}
Assistant: 
"""

    prompt_template_config = PromptTemplateConfig(
        template=prompt,
        input_variables=[
            InputVariable(name="user_input", description="The user input", is_required=True),
            InputVariable(name="history", description="The conversation history", is_required=True),
        ],
    )

    function = kernel.add_function(
        function_name="blabla1_prompt",
        plugin_name="blabla1_plugin",
        prompt_template_config=prompt_template_config,
    )

    chat_history = ChatHistory()
    chat_history.add_system_message("You are a helpful AI assistant.")

    # CALL 1
    user_message = "Hi, I'm looking for book suggestions"
    arguments = KernelArguments(user_input=user_message, history=chat_history)
    result = await kernel.invoke(function, arguments)
    print(result) 

    # Enhance the chat history - MANUALLY
    chat_history.add_user_message(user_message)
    chat_history.add_assistant_message(result)

    # CALL 2
    user_message = "I like science fiction"
    arguments = KernelArguments(user_input=user_message, history=chat_history)
    result = await kernel.invoke(function, arguments)
    print(result)



# Run the main function
if __name__ == "__main__":
    asyncio.run(main())
