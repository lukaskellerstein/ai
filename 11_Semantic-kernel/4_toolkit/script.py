from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.functions import KernelArguments
from semantic_kernel.connectors.ai.prompt_execution_settings import PromptExecutionSettings
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from toolkit import EmailPlugin
import asyncio

async def main():
    kernel = Kernel()

    # LLM (AKA Service) ----------------------
    kernel.add_service(OpenAIChatCompletion())

    # Toolkit -------------------------------------------
    kernel.add_plugin(EmailPlugin(), plugin_name="EmailPlugin")

    arguments = KernelArguments(
        settings=PromptExecutionSettings(
                function_choice_behavior=FunctionChoiceBehavior.Auto(),
        )
    )

    # CALL 1
    result = await kernel.invoke_prompt(
        "What is the content of the email with subject 'Hello my friend'?", 
        arguments=arguments
    )
    print(result) 

    # CALL 2
    result = await kernel.invoke_prompt(
        "Send 'Hello friend' to the Lukas Kellerstein.", 
        arguments=arguments
    )
    print(result) 




# Run the main function
if __name__ == "__main__":
    asyncio.run(main())
