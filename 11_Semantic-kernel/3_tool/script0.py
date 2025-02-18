from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.connectors.ai.open_ai import OpenAIChatPromptExecutionSettings
from semantic_kernel.prompt_template import InputVariable, PromptTemplateConfig
from semantic_kernel.contents import ChatHistory
from semantic_kernel.functions import KernelArguments
from semantic_kernel.functions import kernel_function
from semantic_kernel.connectors.ai.prompt_execution_settings import PromptExecutionSettings
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
import asyncio

async def main():
    kernel = Kernel()

    # LLM (AKA Service) ----------------------
    kernel.add_service(OpenAIChatCompletion())

    # Tool -------------------------------------------
    @kernel_function(description="Get price for Stock ticker")
    def get_stock_price (ticker: str) -> str:
        return f"The price of {ticker} is $100"
    
    kernel.add_function(
        plugin_name="stock_plugin",
        function=get_stock_price
    )

    # Arguments
    arguments = KernelArguments(
        settings=PromptExecutionSettings(
                function_choice_behavior=FunctionChoiceBehavior.Auto(),
        )
    )

    # CALL 1
    prompt = "What is the price of MSFT?"
    result = await kernel.invoke_prompt(prompt, arguments=arguments)

    # THE RESULT CONTAINS THE CALL OF FUNCTION !!!!
    # LIKE AGENT
    print(result) 


# Run the main function
if __name__ == "__main__":
    asyncio.run(main())
