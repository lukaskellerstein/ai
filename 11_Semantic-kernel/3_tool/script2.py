from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.connectors.ai.open_ai import OpenAIChatPromptExecutionSettings
from semantic_kernel.prompt_template import InputVariable, PromptTemplateConfig
from semantic_kernel.contents import ChatHistory
from semantic_kernel.functions import KernelArguments
from semantic_kernel.functions import kernel_function
from toolkit import StocksPlugin
import asyncio

async def main():
    kernel = Kernel()

    # LLM (AKA Service) ----------------------
    service1 = OpenAIChatCompletion(
            service_id="my-openai-service"
        )
    kernel.add_service(service1)

    # Prompt (AKA Semantic function) ----------------------
    prompt = """You are a helpful AI assistant.
    {{$user_input}}
    """

    prompt_template_config = PromptTemplateConfig(
        template=prompt,
        input_variables=[
            InputVariable(name="user_input", description="The user input", is_required=True),
        ],
    )

    prompt = kernel.add_function(
        function_name="blabla1_prompt",
        plugin_name="blabla1_plugin",
        prompt_template_config=prompt_template_config,
    )

    # Tool (AKA Native function) ----------------------
    # how ???

    # @kernel_function(description="Get Stock price", name="get_stock_price")
    # def get_stock_price (ticker: str) -> str:
    #     return f"The price of {ticker} is $100"

    # kernel.add_function(
    #     function_name="stock_price",
    #     plugin_name="stock_plugin",
    #     function=get_stock_price
    # )

    # Toolkit -------------------------------------------
    kernel.add_plugin(StocksPlugin(), plugin_name="StocksPlugin")

    # CALL 1
    arguments = KernelArguments(user_input="What is the price of MSFT?")
    result = await kernel.invoke(prompt, arguments)
    print(result) 




# Run the main function
if __name__ == "__main__":
    asyncio.run(main())
