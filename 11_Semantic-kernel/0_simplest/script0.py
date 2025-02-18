from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.prompt_template import PromptTemplateConfig
import asyncio

async def main():
    kernel = Kernel()

    # LLM (AKA Service) ----------------------
    kernel.add_service(OpenAIChatCompletion())

    # Prompt (AKA function) ----------------------
    result = await kernel.invoke_prompt("Tell me a joke")

    print(result) 



# Run the main function
if __name__ == "__main__":
    asyncio.run(main())
