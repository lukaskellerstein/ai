from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.prompt_template import PromptTemplateConfig
import asyncio

async def main():
    kernel = Kernel()

    # LLM (AKA Service) ----------------------
    service1 = OpenAIChatCompletion(
            service_id="my-openai-service"
        )
    kernel.add_service(service1)

    # Prompt (AKA function) ----------------------
    prompt = """Tell me a joke"""

    prompt_template_config = PromptTemplateConfig(
        template=prompt,
    )

    function = kernel.add_function(
        function_name="blabla1_prompt",
        plugin_name="blabla1_plugin",
        prompt_template_config=prompt_template_config,
    )

    result = await kernel.invoke(function)

    print(result) 



# Run the main function
if __name__ == "__main__":
    asyncio.run(main())
