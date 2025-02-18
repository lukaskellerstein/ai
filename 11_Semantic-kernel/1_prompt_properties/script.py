from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.connectors.ai.open_ai import OpenAIChatPromptExecutionSettings
from semantic_kernel.prompt_template import InputVariable, PromptTemplateConfig
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

    # Parameters for prompt ---
    # req_settings = kernel.get_prompt_execution_settings_from_service_id(service_id)
    # req_settings.max_tokens = 2000
    # req_settings.temperature = 0.7
    # req_settings.top_p = 0.8

    # OR

    # execution_settings = OpenAIChatPromptExecutionSettings(
    #     service_id="my-openai-service",
    #     ai_model_id="gpt-4o-mini",
    #     max_tokens=2000,
    #     temperature=0.7,
    # )
    # --------------------------

    prompt_template_config = PromptTemplateConfig(
        # name="blabla1",
        template=prompt,
        # execution_settings=req_settings,
        # execution_settings=execution_settings,
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
