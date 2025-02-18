import asyncio

from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion, OpenAITextEmbedding
from semantic_kernel.core_plugins import TextMemoryPlugin
from semantic_kernel.memory import SemanticTextMemory, VolatileMemoryStore
from uuid import uuid4 as uuid
from semantic_kernel.connectors.memory.chroma import ChromaMemoryStore

async def main():
    kernel = Kernel()

    kernel.add_service(OpenAIChatCompletion(service_id="default", ai_model_id="gpt-4o"))
    
    embedding_gen = OpenAITextEmbedding(
        service_id="ada",
        ai_model_id="text-embedding-ada-002",
    )
    kernel.add_service(embedding_gen)

    # Nefunguje
    # memory = SemanticTextMemory(storage=ChromaMemoryStore(persist_directory='./chromadb'), embeddings_generator=embedding_gen)
    memory = SemanticTextMemory(storage=VolatileMemoryStore(), embeddings_generator=embedding_gen)
    kernel.add_plugin(TextMemoryPlugin(memory), "memory")

    await memory.save_information(collection="generic", id=uuid(), text="My name is Lukas")
    await memory.save_information(collection="generic", id=uuid(), text="My budget for 2024 is $100,000")
    await memory.save_information(collection="generic", id=uuid(), text="My budget for 2025 is $200,000")

    result = await kernel.invoke_prompt(
        # function_name="budget",
        # plugin_name="BudgetPlugin",
        prompt="{{memory.recall 'budget 2024'}} What is my budget for 2024?",
    )
    print(result)

    result = await kernel.invoke_prompt(
        # function_name="budget",
        # plugin_name="BudgetPlugin",
        prompt="{{memory.recall 'name'}} What is my name?",
    )
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
