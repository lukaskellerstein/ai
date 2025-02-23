![ml-dl](./assets/ml-vs-dl.jpg)

**Artificial intelligence (AI)** refers to the ability of machines to perform tasks that would normally require human intelligence, such as reasoning, perception, learning, and decision making. AI is a broad field that encompasses many different subfields, including machine learning and deep learning.

**Machine learning** is a subset of AI that involves training machines to learn from data without being explicitly programmed. In machine learning, algorithms are developed that can learn from data and improve their performance over time. This process involves feeding large amounts of data into a machine learning model, which then makes predictions or classifications based on that data. Examples of machine learning applications include image recognition, natural language processing, and recommendation systems.

**Deep learning** is a type of machine learning that involves the use of artificial neural networks, which are modeled after the structure and function of the human brain. Deep learning algorithms can automatically extract features from data, allowing them to learn from complex and large datasets. Examples of deep learning applications include speech recognition, object detection, and autonomous driving. Deep learning has shown great potential for solving complex problems in various fields, from healthcare to finance to robotics.

**Reinforcement learning** is another important subset of machine learning that focuses on training agents to make decisions by interacting with an environment. In reinforcement learning, an agent learns through trial and error, receiving rewards or penalties based on its actions. The goal is to maximize cumulative rewards over time by learning an optimal strategy, or policy, for decision-making.

## Notes

Release disk space for WSL2: https://stephenreescarter.net/how-to-shrink-a-wsl2-virtual-disk/

## Tutorial

1. Pytorch
   - ANN
   - CNN
   - RNN
   - Transformer
2. OpenAI
   - Prompt
   - Prompt engineering
   - Chat
3. Hugging Face
   - Explore
     - text2text-generation
     - text-generation
     - text-img
     - text-speech
     - ...etc
   - Models = Pre-trained Transformers
     - Run (32bit, 16bit, 8bit, 4bit) (CPU vs. GPU) (Accelerate)
     - Train / Fine-tune
   - Datasets
   - Multimodal models
4. Langchain

## Finish

- Fine-tuning multimodal models

- Autogen

  - Agent with RAG (ideally ChromaDB) = Memory ?? https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/memory.html

  - Team - SelectorGroupChat with Human-in-the-loop - https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/selector-group-chat.html

  - Team - Magentic-One - https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/magentic-one.html

- Semantic-kernel

  - Agent with RAG (ideally ChromaDB)
  - Orchestration of Agents = Teams ??

- Reinforcement learning

  - Gymansium environments
  - Stable-baseline 3
