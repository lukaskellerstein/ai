# Agents

## Agent Types

### Zero-shot ReAct

zero-shot-react-description
ZERO_SHOT_REACT_DESCRIPTION

- determine which tool to use based solely on the tool's description
- NO memory
- is NOT able to chat with user, only using tools !

### Structured input ReAct

structured-chat-zero-shot-react-description
STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION

- The structured tool chat agent **is capable of using multi-input tools**
- NO memory
- is NOT able to chat with user, only using tools !

### Converational

conversational-react-description
CONVERSATIONAL_REACT_DESCRIPTION

- has memory
- is able to chat with user

chat-conversational-react-description
CHAT_CONVERSATIONAL_REACT_DESCRIPTION

- lets us create a conversational agent using a **chat model instead of an LLM.**
- has memory
- is able to chat with user

### ReAct document store

react-docstore
REACT_DOCSTORE

- ReAct framework to **interact with a docstore.** Two tools must be provided: a Search tool and a Lookup tool

### Self ask with search

self-ask-with-search
SELF_ASK_WITH_SEARCH

- utilizes a single tool that should be named Intermediate Answer. This tool should be able to lookup factual answers to questions.

### OpenAI Functions

FINISH !!!!
