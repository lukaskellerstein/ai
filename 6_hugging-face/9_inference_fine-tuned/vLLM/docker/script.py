from mistralai.models.chat_completion import ChatMessage
from mistralai.client import MistralClient



model = "mistralai/Mistral-7B-Instruct-v0.2"
api_key="TYPE YOUR API KEY"

client = MistralClient(
    api_key=api_key, 
    endpoint="http://localhost:8000"
    )




tools = [
    {
        "type": "function",
        "function": {
            "name": "retrieve_payment_status",
            "description": "Get payment status of a transaction",
            "parameters": {
                "type": "object",
                "properties": {
                    "transaction_id": {
                        "type": "string",
                        "description": "The transaction id.",
                    }
                },
                "required": ["transaction_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "retrieve_payment_date",
            "description": "Get payment date of a transaction",
            "parameters": {
                "type": "object",
                "properties": {
                    "transaction_id": {
                        "type": "string",
                        "description": "The transaction id.",
                    }
                },
                "required": ["transaction_id"],
            },
        },
    }
]


messages = [
    ChatMessage(role="user", content="What's the status of my transaction?")
]

response1 = client.chat(model=model, messages=messages, tools=tools, tool_choice="auto")

# messages.append(response1.choices[0].message)


print("RESPONSE")
print(response1)