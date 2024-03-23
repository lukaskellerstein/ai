import requests

API_URL = (
    "https://api-inference.huggingface.co/models/lukaskellerstein/mistral-7b-lex-16bit"
)
headers = {"Authorization": "Bearer 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"}


def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()


output = query(
    {
        "inputs": "Can you please let us know more details about your ",
    }
)
print(output)
