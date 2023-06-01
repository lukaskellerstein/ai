import os
import openai
import json
from dotenv import load_dotenv, find_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.encoders import jsonable_encoder

_ = load_dotenv(find_dotenv())  # read local .env file

openai.api_key = os.getenv("OPENAI_API_KEY")


# ------------------------------------------------------------


def get_completion_from_messages(messages, model="gpt-3.5-turbo", temperature=0):
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,  # this is the degree of randomness of the model's output
    )
    #     print(str(response.choices[0].message))
    return response.choices[0].message["content"]


# ------------------------------------------------------------

# Example 1
messages = [
    # {"role": "system", "content": "You are an assistant that speaks like Shakespeare."},
]

# ------------------------------------------------------------


class Message(BaseModel):
    role: str
    content: str


app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/init-chatbot")
async def init(initMessages):
    messages.extend(initMessages)
    return {"response": "ok"}


@app.post("/send-message/")
async def create_item(message: Message):
    print("HOHOHOHOH")
    print(message)
    messages.append(message)

    formattedMessages = []

    for m in messages:
        json_compatible_item_data = jsonable_encoder(m)
        formattedMessages.append(json_compatible_item_data)

    print(formattedMessages)

    response = get_completion_from_messages(formattedMessages, temperature=1)

    response: Message = {"role": "assistant", "content": response}
    messages.append(response)
    return response
