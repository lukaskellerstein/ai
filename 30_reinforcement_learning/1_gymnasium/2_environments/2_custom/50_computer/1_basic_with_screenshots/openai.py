import os
import base64
import io
import pyautogui
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv

# Load API key from .env file
_ = load_dotenv(find_dotenv())

def _capture_screenshot():
    """Captures a screenshot and returns it as a base64-encoded string."""
    screenshot = pyautogui.screenshot()
    buffer = io.BytesIO()
    screenshot.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def call_with_image(system_message: str, message: str, image_base64: str):
    """Calls GPT-4o with a message and an image in base64 format."""
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_message},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": message},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}"
                        },
                    },
                ],
            },
        ],
    )
    return response

# Test function
def test():
    """Tests the screenshot capture and GPT-4o call."""
    message = "What do you see in this screenshot?"
    image_base64 = _capture_screenshot()
    response = call_with_image(message, image_base64)
    print(response)
