import base64
import json
import requests
import pyautogui
from io import BytesIO

# Server URL
SERVER_URL = "http://127.0.0.1:8000/parse/"

# Function to encode image to base64
def _encode_image(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# Function to send image to the server
def _send_image_to_server(base64_image):
    payload = json.dumps({"base64_image": base64_image})
    headers = {"Content-Type": "application/json"}
    
    response = requests.post(SERVER_URL, data=payload, headers=headers)
    return response.json()

# Function to save base64 image to file
def _save_base64_image(base64_str, output_file="output.png"):
    image_data = base64.b64decode(base64_str)
    with open(output_file, "wb") as f:
        f.write(image_data)
    print(f"Saved parsed image to {output_file}")

def _save_to_file(data, filename):
    with open(filename, "w") as f:
        json.dump(data, f)
    print(f"Saved data to {filename}")

# Function to analyze image
def analyze_image(screenshot) -> dict:
    base64_image = _encode_image(screenshot)
    response = _send_image_to_server(base64_image)
    
    if "som_image_base64" in response:
        _save_base64_image(response["som_image_base64"], "parsed_output.png")
    
    if "parsed_content_list" in response:
        _save_to_file(response["parsed_content_list"], "parsed_content_list.json")

    return {
        "parsed_image_base64": response.get("som_image_base64", ""),
        "parsed_image_path": "./parsed_output.png",
        "parsed_content_list": response.get("parsed_content_list", []),
    }

# Test function
def test():
    print("Capturing screenshot...")
    screenshot = pyautogui.screenshot()
    
    print("Sending image to Omniparser server...")
    response = analyze_image(screenshot)
    
    print("Parsed Content List:")
    for item in response["parsed_content_list"]:
        print(item)
