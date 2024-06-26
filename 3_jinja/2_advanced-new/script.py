from jinja2 import Environment, FileSystemLoader
import os

conversation = [
    {"role": "user", "content": "What's the weather like in Paris?"},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "id": "call_123",
                    "name": "get_current_weather",
                    "arguments": '{"location": "Paris, France", "format": "celsius"}',
                },
            }
        ],
    },
    {"role": "tool", "content": '{"content": 22}', "tool_call_id": "call_123"},
    {
        "role": "assistant",
        "content": "The current temperature in Paris, France is 22 degrees Celsius.",
    },
    {"role": "user", "content": "What weather will be in San Francisco next week?"},
]
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "format": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The temperature unit to use. Infer this from the users location.",
                    },
                },
                "required": ["location", "format"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_n_day_weather_forecast",
            "description": "Get an N-day weather forecast",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "format": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The temperature unit to use. Infer this from the users location.",
                    },
                    "num_days": {
                        "type": "integer",
                        "description": "The number of days to forecast",
                    },
                },
                "required": ["location", "format", "num_days"],
            },
        },
    },
]

request = {
    "bos_token": "<s>",
    "eos_token": "</s>",
    "messages": conversation,
    "tools": tools,
}

# Set up the Jinja2 environment
template_loader = FileSystemLoader(searchpath=os.path.dirname(__file__))
env = Environment(loader=template_loader, trim_blocks=True, lstrip_blocks=True)

# Load the template
template = env.get_template("trimed.jinja2")

# Render the template with the user object
rendered_template = template.render(**request).rstrip()

# Print or save the rendered template
print(rendered_template)

# Optionally, you can save the rendered template to an HTML file
with open("my-template-rendered.txt", "w") as f:
    f.write(rendered_template)
