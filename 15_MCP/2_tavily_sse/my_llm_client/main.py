import os
from openai import OpenAI
from mcp_server_client import McpServerClient
import asyncio
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize MCP server client
mcp_server = McpServerClient("http://localhost:8001")

# Initialize OpenAI client
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

async def start():

    try:

        # Connect to the MCP server
        await mcp_server.connect()

        # Get the tool list from MCP server
        # ---------------------------------------
        tools_list_response = await mcp_server.list_tools()

        # Print available tools
        print("Available tools:")
        print(tools_list_response)
        # ---------------------------------------

        # Convert the tools list from MCP to OpenAI format
        tools_openai = []
        for tool in tools_list_response:
            tools_openai.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema,
                }
            })



        # Example of messages
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": "What is weather today in Prague?"},
        ]

        # OpenAI client
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=tools_openai,  # Custom tools
            tool_choice="auto"  # Allow AI to decide if a tool should be called
        )

        response_message = response.choices[0].message
        print("----------------------------------------")
        print("First response:", response_message)
        print("----------------------------------------")

        if response_message.tool_calls:
            # Find the tool call content
            tool_call = response_message.tool_calls[0]

            # Extract tool name and arguments
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments) 
            tool_id = tool_call.id

            # Call the function on MCP server
            function_response = await mcp_server.call_tool(function_name, function_args)
            print("Tool response from MCP:", function_response)

            resp_json = json.dumps(function_response[0].text)

            # Append the tool call and response to the messages
            messages.append({
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": tool_id,  
                        "type": "function",
                        "function": {
                            "name": function_name,
                            "arguments": json.dumps(function_args),
                        }
                    }
                ]
            })

            # Append the tool response
            messages.append({
                "role": "tool",
                "tool_call_id": tool_id,  
                "name": function_name,
                "content": resp_json,
            })

            print("Messages after appending tool response:")
            print(messages)

            # Second call to get final response based on function output
            second_response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
            )
            final_answer = second_response.choices[0].message

            print("----------------------------------------")
            print("----------------------------------------")
            print("----------------------------------------")
            print("FINAL response:", final_answer)
            print("----------------------------------------")
            print("----------------------------------------")
            print("----------------------------------------")

    finally:
        # Disconnect from the MCP server
        await mcp_server.disconnect()

   

if __name__ == "__main__":
    asyncio.run(start())