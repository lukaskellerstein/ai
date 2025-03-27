import asyncio
import os
import base64

from mcp.client.sse import sse_client
from mcp import ClientSession

from pydantic import AnyUrl, TypeAdapter

async def start():

    tool_name = "get_stock_price"  # Tool to call (e.g., "get_stock_price" or "get_dividend_date")
    ticker = "MSFT"  # Default stock ticker
    
    server_url = "http://localhost:8001"  # URL of the MCP server

    async with sse_client(server_url + "/sse") as streams:
        async with ClientSession(*streams) as session:
            # Test initialization
            result = await session.initialize()
            print("Initialize result:")
            print(result)

            # Test ping
            ping_result = await session.send_ping()
            print("Ping result:")
            print(ping_result)

            # List tools
            tools_list_response = await session.list_tools()
            print("Available tools:")
            for tool in tools_list_response.tools:
                print(f"Tool name: {tool.name}, Description: {tool.description}")

            # Call the tool
            response = await session.call_tool(tool_name, {"ticker": ticker})
            print("Tool response:")
            print(response.content)

            # List resources
            resources_list_response = await session.list_resources()
            print("Available resources:")
            for resource in resources_list_response.resources:
                print(f"Resource URI: {resource.uri}, Name: {resource.name}, MIMEType: {resource.mimeType}")

            # Get a specific resource
            print("--" * 20)

            # # Get a string
            # resource0_file = "string:///hello"
            # resource0 = await session.read_resource(resource0_file)
            # print(f"Resource '{resource0_file}' content:")
            # print(resource0)

            # print("--" * 20)

            # # Get a text file content as string
            # resource01_file = "string:///sample.txt"
            # resource01 = await session.read_resource(resource01_file)
            # print(f"Resource {resource01_file} content:")
            # print(resource01)

            # print("--" * 20)

            resource2 = await session.read_resource("binary:///image")
            received_blob = resource2.contents[0].blob

            print("RECEIVED B64 (start):", received_blob[:100])
            print("RECEIVED B64 (end):", received_blob[-100:])

            decoded_bytes = base64.urlsafe_b64decode(received_blob)
            print("Decoded length:", len(decoded_bytes))  # On client
            
            output_path = os.path.join(os.path.dirname(__file__), "saved_image.png")
            with open(output_path, "wb") as f:
                f.write(decoded_bytes)


            print("--" * 20)

if __name__ == "__main__":
    asyncio.run(start())