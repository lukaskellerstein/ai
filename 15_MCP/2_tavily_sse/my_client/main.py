import asyncio
from mcp.client.sse import sse_client
from mcp import ClientSession

async def start():

    tool_name = "tavily_search"  
    query = "What is a MCP with context of AI"  
    
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
            response = await session.call_tool(tool_name, {"query": query})
            print("Tool response:")
            print(response.content)


if __name__ == "__main__":
    asyncio.run(start())