import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    server_script = "../my_server/main.py"  # Path to the MCP server script
    tool_name = "get_stock_price"  # Tool to call (e.g., "get_stock_price" or "get_dividend_date")
    ticker = "MSFT"  # Default stock ticker
    
    server_params = StdioServerParameters(command="python", args=[server_script])
    
    async with stdio_client(server_params) as stdio_transport:
        async with ClientSession(*stdio_transport) as session:
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


if __name__ == "__main__":
    asyncio.run(main())
