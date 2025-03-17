import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    server_script = "../server/main.py"  # Path to the MCP server script
    tool_name = "get_stock_price"  # Tool to call (e.g., "get_stock_price" or "get_dividend_date")
    ticker = "MSFT"  # Default stock ticker
    
    server_params = StdioServerParameters(command="python", args=[server_script])
    
    async with stdio_client(server_params) as stdio_transport:
        async with ClientSession(*stdio_transport) as session:
            await session.initialize()
            response = await session.call_tool(tool_name, {"ticker": ticker})
            print(response.content)

if __name__ == "__main__":
    asyncio.run(main())
