import asyncio
from mcp.client.sse import sse_client
from mcp import ClientSession

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

            # Call the tool
            response = await session.call_tool(tool_name, {"ticker": ticker})
            print("Tool response:")
            print(response.content)


if __name__ == "__main__":
    asyncio.run(start())