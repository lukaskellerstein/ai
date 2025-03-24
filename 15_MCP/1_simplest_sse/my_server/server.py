print("Starting server...2")

import json
import uvicorn

from mcp.server.lowlevel import Server
import mcp.types as types

from mcp.server.sse import SseServerTransport
from starlette.applications import Starlette
from starlette.routing import Mount, Route

from tools.get_stock_price import get_stock_price
from tools.get_dividend_date import get_dividend_date

def serve():
    print("Serve()")

    app = Server("mcp-finance")

    @app.call_tool()
    async def fetch_tool(
        name: str, arguments: dict
    ) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:

        print("CALL TOOL")
        print(name)
        print(arguments)

        result = None
        if name == "get_stock_price":
            result = get_stock_price(arguments["ticker"])
        elif name == "get_dividend_date":
            result = get_dividend_date(arguments["ticker"])

        print("Result:")
        print(result)

        result_json = json.dumps(result)

        return [types.TextContent(type="text", text=result_json)]

    @app.list_tools()
    async def list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name="get_stock_price",
                description="Get the current stock price.",
                inputSchema={
                    "type": "object",
                    "required": ["ticker"],
                    "properties": {
                        "ticker": {
                            "type": "string",
                            "description": "The stock ticker symbol (e.g., AAPL, GOOG)",
                        }
                    },
                },
            ),
            types.Tool(
                name="get_dividend_date",
                description="Get the next dividend date of a stock.",
                inputSchema={
                    "type": "object",
                    "required": ["ticker"],
                    "properties": {
                        "ticker": {
                            "type": "string",
                            "description": "The stock ticker symbol (e.g., AAPL, GOOG)",
                        }
                    },
                },
            )
        ]

    # ---------------------------------
    # SSE Server Transport
    # ---------------------------------

    sse = SseServerTransport("/messages/")

    async def handle_sse(request):
        async with sse.connect_sse(
            request.scope, request.receive, request._send
        ) as streams:
            await app.run(
                streams[0], streams[1], app.create_initialization_options()
            )

    starlette_app = Starlette(
        debug=True,
        routes=[
            Route("/sse", endpoint=handle_sse),
            Mount("/messages/", app=sse.handle_post_message),
        ],
    )

    # Run the server
    uvicorn.run(starlette_app, host="0.0.0.0", port=8001)


if __name__ == "__main__":
    print("Starting server...2")
    try:
        serve()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received. Cleaning up before exit...")