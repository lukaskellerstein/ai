print("Starting server...1")
import json
import uvicorn

from mcp.server.lowlevel import Server
import mcp.types as types

from mcp.server.sse import SseServerTransport
from starlette.applications import Starlette
from starlette.routing import Mount, Route
from mcp.shared.exceptions import McpError

from dotenv import load_dotenv
load_dotenv()

from tools.tavily_search import search
from tools.tavily_extract import extract



def serve():
    print("Serve()")

    app = Server("mcp-search")

    @app.call_tool()
    async def fetch_tool(
        name: str, arguments: dict
    ) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:

        print("CALL TOOL")
        print(name)
        print(arguments)

        result = None
        if name == "tavily_search":
            result = await search(arguments)
        elif name == "tavily_extract":
            result = await extract(arguments)
        else:
            raise McpError(f"Unknown tool: {name}")


        print("Result:")
        print(result)

        result_json = json.dumps(result)

        return [types.TextContent(type="text", text=result_json)]

    @app.list_tools()
    async def list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name="tavily_search",
                description="A powerful web search tool using Tavily's AI engine...",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "max_results": {"type": "number", "default": 10}
                    },
                    "required": ["query"]
                },
            ),
            types.Tool(
                name="tavily_extract",
                description="Extracts and processes raw content from URLs...",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "urls": {"type": "array", "items": {"type": "string"}},
                        "extract_depth": {"type": "string", "default": "basic"}
                    },
                    "required": ["urls"]
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
    uvicorn.run(
        starlette_app, 
        host="0.0.0.0", 
        port=8001,
    )


if __name__ == "__main__":
    print("Starting server...2")
    try:
        serve()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received. Cleaning up before exit...")