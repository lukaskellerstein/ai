print("Starting server...1")

import json
from typing import Any
import uvicorn
from pydantic import AnyUrl
import os
import base64

# MCP
from mcp.server.lowlevel import Server
import mcp.types as types 
from mcp.server.lowlevel.helper_types import ReadResourceContents

from mcp.server.sse import SseServerTransport
from starlette.applications import Starlette
from starlette.routing import Mount, Route

# Tools, Resources
from tools.get_stock_price import get_stock_price
from tools.get_dividend_date import get_dividend_date
from resources.read_sample_file import read_sample_file

def serve():
    print("Serve()")

    app = Server("mcp-finance")

    # ----------------------------------
    # ----------------------------------
    # Tools
    # ----------------------------------
    # ----------------------------------

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


    # ---------------------------------
    # ---------------------------------
    # Resources
    # ---------------------------------
    # ---------------------------------

    @app.list_resources()
    async def list_resources() -> list[types.Resource]:
        return [
            types.Resource(
                uri="string:///hello",
                name="Sample Text",
                mimeType="text/plain"
            ),
            types.Resource(
                uri="string:///sample.txt",
                name="Sample Text File's content send as string",
                mimeType="text/plain"
            ),
            types.Resource(
                uri="binary:///image",
                name="Picture in binary format",
                mimeType="image/png"
            ),
        ]
    
    @app.read_resource()
    async def handle_read_resource(uri: AnyUrl) -> list[ReadResourceContents]:

        print("READ RESOURCE")
        print(uri)
        print(type(uri))

        uri = str(uri)

        print(uri == "string:///hello")
        print(uri == "string:///sample.txt")
        print(uri == "binary:///image")

        # if str(uri) == "string:///hello":
        #     return "Hello"

        # if str(uri) == "string:///sample.txt":
        #     print("READING FILE's content")
        #     log_contents = read_sample_file()
        #     return log_contents
        
        # if str(uri) == "binary:///image":
        #     # Path to your image file
        #     IMAGE_PATH = os.path.join(os.path.dirname(__file__), "resources", "test_image.png")
        #     print("READING IMAGE AS BYTES")
        #     with open(IMAGE_PATH, "rb") as img_file:
        #         image_bytes = img_file.read()
                
        #     return image_bytes

        if uri == "string:///hello":
            return [
                ReadResourceContents(
                    content="Hello",
                    mime_type="text/plain"
                )
            ]

        elif uri == "string:///sample.txt":
            text = read_sample_file()
            return [
                ReadResourceContents(
                    content=text,
                    mime_type="text/plain"
                )
            ]

        elif uri == "binary:///image":
            image_path = os.path.join(os.path.dirname(__file__), "resources", "test_image.png")
            with open(image_path, "rb") as f:
                image_bytes = f.read()


            with open("original_check.png", "wb") as f:
                f.write(image_bytes)

            print("IMAGE BYTES LEN:", len(image_bytes))
            import base64
            b64_str = base64.b64encode(image_bytes).decode()
            
            print("Expected base64 length:", len(base64.b64encode(image_bytes).decode()))
            
            print("SAMPLE BASE64 (start):", b64_str[:100])  # Just a sample
            print("SAMPLE BASE64 (end):", b64_str[-100:])  # Just a sample

            return [
                ReadResourceContents(
                    content=image_bytes,
                    mime_type="image/png"
                )
            ]

        raise ValueError(f"Unknown resource: {uri}")


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