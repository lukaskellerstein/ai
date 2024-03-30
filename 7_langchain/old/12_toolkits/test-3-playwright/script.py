# import asyncio
# from dotenv import load_dotenv, find_dotenv

# from langchain.agents.agent_toolkits import PlayWrightBrowserToolkit
# from langchain.tools.playwright.utils import (
#     create_async_playwright_browser,
#     create_sync_playwright_browser
# )

# _ = load_dotenv(find_dotenv())  # read local .env file


# async def main():
#     browser = create_sync_playwright_browser()
#     toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=browser)

#     tools = toolkit.get_tools()
#     tools_by_name = {tool.name: tool for tool in tools}
#     navigate_tool = tools_by_name["navigate_browser"]
#     get_elements_tool = tools_by_name["get_elements"]

#     result = await navigate_tool.arun(
#         {"url": "https://web.archive.org/web/20230428131116/https://www.cnn.com/world"}
#     )
#     print(result)

#     # The browser is shared across tools, so the agent can interact in a stateful manner
#     result = await get_elements_tool.arun(
#         {"selector": ".container__headline", "attributes": ["innerText"]}
#     )
#     print(result)

#     # If the agent wants to remember the current webpage, it can use the `current_webpage` tool
#     result = await tools_by_name["current_webpage"].arun({})
#     print(result)


# # Running the main() function
# if __name__ == "__main__":
#     asyncio.run(main())


from dotenv import load_dotenv, find_dotenv

from langchain.agents.agent_toolkits import PlayWrightBrowserToolkit
from langchain.tools.playwright.utils import create_sync_playwright_browser

_ = load_dotenv(find_dotenv())  # read local .env file


browser = create_sync_playwright_browser()
toolkit = PlayWrightBrowserToolkit.from_browser(sync_browser=browser)

tools = toolkit.get_tools()
tools_by_name = {tool.name: tool for tool in tools}
navigate_tool = tools_by_name["navigate_browser"]
get_elements_tool = tools_by_name["get_elements"]

result = navigate_tool.run(
    {"url": "https://web.archive.org/web/20230428131116/https://www.cnn.com/world"}
)
print(result)

# The browser is shared across tools, so the agent can interact in a stateful manner
result = get_elements_tool.run(
    {"selector": ".container__headline", "attributes": ["innerText"]}
)
print(result)

# If the agent wants to remember the current webpage, it can use the `current_webpage` tool
result = tools_by_name["current_webpage"].run({})
print(result)
