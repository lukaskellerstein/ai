from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())  # read local .env file

import asyncio
from autogen_agentchat.ui import Console
from team import my_team


async def main() -> None:
    stream = my_team.run_stream(
        task="""Give me investment advice for MSFT stock and save it into a file.
                Then draw a graph of the close price and save it into a file.""")
    await Console(stream)

asyncio.run(main())