import os
from typing import Any, Dict
import aiohttp

EXTRACT_URL = "https://api.tavily.com/extract"
API_KEY = os.getenv("TAVILY_API_KEY")

if not API_KEY:
    raise EnvironmentError("TAVILY_API_KEY environment variable is required")

async def extract(params: Dict[str, Any]) -> Dict[str, Any]:
    payload = {
        **params,
        "api_key": API_KEY
    }

    async with aiohttp.ClientSession(headers={
        "accept": "application/json",
        "content-type": "application/json",
        "x-api-key": API_KEY
    }) as session:
        async with session.post(EXTRACT_URL, json=payload) as resp:
            resp.raise_for_status()
            response = await resp.json()
            return response