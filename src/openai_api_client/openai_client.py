import asyncio
import json
from aiohttp import ClientSession
from collections.abc import AsyncGenerator
from dotenv import load_dotenv
import os

load_dotenv()

# Default OpenAI API endpoint
DEFAULT_BASE_URL = "https://api.openai.com/v1"

# Default headers for HTTP requests
DEFAULT_HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",  # Replace with your OpenAI API key
}

class OpenAIClient:
    """Client for interacting with OpenAI's API using default settings."""

    def __init__(self, model: str = "gpt-4o-mini", base_url: str = DEFAULT_BASE_URL, api_key: str = os.getenv("OPENAI_API_KEY")):
        self.model = model
        self.base_url = base_url
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

    async def stream_completion(
        self, chat_thread: list[dict], stream: bool = True, format=""
    ) -> AsyncGenerator[dict, None]:
        """
        Stream responses from the OpenAI API's chat completion endpoint.
        
        Args:
            messages (list[dict]): List of messages for the chat input.
            model (str): Model name (e.g., 'gpt-4').
            stream (bool): Whether to stream the response.
        
        Yields:
            dict: Parsed chunks of the response from the API.
        """
        try:
            url = f"{self.base_url}/chat/completions"
            payload = {
                "model": self.model,
                "messages": chat_thread,
                "stream": stream,
            }

            async with ClientSession() as session:
                async with session.post(url=url, headers=self.headers, json=payload) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"HTTP {response.status}: {error_text}")

                    async for raw_line in response.content:
                        if not raw_line.strip():
                            continue
                        try:
                            line = raw_line.decode("utf-8").strip()
                            if line.startswith("data: "):
                                line = line[6:]  # Remove 'data: ' prefix
                            if line:
                                yield json.loads(line)
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            raise e


async def main():
    client = OpenAIClient()
    messages = [{
                "role": "system",
                "content": "You are a young student. Answer the questions with errors."
            },
            {
                "role": "user",
                "content": "What is 10 * 10",
            }]
    stream = client.stream_completion(messages)
    response = ""

    async for chunk in stream:
        print(chunk)
        # response += chunk
        # print(chunk, end = "")
    print()

if __name__ == "__main__":
    asyncio.run(main()) 