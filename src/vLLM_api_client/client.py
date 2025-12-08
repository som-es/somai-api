import asyncio
import json
import sys
from collections.abc import AsyncGenerator

from aiohttp import ClientSession

# Default Address and Port of your Triton HTTP Server
DEFAULT_BASE_URL = "http://localhost:8083"

# Default headers for HTTP POSTs/GETs
DEFAULT_HEADERS = {
    "User-Agent": "aiohttp",
    "Content-Type": "application/json",
    "Connection": "keep-alive",
    "Accept": "text/event-stream",
}

# Default options for Triton API endpoint
DEFAULT_COMPLETION_OPTIONS = {
    # "model": "/sicbert/models/Qwen2.5-14B-Instruct",
    "model": "models/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218",
    # "model": "/sicbert/models/DeepSeek-R1-Distill-Qwen-32B-AWQ",
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 40,
    "repetition_penalty": 1.05,
    "max_tokens": 1024 * 10,
    "stream": True,
    "stop": ["<|eot_id|>", "<|im_end|>", "<|endoftext|>", "<|end|>", "</s>"],
}


class vLLMAPIClient:
    """headers and options can be overriden at constructions time or per inference call"""

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        headers: dict = {},
        options: dict = {},
        model=None,
        stop=None,
    ):
        # override defaults with whatever userland passes into constructor
        self.base_url = base_url
        self.headers = DEFAULT_HEADERS
        self.headers.update(headers)
        self.options = DEFAULT_COMPLETION_OPTIONS
        self.options.update(options)
        if model:
            self.options["model"] = model

        if stop:
            self.options["stop"] = stop

    async def stream_completion(
        self, chat_thread: list[dict] = [], format: str = "ChatML"
    ) -> AsyncGenerator[dict, None]:
        """Stream Triton HTTP Server API POST /v1/chat/completions responses"""
        try:
            # set the HTTP headers and API options
            url = self.base_url.rstrip("/") + "/v1/chat/completions"
            combined_headers = self.headers

            # Create request body in Triton format
            request_body = {
                "messages": chat_thread,
                "model": self.options["model"],
                "temperature": self.options["temperature"],
                "top_p": self.options["top_p"],
                "repetition_penalty": self.options["repetition_penalty"],
                "max_tokens": self.options["max_tokens"],
                "stream": self.options["stream"],
                "stop": self.options["stop"],
                "top_k": self.options["top_k"],
            }

            async with ClientSession() as session:
                async with session.post(
                    url=url, headers=combined_headers, json=request_body
                ) as response:
                    if not response.status == 200:
                        raise Exception(f"HTTP Response: {response.status}")

                    async for raw_line in response.content:
                        if len(raw_line) == 0:
                            continue
                        try:
                            line = raw_line.decode("utf-8")
                            if line.startswith("data: "):
                                line = line[6:]  # Remove 'data: ' prefix
                            if line.strip():  # Skip empty lines
                                yield json.loads(line)
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            raise e


async def test():
    client = vLLMAPIClient(base_url="http://10.30.0.164:8080")

    messages = [{"role": "user", "content": "Explain why the sky looks blue."}]

    async for chunk in client.stream_completion(messages):
        delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content")
        if delta:
            print(delta, end="", flush=True)


if __name__ == "__main__":
    asyncio.run(test())
