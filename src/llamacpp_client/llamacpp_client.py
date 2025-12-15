import asyncio
import json
from collections.abc import AsyncGenerator

from aiohttp import ClientSession

DEFAULT_BASE_URL = "http://localhost:8080"

DEFAULT_HEADERS = {
    "User-Agent": "aiohttp",
    "Content-Type": "application/json",
    "Accept": "text/event-stream",
    "Connection": "keep-alive",
}

DEFAULT_OPTIONS = {
    "model": "gpt-oss-120b",
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 40,
    "repeat_penalty": 1.05,
    "max_tokens": 4096,
    "stream": True,
    "stop": ["<|eot_id|>", "</s>"],
}


class LlamaCppAPIClient:
    def __init__(
        self, base_url: str = DEFAULT_BASE_URL, headers: dict = {}, options: dict = {}
    ):
        self.base_url = base_url
        self.headers = DEFAULT_HEADERS.copy()
        self.headers.update(headers)
        self.options = DEFAULT_OPTIONS.copy()
        self.options.update(options)

    async def stream_completion(
        self, messages: list[dict], json_schema: dict | None = None
    ) -> AsyncGenerator[dict, None]:
        url = self.base_url.rstrip("/") + "/v1/chat/completions"

        body = {
            "model": self.options["model"],
            "messages": messages,
            "temperature": self.options["temperature"],
            "top_p": self.options["top_p"],
            "top_k": self.options["top_k"],
            "repeat_penalty": self.options["repeat_penalty"],
            "max_tokens": self.options["max_tokens"],
            "stream": self.options["stream"],
            "json_schema": json_schema,
            "stop": self.options["stop"],
        }

        async with ClientSession() as session:
            async with session.post(url, headers=self.headers, json=body) as resp:
                if resp.status != 200:
                    raise Exception(f"HTTP error {resp.status}")

                async for raw in resp.content:
                    if not raw:
                        continue
                    try:
                        line = raw.decode("utf-8")
                        if line.startswith("data: "):
                            line = line[6:]
                        if line.strip():
                            yield json.loads(line)
                    except json.JSONDecodeError:
                        continue


async def test():
    client = LlamaCppAPIClient(base_url="http://10.30.0.164:8080")

    messages = [{"role": "user", "content": "Explain why the sky looks blue."}]

    async for chunk in client.stream_completion(messages):
        delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content")
        if delta:
            print(delta, end="", flush=True)


if __name__ == "__main__":
    asyncio.run(test())
