from collections.abc import Callable
import json
import pathlib
import subprocess
import contextlib
import asyncio
import uvloop
import functools
from typing import Any
import typer
import orjson
import httpx
import httpx_ws
import wat

app = typer.Typer(pretty_exceptions_show_locals=False)

uvloop.install()

HOST = "127.0.0.1"
PORT = 8080

OPTIONS = {
    "temperature": 0.05,
    "top_p": 0.3,
    "num_predict": 512,
    "num_ctx": int(32e3),
}


@contextlib.asynccontextmanager
async def run_server(model: str) -> subprocess.Popen:
    cmd = f"llama-server -m {model} -fa -c 131072 -dt 0.05 -b 8192 -ub 512 --host {HOST} --port {PORT}"
    proc = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
    await asyncio.sleep(2)
    try:
        yield proc
    finally:
        proc.terminate()
        await asyncio.sleep(0.1)
        if proc.returncode is None:
            proc.kill()


def into_request(json_data: str) -> str:
    request = (
        f"POST /chat/completions HTTP/1.1\r\n"
        f"Host: {HOST}:{PORT}\r\n"
        f"Content-Type: application/json\r\n"
        f"Content-Length: {len(json_data)}\r\n"
        f"\r\n"
        f"{json_data}"
    )
    return request


class LlamaClient:
    def __init__(self, model_path: str, url: str = "/chat/completions"):
        base_url = f"http://{HOST}:{PORT}/v1"
        self.headers: dict[str, str] = {
            "Context-Type": "text/json",
            "Authorization": f"Bearer Bearer no-key",
        }
        self.url: str = base_url + url
        self.template: dict[str, Any] = {
            "model": str(model_path),
            "cache_prompt": True,
            "messages": [],
            "params": OPTIONS,
        }
        self.rx: asyncio.StreamReader | None = None
        self.tx: asyncio.StreamWriter | None = None

    async def connect(self):
        print("CONNECTING!")
        self.rx, self.tx = await asyncio.open_connection(HOST, PORT)
        print("DONE CONNECTED!")

    async def chat(self, msg: dict[str, Any]) -> asyncio.Future[dict[str, Any]]:
        print("WRITING!")
        # TODO
        self.tx.write()
        v = orjson.loads(await self.rx.read())
        print("DONE WRITING!")
        return v


async def run_client(model: str):
    client = LlamaClient(model)
    await client.connect()
    v = await client.chat({"user": "say hello world please."})
    print("CLIENT HERE")
    print(v)


def use_curl():
    meta = {
        "slot_id": 0,
        "temperature": 0.1,
        "n_keep": -1,
        "cache_prompt": True,
    }
    msgs = {
        "messages": [
            {"role": "system", "content": "you are a computer, beep boop."},
            {"role": "user", "content": "say hello world please."},
        ]
    }
    msg = meta | msgs

    async def stream_request():
        opts = {"base_url": f"http://localhost:{PORT}", "http2": True}
        async with httpx.AsyncClient(**opts) as client:
            resp = await client.post("/chat/completions", content=orjson.dumps(msg))
            js = resp.json()
            print(js["choices"][0]["message"]["content"])

    asyncio.run(stream_request())


@app.command()
def server():
    async def _server():
        # need to run server,
        # at some point later shut it down
        # ideally all in one process without threads
        # so server should probably be an async context manager
        # we want to clean up after any signals by killing the server
        model = pathlib.Path.home() / "Downloads/qwen2.5-coder-32b-instruct-q6_k.gguf"
        assert model.exists()
        model_str = str(model)
        async with run_server(model_str):
            await asyncio.sleep(30)

    asyncio.run(_server())


@app.command()
def client():
    use_curl()


if __name__ == "__main__":
    app()
