from collections.abc import Callable
import os
from enum import StrEnum
import sys
import json
import pathlib
import subprocess
import contextlib
import asyncio
import uvloop
import functools
from typing import Any, Protocol, Literal
import typer
import orjson
import httpx
import httpx_ws
import wat
from loguru import logger

app = typer.Typer(pretty_exceptions_show_locals=False)

HOST = "127.0.0.1"
PORT = 8080

OPTIONS = {
    "temperature": 0.05,
    "top_p": 0.3,
    "num_predict": 512,
    "num_ctx": int(32e3),
    "n_keep": -1,
    "cache_prompt": True,
}


@contextlib.asynccontextmanager
async def run_server_process(model: str) -> asyncio.Future[subprocess.Popen]:
    cmd = f"llama-server -m {model} -fa -c 131072 -dt 0.05 -b 8192 -ub 512 --host {HOST} --port {PORT}"
    proc = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        yield proc
    finally:
        stdout, stderr = proc.communicate(timeout=1)
        proc.terminate()
        await asyncio.sleep(0.1)
        if proc.returncode is None or proc.returncode:
            proc.kill()


class InputStream(Protocol):
    async def read(self) -> str | None: ...


class OutputStream(Protocol):
    async def write(self, s: str) -> None: ...


async def llama_request(msg: dict[str, Any], client: httpx.AsyncClient) -> str:
    resp = await client.post("/chat/completions", content=orjson.dumps(msg))
    js = resp.json()
    try:
        content = js["choices"][0]["message"]["content"]
    except KeyError as e:
        print(content)
        raise e
    return content


class Client:
    def __init__(self, model_name: str, source: InputStream, drain: OutputStream):
        self.source = source
        self.drain = drain
        self.signal = asyncio.Event()
        self.config = {
            "model": model_name,
            "cache_prompt": True,
            "messages": [],
            "params": OPTIONS,
        }

    async def terminate(self):
        self.signal.set()

    async def completed(self):
        return self.signal.is_set()

    async def run(self):
        opts = {"base_url": f"http://localhost:{PORT}", "http2": True}
        async with httpx.AsyncClient(**opts) as client:
            while not self.signal.is_set():
                call = await self.source.read()
                if call is None:
                    break
                msg = {"role": "user", "content": call}
                self.config["messages"].append(msg)
                response = await llama_request(self.config, client)
                await self.drain.write(response)


class StdinStream:
    async def read(self) -> str | None:
        _quit = {"\n", "!!quit\n"}
        s = await asyncio.get_event_loop().run_in_executor(None, sys.stdin.readline)
        logger.trace(f"got {s=} {type(s)=} from stdin")
        if s in _quit:
            return None
        return s


class StdoutStream:
    async def write(self, s: str) -> None:
        return await asyncio.get_event_loop().run_in_executor(None, print, s)


async def setup_server(model: pathlib.Path, source: InputStream, drain: OutputStream):
    # need to run server,
    # at some point later shut it down
    # ideally all in one process without threads
    # so server should probably be an async context manager
    # we want to clean up after any signals by killing the server
    assert model.exists()
    async with run_server_process(str(model)) as proc:
        c = Client(model.name, source, drain)
        await c.run()
        proc.kill()


class Mode(StrEnum):
    stdin = "stdin"
    socket = "socket"
    ip = "ip"


def debug_mode():
    logger.remove()
    logger.add(sys.stderr, level="TRACE")


@app.command()
def run(mode: Mode = "stdin", model: pathlib.Path | None = None):
    if os.getenv("DEBUG"):
        debug_mode()
    if model is None:
        model = pathlib.Path.home() / "Downloads/qwen2.5-coder-32b-instruct-q6_k.gguf"
    match mode:
        case "stdin":
            source = StdinStream()
            drain = StdoutStream()
        case "socket":
            raise NotImplementedError()
        case "ip":
            raise NotImplementedError()
        case _:
            raise RuntimeError("Not a valid mode")
    if not model.exists():
        raise RuntimeError("Not a valid model")
    uvloop.run(setup_server(model, source, drain))


# how do I want to interact with this? I want to be able to set up a server process that I can shutdown with some
# kind of handle from the client without using some kind of hacky parsing logic.
# I could set up a daemon, but then I would need to use sudo, which, I really dont want to do for unpriveldiged server code.
# so, ideally the client is a service, which sets ups the server it's going to talk to, takes in some amount of requests, initally from input() for testing, need to allow it to work in headless mode so that  Ican run tests transparently too. So the api needs to be just submitting strings to be sent as input with some special quit message to terminate the process entirely.
# maybe !!quit
# we also need some way of tunning tests against the server using it as a pytest fixture that can run for the whole session so we need a function that can be used in main or as a context manager in a  fixture


# @app.command()
# def client():
#     _client()


if __name__ == "__main__":
    app()
