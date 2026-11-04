"""A scripted stand-in for the Claude API. No key, no network, no cost."""

from datetime import datetime, timedelta
from types import SimpleNamespace


def text(t):
    return SimpleNamespace(type="text", text=t)


def tool(name, data, id="tu_1"):
    return SimpleNamespace(type="tool_use", id=id, name=name, input=data)


def response(*blocks, stop="end_turn"):
    return SimpleNamespace(content=list(blocks), stop_reason=stop)


class FakeClaude:
    """`script` is a list of responses (or exceptions) returned in order."""

    def __init__(self, *script):
        self.script = list(script)
        self.calls = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    async def _create(self, **kwargs):
        # Copy the list: the engine keeps appending to the same history object.
        self.calls.append({**kwargs, "messages": list(kwargs["messages"])})
        if not self.script:
            raise AssertionError("FakeClaude: no scripted response left")
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class Clock:
    def __init__(self, start=datetime(2030, 1, 7, 12, 0)):
        self.now = start

    def __call__(self):
        return self.now

    def advance(self, **kw):
        self.now += timedelta(**kw)


class Operator:
    def __init__(self, fail=False):
        self.sent = []
        self.fail = fail

    async def __call__(self, text):
        if self.fail:
            raise RuntimeError("chat not found")
        self.sent.append(text)
