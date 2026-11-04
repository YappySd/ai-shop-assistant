"""The real Anthropic SDK against a local fake API: checks what actually goes over the wire."""

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import anthropic

from aibot.assistant import Assistant
from aibot.leads import LeadStore
from tests.fakes import Clock, Operator


def message(content, stop):
    return {"id": "msg_1", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
            "content": content, "stop_reason": stop, "stop_sequence": None,
            "usage": {"input_tokens": 10, "output_tokens": 5}}


class FakeAPI:
    def __init__(self, replies):
        self.replies, self.requests = list(replies), []
        api = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                api.requests.append({"path": self.path, "headers": dict(self.headers), "body": body})
                data = json.dumps(api.replies.pop(0)).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"


def test_real_sdk_request_and_tool_round_trip():
    api = FakeAPI([
        message([{"type": "tool_use", "id": "toolu_1", "name": "save_lead",
                  "input": {"name": "Анна", "phone": "0501234567", "request": "чайник"}}], "tool_use"),
        message([{"type": "text", "text": "Заявку записал."}], "end_turn"),
    ])
    op = Operator()

    async def scenario():
        async with anthropic.AsyncAnthropic(api_key="sk-test-not-real", base_url=api.url, max_retries=0) as client:
            bot = Assistant(client, shop="ДомМаркет", knowledge="kb", leads=LeadStore(), notify_operator=op,
                            clock=Clock(), canary="CANARY-x")
            return await bot.reply(1, "anna", "Хочу чайник, Анна, 0501234567")

    try:
        reply = asyncio.run(scenario())
    finally:
        api.server.shutdown()

    assert reply.text == "Заявку записал." and reply.lead_id == 1 and op.sent
    first, second = api.requests
    assert first["path"].startswith("/v1/messages")
    assert "server-side-fallback-2026-07-01" in first["headers"].get("anthropic-beta", "")
    body = first["body"]
    assert body["model"] == "claude-opus-5-5" and body["fallbacks"] == "default"
    assert body["output_config"] == {"effort": "low"} and "thinking" not in body
    assert body["tools"][0]["strict"] is True
    # The assistant turn with tool_use goes back unchanged, followed by our tool_result.
    msgs = second["body"]["messages"]
    assert msgs[1]["content"][0]["id"] == "toolu_1"
    assert msgs[2]["content"][0]["type"] == "tool_result" and msgs[2]["content"][0]["tool_use_id"] == "toolu_1"
