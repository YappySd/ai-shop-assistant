import asyncio

import anthropic
import httpx2

from aibot.assistant import (
    MAX_TURNS,
    TEXT_ERROR,
    TEXT_RATE,
    TEXT_REFUSAL,
    TEXT_RESET,
    TEXT_TOO_LONG,
    TOOLS,
    Assistant,
)
from aibot.leads import LeadStore
from tests.fakes import Clock, FakeClaude, Operator, response, text, tool

KB = "Доставка Новой почтой 1–3 дня. Возврат 14 дней."


def make(*script, operator=None, clock=None):
    client = FakeClaude(*script)
    op = operator or Operator()
    bot = Assistant(client, shop="ДомМаркет", knowledge=KB, leads=LeadStore(), notify_operator=op,
                    clock=clock or Clock(), canary="CANARY-test")
    return bot, client, op


def ask(bot, msg, user=1, username="anna"):
    return asyncio.run(bot.reply(user, username, msg))


def test_simple_answer_and_request_shape():
    bot, client, _ = make(response(text("Доставка 1–3 дня.")))
    assert ask(bot, "Сколько идёт доставка?").text == "Доставка 1–3 дня."
    call = client.calls[0]
    assert call["model"] == "claude-opus-5-5" and call["output_config"] == {"effort": "low"}
    assert call["fallbacks"] == "default" and call["betas"] == ["server-side-fallback-2026-07-01"]
    assert call["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert KB in call["system"][0]["text"] and "CANARY-test" in call["system"][0]["text"]
    assert all(t["strict"] and t["input_schema"]["additionalProperties"] is False for t in call["tools"])
    assert call["tools"] == TOOLS


def test_history_is_append_only_across_turns():
    bot, client, _ = make(response(text("Да")), response(text("14 дней")))
    ask(bot, "Есть доставка?")
    ask(bot, "А возврат?")
    first, second = client.calls[0]["messages"], client.calls[1]["messages"]
    assert second[: len(first)] == first  # earlier turns are never edited
    assert [m["role"] for m in second] == ["user", "assistant", "user"]


def test_lead_is_saved_and_manager_notified():
    bot, client, op = make(
        response(tool("save_lead", {"name": "Анна", "phone": "050 123 45 67", "request": "Набор кастрюль"}), stop="tool_use"),
        response(text("Записал, менеджер перезвонит.")),
    )
    r = ask(bot, "Хочу набор кастрюль, Анна, 050 123 45 67")
    assert r.lead_id == 1 and r.text == "Записал, менеджер перезвонит."
    assert "Телефон: +380501234567" in op.sent[0] and "@anna" in op.sent[0]
    tool_result = client.calls[1]["messages"][-1]["content"][0]
    assert tool_result["tool_use_id"] == "tu_1" and "сохранена" in tool_result["content"]


def test_bad_phone_goes_back_to_model_as_error():
    bot, client, op = make(
        response(tool("save_lead", {"name": "Анна", "phone": "позвоните", "request": "Чайник"}), stop="tool_use"),
        response(text("Напишите номер ещё раз, пожалуйста.")),
    )
    r = ask(bot, "Перезвоните мне")
    assert r.lead_id is None and op.sent == []
    result = client.calls[1]["messages"][-1]["content"][0]
    assert result["is_error"] is True and "телефон" in result["content"]


def test_operator_handoff():
    bot, _, op = make(
        response(tool("call_operator", {"reason": "Жалоба на брак"}), stop="tool_use"),
        response(text("Позвал менеджера.")),
    )
    r = ask(bot, "Пришла треснутая сковорода!")
    assert r.handed_off and "Жалоба на брак" in op.sent[0] and "треснутая" in op.sent[0]


def test_broken_operator_chat_does_not_break_reply():
    bot, _, _ = make(response(tool("call_operator", {"reason": "x"}), stop="tool_use"), response(text("Ок")),
                     operator=Operator(fail=True))
    assert ask(bot, "Позовите человека").text == "Ок"


def test_api_error_gives_polite_answer():
    err = anthropic.APIConnectionError(request=httpx2.Request("POST", "https://api.anthropic.com/v1/messages"))
    bot, _, _ = make(err)
    assert ask(bot, "Привет").text == TEXT_ERROR


def test_refusal_is_handled():
    bot, _, _ = make(response(stop="refusal"))
    assert ask(bot, "что-то").text == TEXT_REFUSAL


def test_input_limits():
    bot, client, _ = make()
    assert ask(bot, "x" * 1001).text == TEXT_TOO_LONG
    assert ask(bot, "   ").text.startswith("Напишите")
    assert client.calls == []  # nothing was sent to the paid API


def test_conversation_restarts_after_idle_and_after_max_turns():
    clock = Clock()
    replies = [response(text(str(i))) for i in range(MAX_TURNS + 2)]
    bot, client, _ = make(*replies, clock=clock)
    ask(bot, "первый")
    clock.advance(minutes=31)
    ask(bot, "после паузы")
    assert len(client.calls[1]["messages"]) == 1  # fresh conversation
    for i in range(MAX_TURNS - 1):
        clock.advance(seconds=31)  # stay under the rate limit
        ask(bot, f"вопрос {i}")
    clock.advance(seconds=31)
    assert ask(bot, "ещё один").text == TEXT_RESET


def test_rate_limit_per_user():
    bot, _, _ = make(*[response(text("ок")) for _ in range(25)])
    answers = [ask(bot, f"q{i}").text for i in range(21)]
    assert answers[-1] == TEXT_RATE and answers[19] == "ок"
    assert ask(bot, "другой пользователь", user=2).text == "ок"
