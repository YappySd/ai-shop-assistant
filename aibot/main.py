"""Entry point:  python -m aibot.main"""

from __future__ import annotations

import asyncio
import logging
import sys

import anthropic
from aiogram import Bot, Dispatcher
from dotenv import load_dotenv

from aibot.bot import build_router
from aibot.assistant import Assistant
from aibot.config import ConfigError, load_knowledge, load_settings
from aibot.leads import LeadStore

log = logging.getLogger("aibot")


async def amain() -> None:
    load_dotenv()
    s = load_settings()
    knowledge = load_knowledge(s.knowledge_path)
    bot = Bot(s.bot_token)  # no default parse_mode: everything is plain text
    leads = LeadStore(s.db_path)

    async def notify_operator(text: str) -> None:
        await bot.send_message(s.operator_chat_id, text, parse_mode=None)

    client = anthropic.AsyncAnthropic()  # reads ANTHROPIC_API_KEY from the environment
    assistant = Assistant(client, shop=s.shop_name, knowledge=knowledge, leads=leads,
                          notify_operator=notify_operator, model=s.model, effort=s.effort)
    dp = Dispatcher()
    dp.include_router(build_router(assistant, s.shop_name))
    log.info("ИИ-консультант запущен: %s, модель %s", s.shop_name, s.model)
    try:
        await dp.start_polling(bot)
    finally:
        await bot.session.close()
        await client.close()
        leads.close()


def run() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    try:
        asyncio.run(amain())
    except ConfigError as exc:
        sys.exit(f"Ошибка настройки: {exc}")
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    run()
