"""Telegram adapter (aiogram 3). Replies go out as plain text: no parse_mode, nothing to inject."""

from __future__ import annotations

from aiogram import F, Router
from aiogram.enums import ChatType
from aiogram.filters import CommandStart
from aiogram.types import Message

from aibot.assistant import Assistant

WELCOME = ("Здравствуйте! Я консультант магазина «{shop}». Спросите о товарах, доставке, оплате или возврате. "
           "Если нужно, позову менеджера.")


def build_router(assistant: Assistant, shop: str) -> Router:
    router = Router()
    # Only private chats: in groups strangers could talk to the bot on the shop's API budget.
    router.message.filter(F.chat.type == ChatType.PRIVATE)

    @router.message(CommandStart())
    async def start(message: Message) -> None:
        await message.answer(WELCOME.format(shop=shop), parse_mode=None)

    @router.message(F.text)
    async def text(message: Message) -> None:
        user = message.from_user
        await message.bot.send_chat_action(message.chat.id, "typing")
        reply = await assistant.reply(user.id, user.username or "", message.text)
        # Explicit None: even if someone sets a default parse_mode on the Bot, model text stays plain.
        await message.answer(reply.text, parse_mode=None)

    @router.message()
    async def other(message: Message) -> None:
        await message.answer("Я понимаю только текст. Напишите вопрос словами, пожалуйста.", parse_mode=None)

    return router
