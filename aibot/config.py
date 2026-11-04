"""Settings from .env and the shop's knowledge base file."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

KNOWLEDGE_MAX = 60_000  # characters; a bigger file would make every request expensive


class ConfigError(RuntimeError):
    pass


@dataclass(frozen=True)
class Settings:
    bot_token: str = field(repr=False)
    operator_chat_id: int
    shop_name: str
    knowledge_path: str
    model: str
    effort: str
    db_path: str


def load_settings(env=os.environ) -> Settings:
    token = env.get("BOT_TOKEN", "").strip()
    if not token:
        raise ConfigError("BOT_TOKEN не задан. Скопируйте .env.example в .env.")
    if not env.get("ANTHROPIC_API_KEY", "").strip():
        raise ConfigError("ANTHROPIC_API_KEY не задан: ключ берётся в console.anthropic.com → API Keys.")
    try:
        operator = int(env.get("OPERATOR_CHAT_ID", "").strip())
    except ValueError as exc:
        raise ConfigError("OPERATOR_CHAT_ID: числовой id чата менеджера (узнать: @userinfobot)") from exc
    effort = env.get("CLAUDE_EFFORT", "low").strip()
    if effort not in ("low", "medium", "high", "xhigh", "max"):
        raise ConfigError("CLAUDE_EFFORT: low, medium, high, xhigh или max")
    return Settings(
        bot_token=token,
        operator_chat_id=operator,
        shop_name=env.get("SHOP_NAME", "ДомМаркет").strip() or "ДомМаркет",
        knowledge_path=env.get("KNOWLEDGE_PATH", "knowledge.md"),
        model=env.get("CLAUDE_MODEL", "claude-opus-5-5").strip(),
        effort=effort,
        db_path=env.get("DB_PATH", "leads.db"),
    )


def load_knowledge(path: str | Path) -> str:
    try:
        text = Path(path).read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ConfigError(f"Нет {path}: скопируйте knowledge.example.md в knowledge.md и опишите магазин") from exc
    if len(text) > KNOWLEDGE_MAX:
        raise ConfigError(f"{path} больше {KNOWLEDGE_MAX} символов: сократите, иначе каждый ответ будет дорогим")
    if "</knowledge_base>" in text:
        raise ConfigError(f"{path}: в тексте не должно быть служебного тега </knowledge_base>")
    return text.strip()
