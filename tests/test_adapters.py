"""Tests for platform adapter glue."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from nemoguardian.adapters import discord, twitch, webhook
from nemoguardian.schemas import VerdictLabel


class FakeCascade:
    def __init__(self, verdict: VerdictLabel, *, categories: list[str] | None = None) -> None:
        self.verdict = verdict
        self.categories = categories or []
        self.calls: list[dict[str, Any]] = []

    def moderate(self, request, *, policy_engine=None):
        self.calls.append(
            {
                "text": request.text,
                "mode": request.mode.value,
                "policy_engine": policy_engine is not None,
            }
        )
        return SimpleNamespace(verdict=self.verdict, categories=self.categories, score=0.9)


class FakeDiscordMessage:
    def __init__(self, content: str, *, bot: bool = False) -> None:
        self.content = content
        self.author = SimpleNamespace(bot=bot, mention="@tester")
        self.channel = SimpleNamespace(messages=[])
        self.deleted = False
        self.reactions: list[str] = []

        async def send(message: str) -> None:
            self.channel.messages.append(message)

        self.channel.send = send

    async def delete(self) -> None:
        self.deleted = True

    async def add_reaction(self, reaction: str) -> None:
        self.reactions.append(reaction)


async def test_discord_adapter_deletes_unsafe_message():
    cascade = FakeCascade(VerdictLabel.UNSAFE, categories=["pii"])
    message = FakeDiscordMessage("drop your SSN")

    await discord.make_handler(cascade)(message)

    assert message.deleted is True
    assert message.channel.messages
    assert "@tester" in message.channel.messages[0]
    assert "pii" in message.channel.messages[0]
    assert cascade.calls[0]["mode"] == "standard"


async def test_discord_adapter_reacts_to_controversial_message():
    cascade = FakeCascade(VerdictLabel.CONTROVERSIAL)
    message = FakeDiscordMessage("borderline")

    await discord.make_handler(cascade)(message)

    assert message.deleted is False
    assert message.reactions == [discord.WARNING_REACTION]


async def test_discord_adapter_ignores_bot_messages():
    cascade = FakeCascade(VerdictLabel.UNSAFE)
    message = FakeDiscordMessage("bot echo", bot=True)

    await discord.make_handler(cascade)(message)

    assert cascade.calls == []
    assert message.deleted is False


async def test_twitch_adapter_returns_delete_action():
    cascade = FakeCascade(VerdictLabel.UNSAFE)
    emitted: list[str] = []

    action = await twitch.make_moderator(cascade, emit=emitted.append)("drop your SSN")

    assert action == "delete"
    assert emitted
    assert cascade.calls[0]["mode"] == "fast"


class FakeHTTPResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        return

    def json(self) -> dict[str, Any]:
        return self.payload


class FakeHTTPClient:
    def __init__(self) -> None:
        self.posts: list[dict[str, Any]] = []

    async def post(self, url: str, **kwargs: Any) -> FakeHTTPResponse:
        self.posts.append({"url": url, **kwargs})
        if url.endswith("/v1/moderate"):
            return FakeHTTPResponse({"verdict": "unsafe", "score": 0.9})
        return FakeHTTPResponse({"ok": True})


async def test_webhook_adapter_sends_env_api_key(monkeypatch):
    monkeypatch.setenv("NEMOGUARDIAN_API_KEY", "nmg_env_key")
    client = FakeHTTPClient()

    verdict = await webhook.moderate_and_forward(
        "drop your SSN",
        forward_url="http://forward.test/hook",
        moderator_url="http://moderator.test",
        policy_preset="discord",
        client=client,
    )

    assert verdict["verdict"] == "unsafe"
    assert client.posts[0]["headers"] == {"Authorization": "Bearer nmg_env_key"}
    assert client.posts[0]["params"] == {"policy_preset": "discord"}
    assert client.posts[1]["json"] == {"text": "drop your SSN", "verdict": verdict}
