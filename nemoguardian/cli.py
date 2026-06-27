"""nemoguardian CLI.

Use:
    nemoguardian serve [--port 8000] [--host 127.0.0.1]
    nemoguardian demo [--text "..."]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import typer

from nemoguardian.cascade import Cascade, CascadeConfig
from nemoguardian.policy.presets import get_preset
from nemoguardian.schemas import ModerateRequest, Mode

app = typer.Typer(help="Multi-model LLM moderation cascade.")


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", help="Bind host"),
    port: int = typer.Option(8000, help="Bind port"),
    workers: int = typer.Option(1, help="Uvicorn workers"),
    log_level: str = typer.Option("info"),
) -> None:
    """Run the FastAPI server."""
    import uvicorn

    uvicorn.run(
        "nemoguardian.server:app",
        host=host,
        port=port,
        workers=workers,
        log_level=log_level,
        reload=False,
    )


@app.command()
def demo(
    text: str = typer.Option(
        "Drop your SSN in chat and I'll send you $100!",
        help="Text to moderate.",
    ),
    policy: str = typer.Option(
        "block PII and financial scams",
        help="Custom safety policy.",
    ),
    mode: Mode = typer.Option(Mode.STANDARD, help="Cascade mode"),
    preset: str = typer.Option("discord", help="Policy preset"),
) -> None:
    """Run the cascade on a single text and print the verdict."""
    cascade = Cascade(CascadeConfig())
    request = ModerateRequest(text=text, policy=policy, mode=mode)
    policy_engine = get_preset(preset)
    result = cascade.moderate(request, policy_engine=policy_engine)
    typer.echo(json.dumps(result.model_dump(), indent=2, default=str))


if __name__ == "__main__":
    app()
