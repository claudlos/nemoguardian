# Contributing to nemoguardian

Thanks for your interest in nemoguardian — multi-model LLM moderation built on Qwen3Guard + NVIDIA Nemotron.

> **Heads up:** this repo was built for the Hermes Agent Accelerated Business Hackathon (NVIDIA × Stripe × Nous Research, June 30, 2026). The core architecture and Cascade API are stable, but billing, provider integrations, and the demo UI are still being polished. If you're proposing a large change, please open an issue first.

## Development setup

```bash
git clone https://github.com/claudlos/nemoguardian
cd nemoguardian
python3 -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"
make verify        # ruff + pytest
make serve         # boots the FastAPI service on :8000
```

The first moderation call downloads model weights from Hugging Face (Qwen3Guard-Gen-4B is ~8 GB FP16). For CI we run with `NEMOGUARDIAN_SKIP_PREDOWNLOAD=1` and mock the model wrappers.

## Project layout

```
nemoguardian/
├── server.py            ← FastAPI app, routes, CORS
├── cascade.py           ← orchestration: guard stack → triage → aggregator
├── aggregator.py        ← weighted multi-model verdict combiner
├── schemas.py           ← Pydantic request/response models
├── models/              ← lazy HF / OpenAI-compatible wrappers
├── policy/              ← NemoClaw YAML gate + per-platform presets
├── adapters/            ← Discord / Twitch / generic webhook
├── billing/             ← Stripe-style plans, auth, usage, webhooks
├── providers/           ← GPU provider registry (Vast.ai, stubs)
tests/                   ← pytest, asyncio_mode=auto
scripts/                 ← real_model_smoke, framework_smoke, demo_host_check
demo/                    ← browser moderation console (index.html)
docs/                    ← judge guide, runbook, video script, model catalog
```

## Coding conventions

- Python ≥ 3.10. Use `from __future__ import annotations` in every module.
- Lint with `ruff` (line-length 100, rules `E,F,W,I,B,UP,SIM,RUF`).
- Type hints throughout. `mypy` is advisory; PRs that relax a type need a comment.
- Logging via `structlog`, not `print`. Never log raw API keys or full text payloads.
- Pydantic v2 for all schemas. `model_config = ConfigDict(extra="forbid")` on user-facing inputs.
- Async I/O. FastAPI handlers are `async def`; model wrappers can be sync but must be invoked via `asyncio.to_thread`.
- Prefer small, composable functions in `cascade.py` / `aggregator.py`. The aggregator is the IP; tests should cover it densely.

## Pull request process

1. **Open an issue first** for any non-trivial change. Tag with `enhancement`, `bug`, `security`, or `docs`.
2. Branch off `main`. Use the PR template below.
3. Run `make verify` locally. CI will fail on lint or test errors.
4. Add tests for new behavior. Coverage on `cascade.py`, `aggregator.py`, and `nemoguardian/billing/` is enforced informally at >80%.
5. Update `docs/` if you change user-facing behavior, route signatures, env vars, or model defaults.
6. If your change touches billing, security, or model defaults, request review from `@claudlos` explicitly.

## Commit messages

Conventional commits, scoped to the touched area:

```
cascade: add token-level streaming guard to standard mode
billing: reject placeholder API keys at startup
docs: clarify GPU requirements for deep-mode triage
```

## Security

If you find a vulnerability, **do not open a public issue**. See [`SECURITY.md`](./SECURITY.md) for the disclosure policy and contact channel.

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0 — the same license as this repo.
