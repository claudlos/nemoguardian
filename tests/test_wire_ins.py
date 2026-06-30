"""Wire-in tests (night3): review-queue hookup + GPU guarded provisioning.

Part A — :class:`ModerationEngine` enqueues a controversial / flagged case to a
``ReviewService`` when one is injected (opt-in, default-safe).

Part B — :func:`provision_instance_guarded` routes the real provisioning call
through ``provision_guarded`` so spend caps + the explicit-confirm gate are
enforced before any live spend.
"""

from __future__ import annotations

import pytest

from nemoguardian.billing import db, provisioning
from nemoguardian.billing.plans import Tier
from nemoguardian.bot import (
    AuditLog,
    BotConfig,
    ModerationContext,
    ModerationEngine,
    Platform,
)
from nemoguardian.providers.base import (
    Instance,
    InstanceState,
    InstanceStatus,
    Offer,
    ProviderName,
)
from nemoguardian.providers.ops import GpuOpsConfig, ProvisionStatus
from nemoguardian.review import CaseStatus, ReviewService
from nemoguardian.schemas import Mode, ModerateResponse, VerdictLabel

# --------------------------------------------------------------------------- #
# Part A — review-queue hookup
# --------------------------------------------------------------------------- #


class _FakeCascade:
    """Stand-in cascade returning a fixed verdict (no models, no network)."""

    def __init__(self, response: ModerateResponse) -> None:
        self._response = response

    def moderate(self, request, *, policy_engine=None) -> ModerateResponse:
        return self._response


def _response(verdict: VerdictLabel, *, categories=None) -> ModerateResponse:
    return ModerateResponse(
        verdict=verdict,
        score=0.5,
        reasons=["test"],
        categories=categories or [],
        matched_policy_rule=None,
        model_verdicts={},
        total_latency_ms=1.0,
        mode=Mode.STANDARD,
        request_id=f"req-{verdict.value}",
    )


def _context(*, text: str = "borderline content") -> ModerationContext:
    return ModerationContext(
        platform=Platform.DISCORD,
        workspace_id="guild-1",
        channel_id="channel-1",
        message_id="message-1",
        user_id="user-1",
        username="tester",
        text=text,
    )


def _engine(tmp_path, verdict: VerdictLabel, *, service: ReviewService | None, categories=None):
    return ModerationEngine(
        Platform.DISCORD,
        cascade=_FakeCascade(_response(verdict, categories=categories)),
        audit_log=AuditLog(tmp_path / "audit.jsonl"),
        review_service=service,
    )


def test_controversial_verdict_enqueues_exactly_one_pending_case(tmp_path):
    service = ReviewService.from_dir(tmp_path)
    engine = _engine(tmp_path, VerdictLabel.CONTROVERSIAL, service=service, categories=["harassment"])
    config = BotConfig.default(Platform.DISCORD, "guild-1")

    evaluation = engine.evaluate(_context(), config)
    engine.record(evaluation, execution_status="flag")

    pending = service.queue.list_pending(platform=Platform.DISCORD, workspace_id="guild-1")
    assert len(pending) == 1
    case = pending[0]
    assert case.status is CaseStatus.PENDING
    assert case.verdict is VerdictLabel.CONTROVERSIAL
    assert case.categories == ["harassment"]
    # Raw text is never persisted — only a redacted excerpt + hash.
    assert case.text_sha256
    assert "borderline content" in case.text_excerpt  # no PII to redact here
    assert case.source_case_id == case.case_id


def test_safe_verdict_does_not_enqueue(tmp_path):
    service = ReviewService.from_dir(tmp_path)
    engine = _engine(tmp_path, VerdictLabel.SAFE, service=service)
    config = BotConfig.default(Platform.DISCORD, "guild-1")

    evaluation = engine.evaluate(_context(text="hello friends"), config)
    engine.record(evaluation, execution_status="allow")

    assert service.queue.list_pending() == []


def test_clear_unsafe_verdict_does_not_enqueue(tmp_path):
    service = ReviewService.from_dir(tmp_path)
    engine = _engine(tmp_path, VerdictLabel.UNSAFE, service=service, categories=["threats"])
    config = BotConfig.default(Platform.DISCORD, "guild-1")

    evaluation = engine.evaluate(_context(text="a clear violation"), config)
    engine.record(evaluation, execution_status="delete")

    # Clear-unsafe is auto-actioned (delete/timeout), not sent for human review.
    assert service.queue.list_pending() == []


def test_record_is_idempotent_for_same_case(tmp_path):
    service = ReviewService.from_dir(tmp_path)
    engine = _engine(tmp_path, VerdictLabel.CONTROVERSIAL, service=service)
    config = BotConfig.default(Platform.DISCORD, "guild-1")

    evaluation = engine.evaluate(_context(), config)
    engine.record(evaluation, execution_status="flag")
    engine.record(evaluation, execution_status="flag")

    assert len(service.queue.list_pending()) == 1


def test_no_review_service_is_a_no_op(tmp_path):
    # Backward-compatible default: without an injected service nothing is enqueued.
    service = ReviewService.from_dir(tmp_path)  # standalone store to inspect
    engine = _engine(tmp_path, VerdictLabel.CONTROVERSIAL, service=None)
    config = BotConfig.default(Platform.DISCORD, "guild-1")

    evaluation = engine.evaluate(_context(), config)
    engine.record(evaluation, execution_status="flag")

    assert service.queue.list_pending() == []


def test_review_queue_flag_disables_enqueue(tmp_path):
    service = ReviewService.from_dir(tmp_path)
    engine = _engine(tmp_path, VerdictLabel.CONTROVERSIAL, service=service)
    config = BotConfig.default(Platform.DISCORD, "guild-1")
    config.review_queue = False

    evaluation = engine.evaluate(_context(), config)
    engine.record(evaluation, execution_status="flag")

    assert service.queue.list_pending() == []


# --------------------------------------------------------------------------- #
# Part B — GPU guarded provisioning
# --------------------------------------------------------------------------- #


@pytest.fixture(autouse=True)
def _isolated_db(monkeypatch, tmp_path):
    db_path = tmp_path / "nmg_test.db"
    monkeypatch.setenv("NEMOGUARDIAN_DB_PATH", str(db_path))
    db._conn = None
    db.init_db(str(db_path))
    yield
    db._conn = None


class _MockProvider:
    """Records whether ``provision`` was actually called (i.e. money spent)."""

    name = ProviderName.VAST_AI

    def __init__(self) -> None:
        self.provision_calls = 0

    async def provision(self, offer, *, ssh_public_key=None, image="", env=None):
        self.provision_calls += 1
        return Instance(
            provider=self.name,
            instance_id="vast-live-1",
            gpu_model=offer.gpu_model,
            vram_gb=offer.vram_gb,
            region=offer.region,
            state=InstanceState.LIVE,
            ssh_command="ssh nemoguardian@vast-live-1",
            endpoint_url="https://vast-live-1.nemoguardian.dev",
            hourly_price_usd=offer.price_per_hour_usd,
        )

    async def status(self, instance_id):
        return InstanceStatus(instance_id=instance_id, state=InstanceState.LIVE)

    async def destroy(self, instance_id):
        return None


def _offer(price: float = 0.20) -> Offer:
    return Offer(
        provider=ProviderName.VAST_AI,
        gpu_model="RTX 3090",
        vram_gb=24,
        price_per_hour_usd=price,
        region="US",
        offer_id="offer-1",
    )


def _config(**overrides) -> GpuOpsConfig:
    base = dict(max_hourly_price_cents=50, max_reserve_hours=12.0, require_confirm=True)
    base.update(overrides)
    return GpuOpsConfig(**base)


async def test_over_cap_offer_is_rejected_without_spend():
    customer = db.upsert_customer(email="overcap@example.com")
    provider = _MockProvider()

    job = await provisioning.provision_instance_guarded(
        customer_id=customer.id,
        provider="vastai",
        offer=_offer(price=2.00),  # 200c/hr > 50c/hr cap
        provider_client=provider,
        reserve_hours=4.0,
        confirm=True,
        ops_config=_config(),
    )

    assert job.status == "rejected"
    assert "exceeds cap" in (job.error_message or "")
    assert provider.provision_calls == 0  # no live spend


async def test_over_hours_reservation_is_rejected_without_spend():
    customer = db.upsert_customer(email="overhours@example.com")
    provider = _MockProvider()

    job = await provisioning.provision_instance_guarded(
        customer_id=customer.id,
        provider="vastai",
        offer=_offer(price=0.20),
        provider_client=provider,
        reserve_hours=48.0,  # > 12h cap
        confirm=True,
        ops_config=_config(),
    )

    assert job.status == "rejected"
    assert "exceeds max" in (job.error_message or "")
    assert provider.provision_calls == 0


async def test_unconfirmed_in_cap_request_is_a_planned_dry_run():
    customer = db.upsert_customer(email="planned@example.com")
    provider = _MockProvider()

    job = await provisioning.provision_instance_guarded(
        customer_id=customer.id,
        provider="vastai",
        offer=_offer(price=0.20),
        provider_client=provider,
        reserve_hours=4.0,
        confirm=False,  # caps OK but not confirmed → no spend
        ops_config=_config(),
    )

    assert job.status == "planned"
    assert "dry-run" in (job.error_message or "")
    assert provider.provision_calls == 0  # still no live spend


async def test_confirmed_in_cap_request_provisions():
    customer = db.upsert_customer(email="proceed@example.com")
    provider = _MockProvider()

    job = await provisioning.provision_instance_guarded(
        customer_id=customer.id,
        provider="vastai",
        offer=_offer(price=0.20),
        provider_client=provider,
        reserve_hours=4.0,
        confirm=True,
        ops_config=_config(),
    )

    assert job.status == "live"
    assert job.instance_id == "vast-live-1"
    assert job.endpoint_url == "https://vast-live-1.nemoguardian.dev"
    assert provider.provision_calls == 1


async def test_guarded_job_returns_provision_result():
    customer = db.upsert_customer(email="result@example.com")
    provider = _MockProvider()
    job = db.create_provisioning_job(customer.id, tier=Tier.SELF_HOSTED, provider="vastai")

    result = await provisioning._run_guarded_job(
        job.id,
        _offer(price=0.20),
        provider_client=provider,
        reserve_hours=4.0,
        confirm=True,
        ops_config=_config(),
    )

    assert result.status is ProvisionStatus.PROVISIONED
    assert result.did_spend is True


async def test_guarded_provision_rejects_unknown_provider():
    customer = db.upsert_customer(email="badprovider@example.com")
    with pytest.raises(ValueError, match="unsupported provider"):
        await provisioning.provision_instance_guarded(
            customer_id=customer.id,
            provider="bogus",  # type: ignore[arg-type]
            offer=_offer(),
            provider_client=_MockProvider(),
            reserve_hours=1.0,
        )
