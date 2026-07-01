"""FastAPI server for nemoguardian."""

from __future__ import annotations

import asyncio
import datetime as dt
import os
from decimal import ROUND_CEILING, Decimal
from pathlib import Path
from typing import Annotated, ClassVar
from uuid import uuid4

import structlog
from fastapi import Body, Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, StreamingResponse

from nemoguardian.billing import auth as billing_auth
from nemoguardian.billing import checkout as billing_checkout
from nemoguardian.billing import db as billing_db
from nemoguardian.billing import metered as billing_metered
from nemoguardian.billing import provisioning as billing_provisioning
from nemoguardian.billing import webhook as billing_webhook
from nemoguardian.billing.plans import Tier, get_plan
from nemoguardian.billing.schemas import (
    CheckoutRequest,
    CheckoutResponse,
    CreateKeyRequest,
    CreateKeyResponse,
    GpuCreditBalanceResponse,
    GpuCreditCheckoutRequest,
    GpuCreditCheckoutResponse,
    GpuCreditCheckoutStatusResponse,
    GpuCreditEventResponse,
    PortalRequest,
    PortalResponse,
    ProvisioningRequest,
    ProvisioningResponse,
    UsageResponse,
)
from nemoguardian.cascade import Cascade, CascadeConfig
from nemoguardian.middleware import install_hardening
from nemoguardian.policy.nemoclaw import NemoclawPolicy
from nemoguardian.policy.presets import PRESETS, get_preset
from nemoguardian.providers import (
    ProviderName,
    offers_fitting_cascade,
)
from nemoguardian.providers import (
    list_providers as providers_list,
)
from nemoguardian.providers.base import CASCADE_VRAM_COMFORT_GB
from nemoguardian.providers.registry import default_registry as providers_registry
from nemoguardian.schemas import (
    HealthResponse,
    ModerateRequest,
    ModerateResponse,
    StreamChunk,
)

logger = structlog.get_logger("nemoguardian.server")


# --- Singleton wiring -----------------------------------------------------

class _State:
    cascade: Cascade | None = None
    policies: ClassVar[dict[str, NemoclawPolicy]] = {}

    @classmethod
    def ensure(cls) -> tuple[Cascade, dict[str, NemoclawPolicy]]:
        if cls.cascade is None:
            cls.cascade = Cascade(CascadeConfig.from_env())
        if not cls.policies:
            for name in PRESETS:
                cls.policies[name] = get_preset(name)
        return cls.cascade, cls.policies


def get_cascade() -> Cascade:
    cascade, _ = _State.ensure()
    return cascade


def get_policies() -> dict[str, NemoclawPolicy]:
    _, policies = _State.ensure()
    return policies


# --- App -------------------------------------------------------------------

app = FastAPI(
    title="nemoguardian",
    description="Multi-model LLM moderation cascade (Nemotron + Qwen3Guard).",
    version="0.1.0",
)

# Production hardening: env-configurable CORS allowlist (no wildcard by
# default), per-key/IP rate limiting, and a request body-size cap on the
# moderation endpoints. See nemoguardian.middleware for the config knobs.
install_hardening(app)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    cascade = get_cascade()
    try:
        import torch

        gpu_available = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if gpu_available else None
    except Exception:
        gpu_available = False
        gpu_name = None
    triage_status = cascade.triage_status()
    return HealthResponse(
        status="ok",
        models_loaded=cascade.loaded_models(),
        gpu_available=gpu_available,
        gpu_name=gpu_name,
        runtime_device=f"cuda: {gpu_name}" if gpu_available and gpu_name else "cpu",
        runtime_model_config=cascade.model_config_summary(),
        triage_configured=bool(triage_status["configured"]),
        triage_provider=triage_status["provider"] if isinstance(triage_status["provider"], str) else None,
        triage_status=triage_status,
    )


@app.post("/v1/moderate", response_model=ModerateResponse)
async def moderate(
    request: ModerateRequest,
    auth: billing_auth.AuthContext = Depends(billing_auth.require_api_key),
    policy_preset: Annotated[str | None, Query(description="Built-in policy preset")] = None,
    policy_yaml: Annotated[str | None, Query(description="Inline YAML policy")] = None,
) -> ModerateResponse:
    # Enforce tier limits + features
    billing_auth.enforce_feature(auth, f"cascade.{request.mode.value}")

    allowed, usage = billing_metered.check_allowance(auth.customer.id)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail={
                "error": "monthly allowance exceeded",
                "tier": auth.plan.tier.value,
                **usage,
                "upgrade_url": "/billing/checkout?plan=scale",
            },
        )

    cascade = get_cascade()
    policies = get_policies()

    policy_engine = _resolve_policy_engine(
        policies,
        policy_preset=policy_preset,
        policy_yaml=policy_yaml,
    )

    try:
        result = await asyncio.to_thread(cascade.moderate, request, policy_engine=policy_engine)
    except Exception as exc:
        logger.exception("moderation_failed", error=str(exc))
        raise HTTPException(500, f"moderation failed: {exc}") from exc

    billing_metered.report_usage(
        auth.customer.id,
        call_type=request.mode.value,
        request_id=result.request_id,
    )
    result.usage_info = usage  # type: ignore[attr-defined]
    return result


@app.post("/demo/moderate", response_model=ModerateResponse)
async def demo_moderate(
    request: ModerateRequest,
    policy_preset: Annotated[str | None, Query(description="Built-in policy preset")] = "discord",
    policy_yaml: Annotated[str | None, Query(description="Inline YAML policy")] = None,
) -> ModerateResponse:
    """Hackathon demo endpoint: real cascade, no billing/auth wrapper."""
    if not _demo_endpoint_enabled():
        raise HTTPException(404, "demo endpoint disabled")

    cascade = get_cascade()
    policy_engine = _resolve_policy_engine(
        get_policies(),
        policy_preset=policy_preset,
        policy_yaml=policy_yaml,
    )

    try:
        return await asyncio.to_thread(cascade.moderate, request, policy_engine=policy_engine)
    except Exception as exc:
        logger.exception("demo_moderation_failed", error=str(exc))
        raise HTTPException(500, f"demo moderation failed: {exc}") from exc


@app.post("/v1/moderate/stream")
async def moderate_stream(
    request: ModerateRequest,
    auth: billing_auth.AuthContext = Depends(billing_auth.require_api_key),
) -> StreamingResponse:
    """Streaming endpoint — yields per-token verdicts from Qwen3Guard-Stream.

    Returns newline-delimited JSON (NDJSON).
    """
    billing_auth.enforce_feature(auth, "cascade.stream")

    allowed, usage = billing_metered.check_allowance(auth.customer.id)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail={
                "error": "monthly allowance exceeded",
                "tier": auth.plan.tier.value,
                **usage,
                "upgrade_url": "/billing/checkout?plan=scale",
            },
        )

    cascade = get_cascade()
    request_id = f"stream_{uuid4().hex}"
    billing_metered.report_usage(
        auth.customer.id,
        call_type="stream",
        request_id=request_id,
    )

    async def gen():
        start = dt.datetime.now()
        try:
            for label, score, idx in cascade.stream_token_verdicts(request.text):
                chunk = StreamChunk(
                    token_index=idx,
                    partial_text=request.text[: idx + 1],
                    verdict_so_far=label,
                    score_so_far=score,
                    is_terminal=False,
                    latency_ms=0.0,
                )
                yield chunk.model_dump_json() + "\n"
                await asyncio.sleep(0)
            elapsed_ms = (dt.datetime.now() - start).total_seconds() * 1000.0
            terminal = StreamChunk(
                token_index=len(request.text),
                partial_text=request.text,
                verdict_so_far="safe",
                score_so_far=0.0,
                is_terminal=True,
                latency_ms=elapsed_ms,
            )
            yield terminal.model_dump_json() + "\n"
        except Exception as exc:
            logger.exception("stream_failed", error=str(exc))
            yield StreamChunk(
                token_index=0,
                partial_text=request.text,
                verdict_so_far="safe",
                score_so_far=0.0,
                is_terminal=True,
                latency_ms=0.0,
            ).model_dump_json() + "\n"

    return StreamingResponse(gen(), media_type="application/x-ndjson")


def _yaml_safe_load(s: str) -> dict:
    import yaml

    return yaml.safe_load(s) or {}


def _resolve_policy_engine(
    policies: dict[str, NemoclawPolicy],
    *,
    policy_preset: str | None,
    policy_yaml: str | None,
) -> NemoclawPolicy | None:
    if policy_preset:
        if policy_preset not in policies:
            raise HTTPException(400, f"unknown preset {policy_preset!r}; available: {list(policies)}")
        return policies[policy_preset]
    if policy_yaml:
        return NemoclawPolicy.from_dict(_yaml_safe_load(policy_yaml))
    return None


def _demo_endpoint_enabled() -> bool:
    return os.environ.get("NEMOGUARDIAN_ENABLE_DEMO_ENDPOINT", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


__all__ = ["app"]


# --- Billing endpoints -----------------------------------------------------


@app.post("/billing/checkout", response_model=CheckoutResponse)
async def billing_checkout_endpoint(req: CheckoutRequest) -> CheckoutResponse:
    """Create a Stripe Checkout session for the given tier. Falls back to demo mode."""
    tier = Tier(req.tier)
    session = billing_checkout.create_checkout_session(
        email=req.email,
        tier=tier,
        success_url=req.success_url,
        cancel_url=req.cancel_url,
    )
    customer = billing_db.get_customer_by_email(req.email)
    return CheckoutResponse(
        session_id=session.session_id,
        url=session.url,
        demo_mode=session.demo_mode,
        customer_id=customer.id if customer else 0,
    )


@app.post("/billing/gpu-credits/checkout", response_model=GpuCreditCheckoutResponse)
async def billing_gpu_credit_checkout(req: GpuCreditCheckoutRequest) -> GpuCreditCheckoutResponse:
    """Create a Stripe one-time checkout to fund GPU rental credits."""
    try:
        session = billing_checkout.create_gpu_credit_checkout_session(
            email=req.email,
            amount_cents=req.amount_cents,
            success_url=req.success_url,
            cancel_url=req.cancel_url,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(502, str(exc)) from exc
    return GpuCreditCheckoutResponse(
        session_id=session.session_id,
        url=session.url,
        demo_mode=session.demo_mode,
        customer_id=session.customer_id,
        amount_cents=session.amount_cents,
        balance_cents=session.balance_cents,
    )


@app.get("/billing/gpu-credits/checkout-status", response_model=GpuCreditCheckoutStatusResponse)
async def billing_gpu_credit_checkout_status(
    session_id: Annotated[str, Query(min_length=3, max_length=255)],
) -> GpuCreditCheckoutStatusResponse:
    """Show whether a Stripe GPU-credit Checkout session has reached the ledger."""
    event = billing_db.get_gpu_credit_event_by_checkout_session(session_id)
    if event is None:
        return GpuCreditCheckoutStatusResponse(session_id=session_id, credited=False)
    return GpuCreditCheckoutStatusResponse(
        session_id=session_id,
        credited=True,
        amount_cents=event.amount_cents,
        balance_cents=billing_db.gpu_credit_balance_cents(event.customer_id),
        currency=event.currency,
        event_id=event.id,
        occurred_at=event.occurred_at,
    )


@app.get("/billing/gpu-credits", response_model=GpuCreditBalanceResponse)
async def billing_gpu_credit_balance(
    auth: billing_auth.AuthContext = Depends(billing_auth.require_api_key),
    limit: Annotated[int, Query(ge=0, le=100)] = 25,
) -> GpuCreditBalanceResponse:
    """Show the authenticated customer's GPU credit balance and recent ledger events."""
    events = billing_db.list_gpu_credit_events(auth.customer.id, limit=limit)
    return GpuCreditBalanceResponse(
        customer_id=auth.customer.id,
        email=auth.customer.email,
        balance_cents=billing_db.gpu_credit_balance_cents(auth.customer.id),
        events=[
            GpuCreditEventResponse(
                id=event.id,
                event_type=event.event_type,
                amount_cents=event.amount_cents,
                currency=event.currency,
                provider=event.provider,
                job_id=event.job_id,
                stripe_checkout_session_id=event.stripe_checkout_session_id,
                stripe_payment_intent_id=event.stripe_payment_intent_id,
                description=event.description,
                occurred_at=event.occurred_at,
            )
            for event in events
        ],
    )


@app.post("/billing/webhook")
async def billing_webhook_endpoint(request: Request) -> dict:
    """Stripe webhook receiver (subscription lifecycle)."""
    return await billing_webhook.handle_stripe_webhook(request)


@app.post("/billing/portal", response_model=PortalResponse)
async def billing_portal_endpoint(
    req: PortalRequest,
    auth: billing_auth.AuthContext = Depends(billing_auth.require_api_key),
) -> PortalResponse:
    """Create a Stripe customer-portal session for plan changes and cancellation."""
    session = billing_checkout.create_portal_session(
        customer=auth.customer,
        return_url=req.return_url,
    )
    return PortalResponse(
        url=session.url,
        demo_mode=session.demo_mode,
        customer_id=auth.customer.id,
    )


@app.post("/billing/keys", response_model=CreateKeyResponse)
async def billing_create_key(
    req: CreateKeyRequest,
    auth: billing_auth.AuthContext = Depends(billing_auth.require_api_key),
) -> CreateKeyResponse:
    """Provision a new API key for the authenticated customer."""
    raw, _record = billing_db.create_api_key(auth.customer.id, label=req.label)
    return CreateKeyResponse(
        api_key=raw,
        label=req.label,
        tier=auth.plan.tier.value,  # type: ignore[arg-type]
    )


@app.get("/billing/usage", response_model=UsageResponse)
async def billing_usage(
    auth: billing_auth.AuthContext = Depends(billing_auth.require_api_key),
) -> UsageResponse:
    """Show this customer's current-period usage and overage projection."""
    _allowed, info = billing_metered.check_allowance(auth.customer.id)
    return UsageResponse(
        customer_id=auth.customer.id,
        email=auth.customer.email,
        tier=auth.plan.tier.value,  # type: ignore[arg-type]
        total_calls=info["total_calls"],
        allowance=info["allowance"],
        overage_calls=info["overage_calls"],
        overage_cents=info["overage_cents"],
        period_start=info["period_start"],
        period_end=info["period_end"],
    )


@app.post("/billing/provision", response_model=ProvisioningResponse)
async def billing_provision(
    req: ProvisioningRequest,
    auth: billing_auth.AuthContext = Depends(billing_auth.require_api_key),
) -> ProvisioningResponse:
    """Self-hosted tier: provision a nemoguardian instance for this customer.

    Routed through :func:`provision_instance_guarded` so the operator spend
    guardrails run before any live spend: an offer above the hourly-price cap or a
    reservation past the max-hours cap is **rejected** (HTTP 402, no provider
    call); a request that passes the caps but is not ``confirm``ed comes back as a
    priced ``planned`` dry-run (still no spend); only a confirmed, in-cap request
    calls the provider and goes ``live``.
    """
    if "deploy.self_hosted" not in auth.plan.features:
        raise HTTPException(
            status_code=402,
            detail="self-hosted provisioning requires the self_hosted plan; "
                   "upgrade at /billing/checkout?plan=self_hosted",
        )
    try:
        provider_enum = ProviderName(req.provider)
    except ValueError as exc:
        raise HTTPException(400, f"unknown provider: {req.provider!r}") from exc
    provider = providers_registry().get(provider_enum)

    offers = await provider.list_offers(
        gpu_model=req.gpu_model, max_price_usd=req.max_price_usd,
    )
    if not offers:
        raise HTTPException(404, f"no offers available on {req.provider} for that filter")
    offer = next((o for o in offers if o.offer_id == req.offer_id), None) if req.offer_id else None
    if offer is None:
        offer = min(offers, key=lambda o: o.price_per_hour_usd)

    env = {
        "NEMOGUARDIAN_API_KEY": auth.raw_key,
        "STRIPE_CUSTOMER_ID": auth.customer.stripe_customer_id or "",
        "NEMOGUARDIAN_CUSTOMER_ID": str(auth.customer.id),
        "CASCADE_MODE": "standard",
    }
    job = await billing_provisioning.provision_instance_guarded(
        customer_id=auth.customer.id,
        provider=req.provider,
        offer=offer,
        provider_client=provider,
        reserve_hours=req.reserve_hours,
        confirm=req.confirm,
        ssh_public_key=req.ssh_public_key,
        image=req.image,
        env=env,
    )
    if job.status == "rejected":
        # A spend cap refused the request — no provider call, no money spent.
        raise HTTPException(
            status_code=402,
            detail={
                "error": "spend_cap_exceeded",
                "message": job.error_message or "provision rejected by spend caps",
                "provider": req.provider,
                "hourly_price_usd": offer.price_per_hour_usd,
                "reserve_hours": req.reserve_hours,
            },
        )
    return ProvisioningResponse(
        job_id=job.id,
        status=job.status,
        instance_id=job.instance_id,
        endpoint_url=job.endpoint_url,
        ssh_command=job.ssh_command,
        error_message=job.error_message,
        provider=req.provider,
        hourly_price_usd=offer.price_per_hour_usd,
        reserve_hours=req.reserve_hours,
    )


@app.get("/billing/jobs/{job_id}", response_model=ProvisioningResponse)
async def billing_job_status(
    job_id: int,
    auth: billing_auth.AuthContext = Depends(billing_auth.require_api_key),
) -> ProvisioningResponse:
    """Check the status of a self-hosted provisioning job."""
    job = billing_db.get_provisioning_job(job_id)
    if job.customer_id != auth.customer.id:
        raise HTTPException(403, "not your job")
    return ProvisioningResponse(
        job_id=job.id,
        status=job.status,
        instance_id=job.instance_id,
        endpoint_url=job.endpoint_url,
        ssh_command=job.ssh_command,
        error_message=job.error_message,
    )


@app.get("/billing/plans")
async def billing_list_plans() -> dict:
    """Public — list all plans (used by the pricing page)."""
    return {
        "plans": [
            {
                "tier": plan.tier.value,
                "display_name": plan.display_name,
                "monthly_price_cents": plan.monthly_price_cents,
                "monthly_allowance": plan.monthly_allowance,
                "overage_per_call_cents": plan.overage_per_call_cents,
                "features": sorted(plan.features),
                "max_api_keys": plan.max_api_keys,
                "cascade_mode_default": plan.cascade_mode_default,
                "description": plan.description,
            }
            for plan in [get_plan(t) for t in (Tier.FREE, Tier.PRO, Tier.SCALE, Tier.SELF_HOSTED)]
        ]
    }


def _reset_billing_db_for_tests() -> None:
    """Drop + recreate the SQLite DB. Only used in tests."""
    from nemoguardian.billing import db as _db

    _db._conn = None
    _db.init_db()
    _db.init_db(_db.DEFAULT_DB_PATH)


# --- Provider catalog + cost comparison -----------------------------------


@app.get("/providers")
async def providers_catalog() -> dict:
    """Public — list all known GPU providers with catalog metadata."""
    return {
        "providers": [e.to_dict() for e in providers_list()],
    }


@app.get("/providers/offers")
async def providers_offers(
    gpu_model: Annotated[str | None, Query()] = None,
    max_price_usd: Annotated[float | None, Query()] = None,
    only_fits: Annotated[bool, Query(description="Only return offers that fit the cascade")] = True,
) -> dict:
    """List GPU offers across every provider, filtered and sorted by price.

    Used by the self-hosted provisioning UI to show the customer the cost
    comparison table. The `only_fits=true` filter excludes GPUs too small for
    the full cascade (e.g., RTX 3060 12GB).
    """
    reg = providers_registry()
    all_offers = []
    for provider in reg.all():
        try:
            all_offers.extend(await provider.list_offers(
                gpu_model=gpu_model, max_price_usd=max_price_usd,
            ))
        except Exception:
            continue
    if only_fits:
        all_offers = offers_fitting_cascade(all_offers)
    all_offers.sort(key=lambda o: o.price_per_hour_usd)
    return {
        "count": len(all_offers),
        "cascade_vram_requirement_gb": CASCADE_VRAM_COMFORT_GB,
        "offers": [o.to_dict() for o in all_offers],
    }


async def _select_cheapest_offer(*, max_price_usd: float | None = None):
    reg = providers_registry()
    all_offers = []
    for provider in reg.all():
        try:
            all_offers.extend(await provider.list_offers(max_price_usd=max_price_usd))
        except Exception:
            continue
    fits = offers_fitting_cascade(all_offers)
    if not fits:
        raise HTTPException(
            404,
            "no offers fit the cascade (need >=24GB VRAM)"
            + (f" under ${max_price_usd}/hr" if max_price_usd else ""),
        )
    return sorted(fits, key=lambda offer: offer.price_per_hour_usd)[0]


def _gpu_credit_required_cents(hourly_price_usd: float, reserve_hours: float) -> int:
    cents = Decimal(str(hourly_price_usd)) * Decimal(str(reserve_hours)) * Decimal("100")
    return max(1, int(cents.to_integral_value(rounding=ROUND_CEILING)))


@app.post("/billing/provision/cheapest")
async def billing_provision_cheapest(
    auth: billing_auth.AuthContext = Depends(billing_auth.require_api_key),
    ssh_public_key: Annotated[str | None, Body()] = None,
    image: Annotated[str, Body()] = "nemoguardian/self-hosted:latest",
    max_price_usd: Annotated[float | None, Body()] = None,
    reserve_hours: Annotated[float, Body(ge=0.25, le=168.0)] = 3.0,
) -> ProvisioningResponse:
    """Pick the cheapest viable offer across all providers and provision it.

    This is the headline demo endpoint: "give me the cheapest GPU that fits."
    """
    if "deploy.self_hosted" not in auth.plan.features:
        raise HTTPException(
            status_code=402,
            detail="self-hosted provisioning requires the self_hosted plan; "
                   "upgrade at /billing/checkout?plan=self_hosted",
        )
    offer = await _select_cheapest_offer(max_price_usd=max_price_usd)
    reserve_cents = _gpu_credit_required_cents(offer.price_per_hour_usd, reserve_hours)
    job = billing_db.create_provisioning_job(
        customer_id=auth.customer.id,
        tier=auth.plan.tier,
        provider=offer.provider.value,
    )
    try:
        billing_db.reserve_gpu_credits(
            customer_id=auth.customer.id,
            amount_cents=reserve_cents,
            provider=offer.provider.value,
            job_id=job.id,
            description=f"Reserve {reserve_hours:g}h for {offer.provider.value} {offer.gpu_model}",
            metadata={"offer": offer.to_dict(), "reserve_hours": reserve_hours},
        )
    except ValueError as exc:
        billing_db.update_provisioning_job(job.id, status="failed", error_message=str(exc))
        raise HTTPException(
            status_code=402,
            detail={
                "error": "insufficient_gpu_credits",
                "message": str(exc),
                "required_cents": reserve_cents,
                "balance_cents": billing_db.gpu_credit_balance_cents(auth.customer.id),
                "top_up_url": "/billing/gpu-credits/checkout",
            },
        ) from exc

    env = {
        "NEMOGUARDIAN_API_KEY": auth.raw_key,
        "STRIPE_CUSTOMER_ID": auth.customer.stripe_customer_id or "",
        "NEMOGUARDIAN_CUSTOMER_ID": str(auth.customer.id),
        "CASCADE_MODE": "standard",
    }
    try:
        instance = await providers_registry().get(offer.provider).provision(
            offer,
            ssh_public_key=ssh_public_key,
            image=image,
            env=env,
        )
    except Exception as exc:
        billing_db.record_gpu_credit_event(
            customer_id=auth.customer.id,
            event_type="provision_refund",
            amount_cents=reserve_cents,
            provider=offer.provider.value,
            job_id=job.id,
            description="Refund GPU credit reservation after provisioning failure",
            metadata={"error": str(exc)},
        )
        billing_db.update_provisioning_job(job.id, status="failed", error_message=str(exc))
        raise HTTPException(502, f"provisioning failed: {exc}") from exc

    billing_db.update_provisioning_job(
        job.id,
        status="live" if instance.state.value == "live" else "provisioning",
        instance_id=instance.instance_id,
        endpoint_url=instance.endpoint_url,
        ssh_command=instance.ssh_command,
    )

    return ProvisioningResponse(
        job_id=job.id,
        status="live" if instance.state.value == "live" else "provisioning",
        instance_id=instance.instance_id,
        endpoint_url=instance.endpoint_url,
        ssh_command=instance.ssh_command,
        provider=offer.provider.value,
        hourly_price_usd=offer.price_per_hour_usd,
        reserve_hours=reserve_hours,
        gpu_credit_reserved_cents=reserve_cents,
        gpu_credit_balance_cents=billing_db.gpu_credit_balance_cents(auth.customer.id),
    )


@app.post("/billing/provision/{provider_name}")
async def billing_provision_specific(
    provider_name: str,
    auth: billing_auth.AuthContext = Depends(billing_auth.require_api_key),
    offer_id: Annotated[str | None, Body()] = None,
    gpu_model: Annotated[str | None, Body()] = None,
    max_price_usd: Annotated[float | None, Body()] = None,
    ssh_public_key: Annotated[str | None, Body()] = None,
    image: Annotated[str, Body()] = "nemoguardian/self-hosted:latest",
    reserve_hours: Annotated[float, Body(ge=0.25, le=168.0)] = 3.0,
) -> ProvisioningResponse:
    """Provision on a specific provider (e.g. 'hetzner', 'digitalocean')."""
    if "deploy.self_hosted" not in auth.plan.features:
        raise HTTPException(
            status_code=402,
            detail="self-hosted provisioning requires the self_hosted plan",
        )
    try:
        provider_enum = ProviderName(provider_name)
    except ValueError as exc:
        raise HTTPException(400, f"unknown provider: {provider_name!r}") from exc
    provider = providers_registry().get(provider_enum)

    offers = await provider.list_offers(
        gpu_model=gpu_model, max_price_usd=max_price_usd,
    )
    fits = offers_fitting_cascade(offers)
    if not fits:
        raise HTTPException(404, f"no fitting offers on {provider_name} for that filter")
    offer = next((o for o in fits if o.offer_id == offer_id), fits[0])
    reserve_cents = _gpu_credit_required_cents(offer.price_per_hour_usd, reserve_hours)
    job = billing_db.create_provisioning_job(
        customer_id=auth.customer.id, tier=auth.plan.tier, provider=provider_name,
    )
    try:
        billing_db.reserve_gpu_credits(
            customer_id=auth.customer.id,
            amount_cents=reserve_cents,
            provider=provider_name,
            job_id=job.id,
            description=f"Reserve {reserve_hours:g}h for {provider_name} {offer.gpu_model}",
            metadata={"offer": offer.to_dict(), "reserve_hours": reserve_hours},
        )
    except ValueError as exc:
        billing_db.update_provisioning_job(job.id, status="failed", error_message=str(exc))
        raise HTTPException(
            status_code=402,
            detail={
                "error": "insufficient_gpu_credits",
                "message": str(exc),
                "required_cents": reserve_cents,
                "balance_cents": billing_db.gpu_credit_balance_cents(auth.customer.id),
                "top_up_url": "/billing/gpu-credits/checkout",
            },
        ) from exc

    env = {
        "NEMOGUARDIAN_API_KEY": auth.raw_key,
        "STRIPE_CUSTOMER_ID": auth.customer.stripe_customer_id or "",
        "NEMOGUARDIAN_CUSTOMER_ID": str(auth.customer.id),
        "CASCADE_MODE": "standard",
    }
    try:
        instance = await provider.provision(
            offer, ssh_public_key=ssh_public_key, image=image, env=env,
        )
    except Exception as exc:
        billing_db.record_gpu_credit_event(
            customer_id=auth.customer.id,
            event_type="provision_refund",
            amount_cents=reserve_cents,
            provider=provider_name,
            job_id=job.id,
            description="Refund GPU credit reservation after provisioning failure",
            metadata={"error": str(exc)},
        )
        billing_db.update_provisioning_job(job.id, status="failed", error_message=str(exc))
        raise HTTPException(502, f"{provider_name} provisioning failed: {exc}") from exc

    billing_db.update_provisioning_job(
        job.id,
        status="live" if instance.state.value == "live" else "provisioning",
        instance_id=instance.instance_id,
        endpoint_url=instance.endpoint_url,
        ssh_command=instance.ssh_command,
    )

    return ProvisioningResponse(
        job_id=job.id,
        status="live" if instance.state.value == "live" else "provisioning",
        instance_id=instance.instance_id,
        endpoint_url=instance.endpoint_url,
        ssh_command=instance.ssh_command,
        provider=provider_name,
        hourly_price_usd=offer.price_per_hour_usd,
        reserve_hours=reserve_hours,
        gpu_credit_reserved_cents=reserve_cents,
        gpu_credit_balance_cents=billing_db.gpu_credit_balance_cents(auth.customer.id),
    )


# --- Demo UI --------------------------------------------------------------

_DEMO_DIR = Path(__file__).resolve().parent.parent / "demo"


@app.get("/demo", response_class=FileResponse)
async def demo_ui() -> FileResponse:
    """Serve the static cost-comparison demo page."""
    return FileResponse(str(_DEMO_DIR / "index.html"))
