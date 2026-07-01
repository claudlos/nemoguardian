"""Pydantic schemas for the billing/subscription API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

TierName = Literal["free", "pro", "scale", "self_hosted"]


class CheckoutRequest(BaseModel):
    email: str = Field(..., min_length=3, max_length=254, description="Customer email address.")
    tier: TierName
    success_url: str = Field(default="https://nemoguardian.dev/billing/welcome")
    cancel_url: str = Field(default="https://nemoguardian.dev/billing")


class CheckoutResponse(BaseModel):
    session_id: str
    url: str
    demo_mode: bool
    customer_id: int


class GpuCreditCheckoutRequest(BaseModel):
    email: str = Field(..., min_length=3, max_length=254, description="Customer email address.")
    amount_cents: int = Field(..., ge=500, le=500_000, description="One-time GPU credit top-up.")
    success_url: str = Field(default="https://nemoguardian.dev/billing/gpu-credits/success")
    cancel_url: str = Field(default="https://nemoguardian.dev/billing/gpu-credits")


class GpuCreditCheckoutResponse(BaseModel):
    session_id: str
    url: str
    demo_mode: bool
    customer_id: int
    amount_cents: int
    balance_cents: int | None = None


class GpuCreditEventResponse(BaseModel):
    id: int
    event_type: str
    amount_cents: int
    currency: str
    provider: str | None = None
    job_id: int | None = None
    stripe_checkout_session_id: str | None = None
    stripe_payment_intent_id: str | None = None
    description: str | None = None
    occurred_at: str


class GpuCreditBalanceResponse(BaseModel):
    customer_id: int
    email: str
    balance_cents: int
    currency: str = "usd"
    events: list[GpuCreditEventResponse] = Field(default_factory=list)


class GpuCreditCheckoutStatusResponse(BaseModel):
    session_id: str
    credited: bool
    amount_cents: int | None = None
    balance_cents: int | None = None
    currency: str = "usd"
    event_id: int | None = None
    occurred_at: str | None = None


class PortalRequest(BaseModel):
    return_url: str = Field(default="https://nemoguardian.dev/billing")


class PortalResponse(BaseModel):
    url: str
    demo_mode: bool
    customer_id: int


class UsageResponse(BaseModel):
    customer_id: int
    email: str
    tier: TierName
    total_calls: int
    allowance: int
    overage_calls: int
    overage_cents: float
    period_start: str
    period_end: str


class CreateKeyRequest(BaseModel):
    label: str | None = None


class CreateKeyResponse(BaseModel):
    api_key: str = Field(..., description="Shown ONCE. Store securely.")
    label: str | None = None
    tier: TierName


class ProvisioningRequest(BaseModel):
    provider: Literal["vastai", "digitalocean", "lambda", "on_prem"] = "vastai"
    ssh_public_key: str | None = None
    image: str = "nemoguardian/self-hosted:latest"
    # Guarded-provisioning controls (audit #45). ``confirm`` is the no-auto-spend
    # gate: without it the endpoint returns a priced dry-run ("planned") and spends
    # nothing. The caps (max hourly price / max reserve hours) are enforced server
    # side regardless.
    confirm: bool = False
    reserve_hours: float = Field(default=3.0, ge=0.25, le=168.0)
    gpu_model: str | None = None
    max_price_usd: float | None = None
    offer_id: str | None = None


class ProvisioningResponse(BaseModel):
    job_id: int
    status: str
    instance_id: str | None = None
    endpoint_url: str | None = None
    ssh_command: str | None = None
    error_message: str | None = None
    provider: str | None = None
    hourly_price_usd: float | None = None
    reserve_hours: float | None = None
    gpu_credit_reserved_cents: int | None = None
    gpu_credit_balance_cents: int | None = None


__all__ = [
    "CheckoutRequest",
    "CheckoutResponse",
    "CreateKeyRequest",
    "CreateKeyResponse",
    "GpuCreditBalanceResponse",
    "GpuCreditCheckoutRequest",
    "GpuCreditCheckoutResponse",
    "GpuCreditCheckoutStatusResponse",
    "GpuCreditEventResponse",
    "PortalRequest",
    "PortalResponse",
    "ProvisioningRequest",
    "ProvisioningResponse",
    "TierName",
    "UsageResponse",
]
