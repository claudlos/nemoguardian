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


class ProvisioningResponse(BaseModel):
    job_id: int
    status: str
    instance_id: str | None = None
    endpoint_url: str | None = None
    ssh_command: str | None = None
    error_message: str | None = None


__all__ = [
    "TierName",
    "CheckoutRequest",
    "CheckoutResponse",
    "UsageResponse",
    "CreateKeyRequest",
    "CreateKeyResponse",
    "ProvisioningRequest",
    "ProvisioningResponse",
]
