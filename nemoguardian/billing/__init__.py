"""Billing + subscriptions module.

Public API:
- plans: pricing tiers, allowance, overage rates
- db: SQLite-backed customers / api_keys / usage tables
- auth: API-key authentication dependency
- checkout: Stripe Checkout session creation
- webhook: Stripe webhook handler with signature verification
- metered: usage tracking + Stripe metered event reporting
- provisioning: self-hosted tier instance provisioning (stub)
"""
