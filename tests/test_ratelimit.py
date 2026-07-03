"""Rate limiting (NMG-011) — bucket math and API enforcement."""

from __future__ import annotations

from nemoguardian.ratelimit import RateLimiter


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_bucket_allows_burst_up_to_capacity():
    clock = FakeClock()
    rl = RateLimiter(rpm=5, clock=clock)
    results = [rl.check("k")[0] for _ in range(5)]
    assert all(results)
    allowed, retry_after = rl.check("k")
    assert not allowed
    assert retry_after > 0


def test_bucket_refills_over_time():
    clock = FakeClock()
    rl = RateLimiter(rpm=60, clock=clock)  # 1 token/second
    for _ in range(60):
        assert rl.check("k")[0]
    assert not rl.check("k")[0]
    clock.advance(2.5)
    assert rl.check("k")[0]  # ~2.5 tokens refilled
    assert rl.check("k")[0]
    assert not rl.check("k")[0]


def test_retry_after_matches_refill_rate():
    clock = FakeClock()
    rl = RateLimiter(rpm=60, clock=clock)
    for _ in range(60):
        rl.check("k")
    _, retry_after = rl.check("k")
    assert 0.9 <= retry_after <= 1.1  # 1 token/second → ~1s wait


def test_buckets_are_per_key():
    clock = FakeClock()
    rl = RateLimiter(rpm=1, clock=clock)
    assert rl.check("a")[0]
    assert not rl.check("a")[0]
    assert rl.check("b")[0]  # different caller, fresh bucket


def test_zero_rpm_disables_limiting():
    rl = RateLimiter(rpm=0)
    assert all(rl.check("k")[0] for _ in range(1000))


def test_rpm_from_env(monkeypatch):
    monkeypatch.setenv("NEMOGUARDIAN_RATE_LIMIT_RPM", "2")
    clock = FakeClock()
    rl = RateLimiter(clock=clock)
    assert rl.check("k")[0]
    assert rl.check("k")[0]
    assert not rl.check("k")[0]
    monkeypatch.setenv("NEMOGUARDIAN_RATE_LIMIT_RPM", "not-a-number")
    assert rl.rpm == 60.0  # bad values fall back to the default
