.PHONY: install-dev lint test verify serve smoke smoke-deep platform-smoke eval triage-api-smoke demo-check framework-smoke discord-env-setup discord-live-smoke discord-actor-scenario pre-submit-local final-submission-check docker-build docker-run link-check secret-scan build-check ci-guards

PYTHON ?= .venv/bin/python
PORT ?= 8000
IMAGE ?= nemoguardian/self-hosted:latest
DOCKER_BUILD_PROGRESS ?= plain
DEMO_BASE_URL ?= http://localhost:8000
DEMO_CHECK_FLAGS ?=
FRAMEWORK_SMOKE_FLAGS ?=
DISCORD_LIVE_SMOKE_FLAGS ?=
DISCORD_ACTOR_SCENARIO_FLAGS ?=
PLATFORM_SMOKE_FLAGS ?=
TRIAGE_API_SMOKE_FLAGS ?=
FINAL_CHECK_FLAGS ?=

install-dev:
	python3 -m venv --system-site-packages .venv
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e ".[dev]"

lint:
	$(PYTHON) -m ruff check .

test:
	$(PYTHON) -m pytest -q

verify: lint test

# Offline CI reproducibility guards (no network / GPU / secrets).
link-check:
	$(PYTHON) scripts/check_doc_links.py

secret-scan:
	$(PYTHON) scripts/secret_scan.py

build-check:
	$(PYTHON) scripts/build_check.py $(BUILD_CHECK_FLAGS)

ci-guards: link-check secret-scan build-check

serve:
	$(PYTHON) -m uvicorn nemoguardian.server:app --host 0.0.0.0 --port $(PORT)

smoke:
	$(PYTHON) scripts/real_model_smoke.py

smoke-deep:
	$(PYTHON) scripts/real_model_smoke.py --deep

# Offline umbrella smoke for every platform adapter (no GPU / network / secrets):
# drives each adapter through a synthetic event end-to-end and reports a
# per-platform pass/skip summary. See docs/LIVE_TEST_STRATEGY.md for the live tiers.
platform-smoke:
	$(PYTHON) scripts/platform_smoke.py $(PLATFORM_SMOKE_FLAGS)

eval:
	$(PYTHON) scripts/eval_benchmark.py $(EVAL_FLAGS)

triage-api-smoke:
	$(PYTHON) scripts/triage_api_smoke.py $(TRIAGE_API_SMOKE_FLAGS)

demo-check:
	$(PYTHON) scripts/demo_host_check.py --base-url $(DEMO_BASE_URL) $(DEMO_CHECK_FLAGS)

framework-smoke:
	$(PYTHON) scripts/framework_smoke.py --base-url $(DEMO_BASE_URL) $(FRAMEWORK_SMOKE_FLAGS)

discord-env-setup:
	bash scripts/setup_discord_live_env.sh

discord-live-smoke:
	$(PYTHON) scripts/discord_live_smoke.py $(DISCORD_LIVE_SMOKE_FLAGS)

discord-actor-scenario:
	$(PYTHON) scripts/discord_actor_scenario.py $(DISCORD_ACTOR_SCENARIO_FLAGS)

pre-submit-local:
	$(PYTHON) scripts/pre_submit_local.py --image $(IMAGE)

final-submission-check:
	$(PYTHON) scripts/final_submission_check.py $(FINAL_CHECK_FLAGS)

docker-build:
	docker build --progress=$(DOCKER_BUILD_PROGRESS) --build-arg NEMOGUARDIAN_SKIP_PREDOWNLOAD=1 -t $(IMAGE) .

docker-run:
	docker run --rm --gpus all --env-file .env -p $(PORT):8000 $(IMAGE)
