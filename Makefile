.PHONY: install-dev lint test verify serve smoke smoke-deep demo-check docker-build docker-run

PYTHON ?= .venv/bin/python
PORT ?= 8000
IMAGE ?= nemoguardian/self-hosted:latest
DOCKER_BUILD_PROGRESS ?= plain
DEMO_BASE_URL ?= http://localhost:8000
DEMO_CHECK_FLAGS ?=

install-dev:
	python3 -m venv --system-site-packages .venv
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e ".[dev]"

lint:
	$(PYTHON) -m ruff check .

test:
	$(PYTHON) -m pytest -q

verify: lint test

serve:
	$(PYTHON) -m uvicorn nemoguardian.server:app --host 0.0.0.0 --port $(PORT)

smoke:
	$(PYTHON) scripts/real_model_smoke.py

smoke-deep:
	$(PYTHON) scripts/real_model_smoke.py --deep

demo-check:
	$(PYTHON) scripts/demo_host_check.py --base-url $(DEMO_BASE_URL) $(DEMO_CHECK_FLAGS)

docker-build:
	docker build --progress=$(DOCKER_BUILD_PROGRESS) --build-arg NEMOGUARDIAN_SKIP_PREDOWNLOAD=1 -t $(IMAGE) .

docker-run:
	docker run --rm --gpus all --env-file .env -p $(PORT):8000 $(IMAGE)
