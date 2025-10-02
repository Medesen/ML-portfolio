.PHONY: help build up down logs health metrics train-sample

COMPOSE := docker compose
SERVICE := churn-service
HOST ?= localhost
PORT ?= 8000
TOKEN ?= changeme

help:
	@echo "Available targets:"
	@echo "  build   - build images via docker compose"
	@echo "  up      - start the service detached"
	@echo "  down    - stop and remove containers"
	@echo "  logs    - follow service logs"
	@echo "  health  - curl GET /health (uses TOKEN, HOST, PORT)"
	@echo "  metrics - curl GET /metrics"
	@echo "  train-sample - run a quick local training using config.sample.json"

build:
	$(COMPOSE) build

up:
	$(COMPOSE) up -d

down:
	$(COMPOSE) down

logs:
	$(COMPOSE) logs -f $(SERVICE)

health:
	curl -fsS -H "Authorization: Bearer $(TOKEN)" http://$(HOST):$(PORT)/health || true

metrics:
	curl -fsS http://$(HOST):$(PORT)/metrics || true

train-sample:
	CHURN_ROOT=$(PWD) churn train --pos-label Yes --config config.sample.json --no-show-plots || true



