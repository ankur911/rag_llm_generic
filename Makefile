# Makefile for RAG+LLM project
# Usage: `make <target>`  (run `make help` to list all)

PY ?= python
HOST ?= 127.0.0.1
PORT ?= 8000

.PHONY: help setup dev-setup venv build-vstore plan-a plan-b api eval test lint format typecheck clean freeze health

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | sed -E 's/:.*?##/: /' | sort

# -------------------- Setup --------------------

setup: ## Install runtime deps
	$(PY) -m pip install -U pip
	$(PY) -m pip install -r requirements.txt

dev-setup: ## Install runtime + dev deps
	$(PY) -m pip install -U pip
	$(PY) -m pip install -r requirements.txt -r requirements-dev.txt

venv: ## Create virtualenv at .venv (optional)
	$(PY) -m venv .venv
	@echo "Activate with: source .venv/bin/activate  (Windows: .venv\\Scripts\\activate)"

freeze: ## Freeze current env to requirements-lock.txt
	$(PY) -m pip freeze > requirements-lock.txt

# -------------------- App / API --------------------

build-vstore: ## Build FAISS vector store once (uses DOCUMENT_SOURCES)
	$(PY) -m app.main

plan-a: ## Run CLI test (PLAN-A)
	$(PY) -m app.main plan-a

plan-b: ## Run API smoke test (starts server, hits /health + /query, stops)
	$(PY) -m app.main plan-b

api: ## Run FastAPI locally with reload
	$(PY) -m uvicorn app.api:app --host $(HOST) --port $(PORT) --reload

health: ## GET /health (requires server running)
	curl -s http://$(HOST):$(PORT)/health | jq .

# -------------------- Evaluation --------------------

eval: ## Batch evaluation; writes rag_eval_results.jsonl
	$(PY) -m app.eval_rag

# -------------------- Quality --------------------

test: ## Run tests (pytest)
	pytest -q

lint: ## Lint (flake8)
	flake8 src app

format: ## Auto-format (black + isort)
	black src app
	isort src app

typecheck: ## Static type checks (mypy)
	mypy src app

clean: ## Remove caches, temp files, eval artifacts
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
	rm -rf .pytest_cache .mypy_cache
	rm -f rag_eval_results.jsonl
