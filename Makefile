.PHONY: help setup setup-all lint format typecheck test check fsm-diagram detail-study benchmark benchmark-pretrained dev-api dev-web clean

PYTHON ?= /opt/homebrew/bin/python3.11
VENV := .venv
BIN := $(VENV)/bin

help:
	@echo "faro-cv developer commands:"
	@echo "  make setup                Create .venv and install core + dev dependencies"
	@echo "  make setup-all            Install full ML (PyTorch MPS) + FastAPI + dev stack"
	@echo "  make lint                 Run ruff linter and format check"
	@echo "  make format               Auto-format source, tests, and experiments with ruff"
	@echo "  make typecheck            Run strict mypy type checking"
	@echo "  make test                 Run pytest with coverage"
	@echo "  make check                Run lint + typecheck + test"
	@echo "  make fsm-diagram          Print canonical Mermaid state diagram from FSM"
	@echo "  make detail-study         Run controlled SVD vs. 2D-FFT denoising/detail study"
	@echo "  make benchmark            Run deterministic offline adversarial ablation benchmark"
	@echo "  make benchmark-pretrained Run full ImageNet-pretrained ablation benchmark"
	@echo "  make dev-api              Start FastAPI + WebSocket workbench backend on :8000"
	@echo "  make dev-web              Start Next.js zero-scroll workbench frontend on :3000"
	@echo "  make clean                Remove local build and tool caches"

$(VENV):
	$(PYTHON) -m venv $(VENV)

setup: $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -e ".[dev]"

setup-all: $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -e ".[all]"

lint:
	$(BIN)/ruff check src tests experiments
	$(BIN)/ruff format --check src tests experiments

format:
	$(BIN)/ruff format src tests experiments
	$(BIN)/ruff check --fix src tests experiments

typecheck:
	$(BIN)/mypy src tests experiments

test:
	$(BIN)/pytest --cov=faro_cv --cov-report=term-missing

check: lint typecheck test

fsm-diagram:
	$(BIN)/python -c "from faro_cv.core.fsm import ExperimentFSM; print(ExperimentFSM.export_mermaid())"

detail-study:
	$(BIN)/python experiments/detail_tradeoff_study.py

benchmark:
	$(BIN)/python experiments/benchmark.py --model resnet18 --offline --output-dir experiments/results

benchmark-pretrained:
	$(BIN)/python experiments/benchmark.py --model resnet18 --pretrained --output-dir experiments/results

dev-api:
	$(BIN)/uvicorn faro_cv.api.app:app --reload --host 127.0.0.1 --port 8000

dev-web:
	npm --prefix web run dev

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage dist build src/*.egg-info
