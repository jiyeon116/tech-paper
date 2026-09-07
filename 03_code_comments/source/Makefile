PYTHON ?= python3.10
VENV ?= env_channel
VENV_PYTHON := $(VENV)/bin/python

.PHONY: setup test syntax status commands artifact-write artifact-verify clean

setup:
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -r requirements.txt

test:
	$(VENV_PYTHON) -m unittest discover -s tests -v

syntax:
	$(VENV_PYTHON) -m compileall -q algorithm_adapters channel_env qeco_runtime dynamic_experiment scripts tests *.py

status:
	$(VENV_PYTHON) scripts/qeco_workflow.py --family adapt2 status

commands:
	$(VENV_PYTHON) scripts/qeco_workflow.py --family adapt2 --users 300 commands

artifact-write:
	$(VENV_PYTHON) scripts/qeco_artifact_manifest.py write

artifact-verify:
	$(VENV_PYTHON) scripts/qeco_artifact_manifest.py verify

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
