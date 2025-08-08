PYTHON ?= python

.PHONY: install-dev format lint lint-fix

install-dev:
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -r requirements-dev.txt

format:
	ruff format .

lint:
	ruff check .

lint-fix:
	ruff check . --fix
