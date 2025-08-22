.PHONY: test lint format security docs

test:
	python -m pytest tests/ -v --cov=src --cov-report=term

test-html:
	python -m pytest tests/ -v --cov=src --cov-report=html
	open htmlcov/index.html

lint:
	black --check src/ tests/ scripts/
	flake8 src/ tests/ scripts/ --max-line-length=88 --ignore=E203,W503
	isort --check-only src/ tests/ scripts/

format:
	black src/ tests/ scripts/
	isort src/ tests/ scripts/

security:
	safety check --full-report

docs:
	pdoc --html src --output-dir docs --force

install:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt

ci: lint test security