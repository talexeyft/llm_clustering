.PHONY: install dev-install test lint format clean run

install:
	pip install -r requirements.txt

dev-install:
	pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check src/ tests/
	mypy src/

format:
	black src/ tests/
	ruff check --fix src/ tests/

clean:
	find . -type d -name __pycache__ -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	rm -rf build/ dist/ *.egg-info .pytest_cache .mypy_cache .coverage htmlcov/

run:
	@if [ -z "$(INPUT)" ]; then \
		echo "Usage: make run INPUT=ai_data/raw_requests.csv [BATCH=batch-2024] [FORMAT=csv]"; \
		exit 1; \
	fi
	PYTHONPATH=src:$$PYTHONPATH python -m llm_clustering --input $(INPUT) --format $${FORMAT:-auto} --batch-id $${BATCH:-} --text-column $${TEXT_COL:-text}

