# LEARNING NOTE: Makefile simplifica comandos comunes

.PHONY: help install run test clean

help:
	@echo "Available commands:"
	@echo "  make install    - Install dependencies"
	@echo "  make run       - Run the API locally"
	@echo "  make test      - Run tests"
	@echo "  make clean     - Clean cache files"

install:
	pip install -r requirements.txt

run:
	python -m backend.main

test:
	pytest backend/tests/ -v

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

format:
	black backend/
	isort backend/

lint:
	flake8 backend/
	mypy backend/