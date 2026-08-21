.PHONY: install test smoke run admin

install:
	pip install -e .

test:
	pytest -q

smoke:
	python scripts/smoke.py

run:
	mcp-rag-opencode

admin:
	python -m mcp_rag.admin
