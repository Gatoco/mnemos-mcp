.PHONY: install test run admin

install:
	pip install -e .

test:
	pytest -q

run:
	mcp-rag-opencode

admin:
	python -m mcp_rag.admin
