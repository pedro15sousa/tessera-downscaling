# Developer entry points. Everything runs through uv (see pyproject.toml).
.PHONY: setup-env test format figures tables

setup-env:            ## create/refresh .venv with the dev tools and the data-ingest extras
	uv sync --group dev --extra ingest

test:                 ## run the unit tests
	uv run pytest -q

format:               ## run every pre-commit hook (ruff check/format, nbstripout, ...) on the tree
	uv run pre-commit run --all-files

figures:              ## regenerate paper/figures/*.pdf from the runs under $TESSERA_DATA_ROOT
	uv run python scripts/paper/make_paper_figures.py

tables:               ## regenerate the paper's LaTeX tables from the runs under $TESSERA_DATA_ROOT
	uv run python scripts/paper/make_paper_tables.py
