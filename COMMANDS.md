# Commands Cheat Sheet

Quick reference for setup, linting, testing, and running the server. See [README.md](README.md) for full endpoint docs and architecture.

---

## Setup

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
pip install -r requirements-test.txt
```

## Redis (optional — app degrades gracefully without it)

```bash
docker-compose up -d redis
```

## Seed the vector DB

```bash
python scripts/seed.py
```

## Run the server

```bash
uvicorn main:app --reload
```

API docs: http://localhost:8000/docs

---

## Lint & Format (Ruff)

```bash
ruff check .              # lint
ruff check . --fix        # lint + auto-fix
ruff format --check .     # check formatting (what CI runs)
ruff format .             # auto-format
```

## Pre-commit

```bash
pre-commit install            # one-time, sets up the git hook
pre-commit run --all-files    # run all hooks against the whole repo
```

## Tests

```bash
pytest tests/ -v

# with coverage (what CI runs)
pytest --cov=app --cov=config --cov-report=term-missing --cov-report=xml

# smoke test — requires a running server
python scripts/smoke_test.py
python scripts/smoke_test.py http://localhost:8000
```

---

## CI Reference

`.github/workflows/ci.yml` runs, in order:

```bash
ruff check .
ruff format --check .
pytest --cov=app --cov=config --cov-report=term-missing --cov-report=xml
```
