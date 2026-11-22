# Contributing

Issues and pull requests are welcome.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

The tests need no network and no API keys. CI runs them on Python 3.11 and 3.12.

## Guidelines

- Open a pull request from a branch. Keep each one focused on a single change.
- Add or update a test for any behaviour you change.
- The scoring and critic steps must stay deterministic. Anything that calls an LLM needs a fallback and a test for the failure path.
- Do not commit real resumes or any personal data.
