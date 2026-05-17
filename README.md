# Multi-Agent System (MAS) with LangGraph

## Readings

1. **Documentation**: ["A Multi-Agent System for Interrogative Decomposition of Human Inquiries Using LLM-Backed Sparse Dynamic Graphs"](docs/README.md)
2. **Literature review**: [Literature review and research on related work](docs/02-literature-search.md)
3. **The initial prompt**: [The initial prompt](docs/01-prompt-original.md)

## Setup

To set up the development environment, run the following commands:

```bash
uv venv --python 3.12
source .venv/bin/activate
uv sync
```

## Add API Keys

```bash
export GEMINI_API_KEY="your-gemini-api-key"

export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY="your-langsmith-api-key"
export LANGCHAIN_PROJECT="mas-inquiry"
export LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
```

or write it in an `.env` file.


## Unit Tests

```bash
PYTHONPATH=. uv run pytest tests/
```


## CLI

```bash
python -m src.cli --query "Correct push-ups?"
python -m src.cli --query "Healthier options than burger and fries"
python -m src.cli --query "Research the current state of AI agents"
```


