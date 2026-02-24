---
name: python_coding_standards
description: Global coding standards, Python conventions, and code review guidelines.
---

# Python Coding Standards & Best Practices

This skill outlines the mandatory coding standards, architecture patterns, and review guidelines for Python development in this repository.

## 1. Technology Stack & Environment

- **Language:** Python
- **Framework:** LangGraph (when applicable)
- **Testing:** Pytest
- **Dependency Management:** `pyproject.toml` managed by `uv`

## 2. Python Specific Conventions

### Naming
- **Variables/Functions:** `snake_case`
- **Classes/Types:** `PascalCase`
- **Constants:** `UPPER_SNAKE_CASE`
- **File Names:** `kebab-case` or `snake_case`
- **Private Members:** leading underscore `_`

### Style & Typing
- **Type Hints:** Use type hints for function parameters and return values.
- **Modern Syntax:** Use Python 3.10+ syntax (e.g., `str | None` instead of `Optional[str]`).
- **Line Length:** 88 characters (Black standard).
- **Strings:** Use f-strings for formatting.

```python
# ✅ Good: Type hints and clear naming
def calculate_total_price(items: list[dict], tax_rate: float) -> float:
    subtotal = sum(item['price'] for item in items)
    return subtotal * (1 + tax_rate)
```

### Async/Concurrency
- Prefer `asyncio` for I/O-bound tasks.
- Avoid blocking calls inside async functions.

### Dependencies
- **Configuration:** Use `pyproject.toml` exclusively.
- **Management:** Use `uv`.
- **Imports:** Sort imports (isort standard).

## 3. Code Quality & Review

Refer to the `code_review` skill for detailed review guidelines and checklists.

### Essentials
- **Function Size:** Max 50 lines per function.
- **Complexity:** Avoid deep nesting.
- **DRY:** Avoid duplication.
- **Composability:** Favor composition over inheritance.

### Security
- **Input Validation:** Validate and sanitize all external input.
- **Secrets:** Never hardcode secrets. Use environment variables.
- **SQL/NoSQL:** Use parameterized queries.
- **Defaults:** Use secure defaults.
- **Best Practices:** Follow OWASP guidelines appropriate for the stack.

### Performance
- **Optimization:** Avoid premature optimization.
- **Loops:** Identify and fix existing N+1 query problems.
- **Resources:** Use context managers (`with` statements) for resource management.

## 4. Testing

- **Framework:** `pytest`
- **Goal:** At least 80% code coverage.
- **Requirement:** Every new feature must include tests.
- **Pattern:** Follow the Arrange–Act–Assert pattern.
- **Mocking:** Mock external dependencies.

## 5. Logging & Error Handling

- **Logging:** Use structured JSON logging in production.
- **Context:** Include correlation/request IDs where applicable.
- **Exceptions:** Catch specific exceptions (no bare `except:`).
- **Flow:** Fail fast for invalid inputs.
